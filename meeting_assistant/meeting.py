from __future__ import annotations

import fcntl
import json
import re
import shutil
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

RECORD_FILE = "meeting.json"
STAMP_FORMAT = "%Y-%m-%d_%H%M%S"
STAMP_LEN = 17  # len("2026-10-10_131336")
ABANDONED_SECONDS = 12 * 3600  # an empty meeting folder this old is left over from a crash


def attendee_note(attendees: list[str], user: str = "") -> str:
    """Prompt line naming the people in the meeting, so the LLM spells misheard names right."""
    if not attendees:
        return ""
    names = ", ".join(f"{n} (the user)" if user and n.lower() == user.lower() else n for n in attendees)
    return (f"ATTENDEES: {names}. The transcript is automatic and may mangle these names "
            "(e.g. 'given' for 'Gavin'); always use these spellings.\n\n")


def slugify(title: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug[:max_len].rstrip("-")


def _claimed(record_file: Path) -> bool:
    with record_file.open("rb") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _has_speech(transcript: Path) -> bool:
    return transcript.exists() and transcript.stat().st_size > 0


def fmt_duration(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


@dataclass
class MeetingRecord:
    """One meeting = one folder of plain files, described by meeting.json.

    transcript.jsonl  exact sentences with timestamps (what the app reads back)
    transcript.md     the same, readable
    notes.md          last state of the running notes
    briefs.md         every brief that was generated
    summary.md        wrap-up written when the meeting ends
    """

    dir: Path
    title: str = ""
    started: float = field(default_factory=time.time)
    ended: float | None = None
    words: int = 0
    segments: int = 0
    stt_model: str = ""
    llm_model: str = ""
    attendees: list[str] = field(default_factory=list)

    # ----- identity -----
    @property
    def id(self) -> str:
        return self.dir.name

    @property
    def stamp(self) -> str:
        return datetime.fromtimestamp(self.started).strftime(STAMP_FORMAT)

    @property
    def date_str(self) -> str:
        return datetime.fromtimestamp(self.started).strftime("%Y-%m-%d %H:%M")

    @property
    def duration(self) -> float:
        return (self.ended or time.time()) - self.started

    def path(self, name: str) -> Path:
        return self.dir / name

    # ----- lifecycle -----
    @classmethod
    def create(cls, root: Path, title: str = "", **meta: str) -> "MeetingRecord":
        rec = cls(dir=root, title=title.strip(), **meta)
        root.mkdir(parents=True, exist_ok=True)
        name, n = rec._folder_name(), 1
        while True:  # two meetings started in the same second each get their own folder
            rec.dir = root / (name if n == 1 else f"{name}-{n}")
            try:
                rec.dir.mkdir()
                break
            except FileExistsError:
                n += 1
        rec.save()
        return rec

    @classmethod
    def load(cls, folder: Path) -> "MeetingRecord":
        f = folder / RECORD_FILE
        if f.exists():
            data = json.loads(f.read_text())
            known = {x.name for x in fields(cls)} - {"dir"}
            return cls(dir=folder, **{k: v for k, v in data.items() if k in known})
        # Folder from before meeting.json existed: derive what we can.
        try:
            started = datetime.strptime(folder.name[:STAMP_LEN], STAMP_FORMAT).timestamp()
        except ValueError:
            started = folder.stat().st_mtime
        rec = cls(dir=folder, started=started)
        rec.count_transcript()
        return rec

    @staticmethod
    def list_all(root: Path) -> list["MeetingRecord"]:
        """Meetings with something transcribed, newest first. A folder whose transcript is empty
        (the app was killed before anyone spoke) is left out, so it never becomes "latest"."""
        if not root.exists():
            return []
        recs = [MeetingRecord.load(d) for d in root.iterdir() if d.is_dir() and _has_speech(d / "transcript.jsonl")]
        return sorted(recs, key=lambda r: r.started, reverse=True)

    @staticmethod
    def prune_empty(root: Path, older_than: float = ABANDONED_SECONDS) -> list[Path]:
        """Delete folders left by runs that ended before anyone spoke (crash, kill, battery).

        Only a folder holding exactly meeting.json and an empty transcript, started more than
        `older_than` ago and not claimed by a running session (see `claim`) — so nothing with
        content, and never a meeting another window is still waiting in. Returns what went.
        """
        if not root.exists():
            return []
        removed = []
        for d in root.iterdir():
            files = {f.name: f for f in d.iterdir()} if d.is_dir() else {}
            if set(files) != {RECORD_FILE, "transcript.jsonl", "transcript.md"}:
                continue
            if files["transcript.jsonl"].stat().st_size or files["transcript.md"].stat().st_size:
                continue
            try:
                started = float(json.loads(files[RECORD_FILE].read_text()).get("started", 0))
            except (ValueError, TypeError, AttributeError):
                continue
            if time.time() - started > older_than and not _claimed(files[RECORD_FILE]):
                shutil.rmtree(d, ignore_errors=True)
                removed.append(d)
        return removed

    def claim(self):
        """Mark this meeting as in use until the returned file is closed (or the process ends),
        so another window's `prune_empty` leaves it alone however long it stays quiet."""
        f = (self.dir / RECORD_FILE).open("rb")
        fcntl.flock(f, fcntl.LOCK_SH)
        return f

    def save(self) -> None:
        data = {k: v for k, v in asdict(self).items() if k != "dir"}
        (self.dir / RECORD_FILE).write_text(json.dumps(data, indent=2) + "\n")

    def count_transcript(self) -> None:
        f = self.dir / "transcript.jsonl"
        if not f.exists():
            return
        lines = [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
        self.segments = len(lines)
        self.words = sum(len(s["text"].split()) for s in lines)
        if lines and self.ended is None:
            self.ended = max(self.ended or 0, lines[-1]["end"])

    def finish(self, title: str | None = None) -> "MeetingRecord":
        """Close the record: final counts, end time, title, and a readable folder name."""
        if title is not None:
            self.title = title.strip()
        self.count_transcript()
        self.ended = time.time()
        self.save()
        target = self.dir.parent / self._folder_name()
        if target != self.dir and not target.exists():
            self.dir.rename(target)
            self.dir = target
        return self

    def _folder_name(self) -> str:
        slug = slugify(self.title)
        return self.stamp + (f"_{slug}" if slug else "")
