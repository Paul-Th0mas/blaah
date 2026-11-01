from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path

RECORD_FILE = "meeting.json"
STAMP_FORMAT = "%Y-%m-%d_%H%M%S"
STAMP_LEN = 17  # len("2026-10-10_131336")


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
        rec.dir = root / rec._folder_name()
        rec.dir.mkdir(parents=True, exist_ok=True)
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
