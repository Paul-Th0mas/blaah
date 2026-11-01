from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable


@dataclass
class Segment:
    start: float  # wall-clock epoch seconds
    end: float
    text: str
    speaker: str = ""  # voice id ("S2"), or a name in hand-made transcripts; "" = unknown
    source: str = ""  # "me" (your mic), "others" (the call), "" (can't tell / not a call)
    levels: list[float] | None = None  # mean [mic, call] loudness, for tuning the source check

    @property
    def clock(self) -> str:
        return datetime.fromtimestamp(self.start).strftime("%H:%M:%S")


class TranscriptStore:
    """Thread-safe log of finalised sentences plus the current live (not yet final) text.

    Segments store a stable voice id; `namer` turns it into what people read ("Dev",
    "Speaker 3"). Names can arrive after the fact, so rendering always uses the current
    namer, and transcript.md is rewritten whenever names change.
    """

    def __init__(self, session_dir: Path | None = None):
        self._segments: list[Segment] = []
        self._live = ""
        self._lock = threading.Lock()
        self._listeners: list[Callable[[Segment], None]] = []
        self.namer: Callable[[str], str] = lambda speaker: speaker
        self._dir = session_dir
        self._file = None
        self._readable = None
        if session_dir is not None:
            session_dir.mkdir(parents=True, exist_ok=True)
            self._file = (session_dir / "transcript.jsonl").open("a")  # exact, for the app
            self._readable = (session_dir / "transcript.md").open("a")  # for humans

    # ----- writers (transcriber thread) -----
    def add(self, seg: Segment) -> None:
        with self._lock:
            self._segments.append(seg)
            if self._file:
                self._file.write(json.dumps(asdict(seg)) + "\n")
                self._file.flush()
            if self._readable:
                self._readable.write(self.line(seg) + "\n")
                self._readable.flush()
        for cb in self._listeners:
            cb(seg)

    def set_live(self, text: str) -> None:
        with self._lock:
            self._live = text

    def rewrite_markdown(self) -> None:
        """Re-render transcript.md with the current names (they may have been learned late)."""
        if self._dir is None:
            return
        with self._lock:
            text = "".join(self.line(s) + "\n" for s in self._segments)
            if self._readable:
                self._readable.close()
            path = self._dir / "transcript.md"
            path.write_text(text)
            self._readable = path.open("a") if self._file else None

    # ----- readers (UI / tracker / brief) -----
    def on_segment(self, cb: Callable[[Segment], None]) -> None:
        self._listeners.append(cb)

    @property
    def live(self) -> str:
        with self._lock:
            return self._live

    def __len__(self) -> int:
        with self._lock:
            return len(self._segments)

    def segments(self) -> list[Segment]:
        with self._lock:
            return list(self._segments)

    def since_index(self, idx: int) -> tuple[list[Segment], int]:
        """Segments added after position idx, and the new high-water mark."""
        with self._lock:
            new = self._segments[idx:]
            return new, len(self._segments)

    def who(self, seg: Segment) -> str:
        return self.namer(seg.speaker) if seg.speaker else ""

    def line(self, seg: Segment) -> str:
        who = self.who(seg)
        return f"[{seg.clock}] {who}: {seg.text}" if who else f"[{seg.clock}] {seg.text}"

    def recent(self, seconds: float, include_live: bool = True) -> str:
        """Timestamped transcript for the last N seconds, newest last."""
        cutoff = time.time() - seconds
        with self._lock:
            lines = [self.line(s) for s in self._segments if s.end >= cutoff]
            live = self._live
        if include_live and live.strip():
            lines.append(f"[live] {live.strip()}")
        return "\n".join(lines)

    def render(self, segments: list[Segment]) -> str:
        return "\n".join(self.line(s) for s in segments)

    def close(self) -> None:
        with self._lock:
            for f in (self._file, self._readable):
                if f:
                    f.close()
            self._file = self._readable = None
