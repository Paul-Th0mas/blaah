"""Fix who-said-what in a finished meeting: set attendees, re-read the cues, name voices by hand.

Works from the saved files only (transcript.jsonl keeps every sentence's voice id), so no audio
is needed. Voice grouping itself can't change after the fact; names can.
"""

from __future__ import annotations

import json

from .ask import load_segments
from .config import Settings
from .meeting import MeetingRecord
from .speakers import SpeakerTracker
from .transcript import Segment, TranscriptStore


def load_naming(record: MeetingRecord, settings: Settings) -> tuple[SpeakerTracker, list[Segment], dict]:
    segments = load_segments(record)
    f = record.path("speakers.json")
    saved = json.loads(f.read_text()) if f.exists() else {}
    tracker = SpeakerTracker.from_meeting(settings.speakers, settings.user.names, segments, saved)
    if record.attendees and not tracker.attendees:
        tracker.set_attendees(record.attendees)
    return tracker, segments, saved


def save_naming(record: MeetingRecord, tracker: SpeakerTracker, segments: list[Segment], saved: dict) -> None:
    """Write speakers.json and transcript.md with the current names; keep live-only stats."""
    snap = tracker.snapshot()
    for vid, info in snap.items():
        if not vid.startswith("_") and isinstance(saved.get(vid), dict):
            info["seconds"] = saved[vid].get("seconds", info["seconds"])
            info["sentences"] = saved[vid].get("sentences", info["sentences"])
    record.path("speakers.json").write_text(json.dumps(snap, indent=1) + "\n")
    store = TranscriptStore()
    store.namer = tracker.display
    record.path("transcript.md").write_text(store.render(segments) + "\n")
    record.attendees = list(tracker.attendees)
    record.save()


def describe(tracker: SpeakerTracker, width: int = 90) -> str:
    """One line per voice: id, current name, sentence count, an example of what they said."""
    lines = []
    for vid, name, count, latest in tracker.voice_summary():
        tag = " (set by hand)" if vid in tracker.manual else ""
        example = latest if len(latest) <= width else latest[: width - 1] + "…"
        lines.append(f"  {vid:4s} {name + tag:24s} {count:4d} sentences   “{example}”")
    return "\n".join(lines)
