"""A fabricated meeting with planted facts and known speakers, as text and (optionally) audio.

Usage:
  uv run python scripts/make_fake_meeting.py [--force]
      writes meetings/<stamp>_fake-q3-planning (transcript with the true speakers), used by
      scripts/qa_eval.py to score answers against the planted facts.
  uv run python scripts/make_fake_meeting.py --audio OUT.wav
      voices every line with a different macOS voice and writes OUT.wav (16 kHz mono) plus
      OUT.truth.json (speaker, start, end, text per line, in seconds from the start of the file),
      used by scripts/speaker_eval.py to score speaker identification.

A few lines carry transcription-style slips ("north wind") on purpose. The chair is never
named, so a correct system leaves them unidentified; "Karen, hold the comms" is followed by
the chair, not Karen — a hand-off cue that is wrong on purpose.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from meeting_assistant.config import Settings
from meeting_assistant.meeting import MeetingRecord

CHAIR = "Chair"  # never named in the meeting

# macOS voices (`say -v '?'`); Dev and Raj are both Indian-English male voices on purpose.
VOICES = {CHAIR: "Samantha", "Dev": "Rishi", "Raj": "Aman", "Karen": "Karen", "Paul": "Daniel", "Maya": "Tara"}

# (seconds since start, speaker, text)
LINES = [
    (0, CHAIR, "Okay, let's get going. Three things today: the database cutover, the north wind escalation, and hiring. Then the offsite if we have time."),
    (9, "Raj", "Hi everyone, Raj here, sorry I'm a couple of minutes late."),
    (12, CHAIR, "No problem. Dev, start with the cutover."),
    (16, "Dev", "Sure. The dry run on Tuesday went fine. Replication lag peaked at forty seconds, the target we agreed is under fifteen, so we need the replicas resized before we commit."),
    (31, CHAIR, "Raj, finance approved the bigger instances, right?"),
    (35, "Raj", "Yes, approved yesterday. It's an extra eighteen hundred dollars a month, which is inside the Q4 infra budget."),
    (44, CHAIR, "Great. Karen, can you file the resize ticket today?"),
    (48, "Karen", "Yes, I'll file it today and tag Dev on it."),
    (52, CHAIR, "So the date. We said November fourteenth for the cutover. Is that still realistic?"),
    (58, "Dev", "It is, if the replicas are resized by the seventh. I want a change freeze from November tenth, so nothing else lands that week."),
    (69, CHAIR, "Okay, cutover stays November fourteenth, freeze from the tenth. Everyone good? Good."),
    (76, "Dev", "One more thing on the migration. We go to Postgres sixteen, not seventeen. Seventeen isn't supported by the backup tooling yet, so we decided to stay on sixteen for now and revisit in Q1."),
    (92, CHAIR, "Noted, sixteen. Next, north wind. Karen?"),
    (97, "Karen", "They're seeing duplicate invoices after the nightly batch. It's happened twice this week. Support traced it to the exporter: it retries on timeout, but the first write had already succeeded."),
    (112, CHAIR, "Paul, you touched the exporter last month. Can you own the fix?"),
    (117, "Paul", "Yes, I'll take it. I think it's the retry wrapper, I'll have a fix up by November seventh so it's in before the freeze."),
    (126, CHAIR, "Thanks. And Dev, add an idempotency key to the export job regardless, so this can't happen again."),
    (133, "Dev", "Will do, I'll put that in the same release."),
    (137, "Karen", "Do we tell all customers or just north wind? I don't want to spook everyone over a single account issue."),
    (146, CHAIR, "Maya, that's your call. Can you decide by Wednesday so Karen can draft the comms?"),
    (152, "Maya", "Yes, I'll decide by Wednesday. Karen, hold the comms until then."),
    (158, CHAIR, "Hiring. We had two backend offers out, one accepted. Priya starts November third."),
    (166, CHAIR, "The second candidate, Tom, is asking to be fully remote. I'm fine with that as long as he overlaps at least four hours with the team."),
    (177, "Maya", "Agreed, remote is fine with the four hour overlap. I'll tell the recruiter today."),
    (183, CHAIR, "Last thing. The offsite moved to December first to third, in Lisbon. Budget is twelve thousand, Raj has signed it off."),
    (194, "Raj", "Flights should be booked by the end of October, so please get your requests in."),
    (200, CHAIR, "That's it. Thanks everyone."),
]

QUESTIONS = [
    ("When is the database cutover?", ["november 14", "nov 14", "fourteenth"]),
    ("When does the change freeze start?", ["november 10", "nov 10", "tenth"]),
    ("Who is fixing the Northwind duplicate invoices, and by when?", ["paul", "november 7", "seventh"]),
    ("What extra monthly cost did finance approve for the bigger replicas?", ["1,800", "1800", "eighteen hundred"]),
    ("Which Postgres version did we decide to migrate to, and why not the newer one?", ["16", "sixteen", "backup"]),
    ("What is the replication lag target?", ["15", "fifteen"]),
    ("Who files the replica resize ticket?", ["karen"]),
    ("When does the new hire start?", ["november 3", "nov 3", "third", "priya"]),
    ("Was remote work approved for the second candidate? Any condition?", ["yes", "four hour", "4 hour", "4-hour", "overlap"]),
    ("Where and when is the offsite, and what is the budget?", ["lisbon", "december 1", "dec 1", "12,000", "12000", "twelve thousand"]),
    ("Who decides whether to notify all customers, and by when?", ["maya", "wednesday"]),
    ("Who is adding the idempotency key to the export job?", ["dev"]),
    ("What was decided about the mobile app release?", ["not discussed", "no mention", "wasn't discussed", "was not discussed", "doesn't mention", "does not mention", "not mentioned", "no information"]),
]

GAP_SECONDS = 0.6  # pause between turns in the audio version


def write_text_meeting(force: bool) -> None:
    settings = Settings.load()
    root = settings.meetings_dir
    existing = [d for d in root.glob("*_fake-q3-planning") if d.is_dir()]
    if existing and not force:
        print(f"already exists: {existing[0]}  (pass --force to recreate)")
        return
    for d in existing:
        for f in d.iterdir():
            f.unlink()
        d.rmdir()

    start = datetime(2026, 10, 9, 10, 0).timestamp()
    rec = MeetingRecord.create(root, title="Fake: Q3 planning", stt_model="fabricated", llm_model="")
    rec.started = start
    rec.dir.rename(root / (rec.stamp + "_fake-q3-planning"))
    rec.dir = root / (rec.stamp + "_fake-q3-planning")
    with rec.path("transcript.jsonl").open("w") as jl, rec.path("transcript.md").open("w") as md:
        for offset, speaker, text in LINES:
            seg = {
                "start": start + offset,
                "end": start + offset + max(2.0, len(text.split()) / 2.5),
                "text": text,
                "speaker": speaker,  # ground truth: a name, not a cluster id
            }
            jl.write(json.dumps(seg) + "\n")
            md.write(f"[{datetime.fromtimestamp(seg['start']).strftime('%H:%M:%S')}] {speaker}: {text}\n")
    rec.finish()
    rec.ended = start + LINES[-1][0] + 5
    rec.save()
    print(f"created {rec.dir}  ({rec.words} words, {rec.segments} lines)")
    print(f"{len(QUESTIONS)} questions with expected answers in QUESTIONS")


def write_audio_meeting(out: Path) -> None:
    import numpy as np
    import soundfile as sf

    sr = 16000
    parts, truth, cursor = [], [], 0.0
    with tempfile.TemporaryDirectory() as tmp:
        for i, (_, speaker, text) in enumerate(LINES):
            path = Path(tmp) / f"{i}.wav"
            subprocess.run(["say", "-v", VOICES[speaker], "-o", str(path), "--data-format=LEI16@16000", text], check=True)
            audio, _ = sf.read(str(path), dtype="float32")
            truth.append({"speaker": speaker, "start": round(cursor, 3), "end": round(cursor + len(audio) / sr, 3), "text": text})
            parts += [audio, np.zeros(int(sr * GAP_SECONDS), dtype=np.float32)]
            cursor += len(audio) / sr + GAP_SECONDS
    full = np.concatenate(parts)
    sf.write(str(out), full, sr)
    out.with_suffix(".truth.json").write_text(json.dumps({"voices": VOICES, "lines": truth}, indent=1))
    print(f"wrote {out} ({len(full) / sr:.0f} s, {len(LINES)} lines, {len(VOICES)} speakers) and {out.with_suffix('.truth.json').name}")


if __name__ == "__main__":
    if "--audio" in sys.argv:
        write_audio_meeting(Path(sys.argv[sys.argv.index("--audio") + 1]))
    else:
        write_text_meeting(force="--force" in sys.argv)
