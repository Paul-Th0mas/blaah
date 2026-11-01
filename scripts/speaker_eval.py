"""Score speaker identification on a recording with known speakers.

Usage:  uv run python scripts/speaker_eval.py AUDIO.wav TRUTH [--lines 40]
  TRUTH is either an RTTM file (who spoke when; e.g. the AMI corpus annotations) or the
  .truth.json written by `make_fake_meeting.py --audio`.

Runs the real pipeline — Parakeet, word timing, voiceprints, clustering, name cues — on the
file as fast as possible (no LLM), then reports:
  voice purity    share of speech time whose voice cluster maps to one true speaker
  naming          which names each true speaker ended up with
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from meeting_assistant.audio import FileCapture
from meeting_assistant.config import Settings
from meeting_assistant.speakers import SpeakerTracker, VoiceEmbedder
from meeting_assistant.transcriber import Transcriber
from meeting_assistant.transcript import TranscriptStore


def load_truth(path: Path) -> list[tuple[float, float, str]]:
    if path.suffix == ".json":
        return [(l["start"], l["end"], l["speaker"]) for l in json.loads(path.read_text())["lines"]]
    rows = []
    for line in path.read_text().splitlines():
        p = line.split()
        if p and p[0] == "SPEAKER":
            rows.append((float(p[3]), float(p[3]) + float(p[4]), p[7]))
    return sorted(rows)


def truth_speaker(truth, t0: float, t1: float) -> str | None:
    overlap = Counter()
    for s, e, spk in truth:
        o = min(e, t1) - max(s, t0)
        if o > 0:
            overlap[spk] += o
    return overlap.most_common(1)[0][0] if overlap else None


def main() -> None:
    audio_path, truth_path = Path(sys.argv[1]), Path(sys.argv[2])
    show = int(sys.argv[sys.argv.index("--lines") + 1]) if "--lines" in sys.argv else 40
    settings = Settings.load()
    truth = load_truth(truth_path)

    store = TranscriptStore()
    tracker = SpeakerTracker(settings.speakers, settings.user.names, embedder=VoiceEmbedder(settings.speakers.model))
    store.namer = tracker.display
    source = FileCapture(audio_path, settings.audio, realtime=False)
    tr = Transcriber(settings.stt, source, store, speakers=tracker)
    t0 = time.perf_counter()
    source.start()
    tr.start()
    source.finished.wait()
    source.stop()
    tr.finish(timeout=600)
    elapsed = time.perf_counter() - t0
    print(f"processed {source.duration / 60:.1f} min of audio in {elapsed:.0f} s; {len(store)} sentences, {len(tracker.voices)} voices\n")

    by_voice: dict[str, Counter] = defaultdict(Counter)
    by_truth: dict[str, Counter] = defaultdict(Counter)
    total = scored = correct_named = 0.0
    for seg in store.segments():
        a, b = seg.start - tr.stream_start, seg.end - tr.stream_start
        dur = max(b - a, 0.3)
        spk = truth_speaker(truth, a, b)
        if spk is None or not seg.speaker:
            continue
        total += dur
        vid = tracker.resolve(seg.speaker)
        by_voice[vid][spk] += dur
        name = tracker.display(seg.speaker)
        by_truth[spk][name] += dur
        if spk == name or (spk.lower() == "chair" and name.startswith("Speaker")):
            correct_named += dur
        scored += dur

    purity = sum(c.most_common(1)[0][1] for c in by_voice.values()) / total if total else 0
    print(f"voice purity: {purity:.0%} of speech time is in a voice that maps to one true speaker")
    for vid, c in sorted(by_voice.items(), key=lambda kv: -sum(kv[1].values())):
        top, d = c.most_common(1)[0]
        print(f"  {vid:4s} {tracker.display(vid):12s} {sum(c.values()):6.0f}s  mostly {top} ({d / sum(c.values()):.0%})")
    print("\nnames given to each true speaker (by speech time):")
    for spk, c in sorted(by_truth.items()):
        parts = ", ".join(f"{n} {d / sum(c.values()):.0%}" for n, d in c.most_common(3))
        print(f"  {spk:8s} {parts}")
    if truth_path.suffix == ".json":
        print(f"\nnaming accuracy: {correct_named / scored:.0%} of speech time shows the right name")
    print(f"\nevidence: {json.dumps({v: s['evidence'] for v, s in tracker.snapshot().items() if v != '_merged'}, indent=None)}")
    print(f"\n--- first {show} lines ---")
    for seg in store.segments()[:show]:
        print(store.line(seg))


if __name__ == "__main__":
    main()
