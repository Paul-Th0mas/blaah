"""A/B harness: replay a recorded transcript through notes + brief for several LM Studio models.

Usage:  uv run python scripts/compare_models.py sessions/<run>/transcript.jsonl model-key [model-key ...]

Same input for every model, Parakeet kept resident so memory conditions match the real app,
each model loaded alone (others unloaded first). Prints notes, brief, and timings per model.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
from parakeet_mlx import from_pretrained

from meeting_assistant.brief import brief_stream
from meeting_assistant.config import Settings
from meeting_assistant.providers import make_llm
from meeting_assistant.tracker import Tracker
from meeting_assistant.transcript import Segment, TranscriptStore

SPLIT = 0.6  # first 60 % of the transcript feeds the notes; the rest is "recent" for the brief
# Replay stops right after the line containing this, i.e. the moment the user's name is called.
STOP_AT = "your read"


def load_segments(path: Path) -> list[Segment]:
    return [Segment(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    transcript, *models = sys.argv[1:]
    settings = Settings.load()
    segments = load_segments(Path(transcript))
    stop = next((i for i, s in enumerate(segments) if STOP_AT in s.text.lower()), len(segments) - 1)
    segments = segments[: stop + 1]
    cut = int(len(segments) * SPLIT)
    print(f"{len(segments)} segments replayed; notes get {cut}, the brief sees the last {len(segments) - cut} "
          f"(ending: {segments[-1].text!r})")

    print("loading Parakeet so memory conditions match the real app…", flush=True)
    stt = from_pretrained(settings.stt.model)
    with stt.transcribe_stream() as s:
        s.add_audio(mx.zeros(16000))

    for model in models:
        print(f"\n{'=' * 78}\nMODEL {model}\n{'=' * 78}", flush=True)
        subprocess.run(["lms", "unload", "--all"], capture_output=True)
        llm = make_llm(replace(settings.llm, model=model))
        t = time.perf_counter()
        llm.ensure_loaded(lambda m: print(f"  ({m})", flush=True))
        print(f"  load+warmup {time.perf_counter() - t:.1f}s, reasoning_off={llm.reasoning_off}", flush=True)

        store = TranscriptStore()
        for seg in segments[:cut]:
            store.add(seg)
        tracker = Tracker(settings.tracker, llm, store, settings.user.names, settings.llm.max_notes_tokens)
        tracker.update(force=True)
        st = llm.last_stats
        print(f"\n--- NOTES ({tracker.last_latency:.1f}s, ttft {st.get('time_to_first_token_seconds', 0):.1f}s, "
              f"{st.get('tokens_per_second', 0):.0f} tok/s, {st.get('input_tokens', 0)} in / {st.get('total_output_tokens', 0)} out)")
        print(tracker.notes)

        for seg in segments[cut:]:
            store.add(seg)
        recent = store.render(segments[cut:])
        t = time.perf_counter()
        first = None
        out = []
        for delta in brief_stream(llm, user_names=settings.user.names, notes=tracker.notes, recent=recent,
                                  window_seconds=settings.tracker.recent_window_seconds,
                                  reason='heard "Paul"', max_tokens=settings.llm.max_brief_tokens):
            first = first or time.perf_counter()
            out.append(delta)
        total = time.perf_counter() - t
        st = llm.last_stats
        print(f"\n--- BRIEF (first word {first - t:.1f}s, total {total:.1f}s, {st.get('tokens_per_second', 0):.0f} tok/s, "
              f"{st.get('input_tokens', 0)} in / {st.get('total_output_tokens', 0)} out)")
        print("".join(out).strip())
        sw = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
        print(f"\n  swap: {sw}")


if __name__ == "__main__":
    main()
