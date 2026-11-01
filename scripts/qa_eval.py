"""Ask the planted-fact questions about the fake meeting and score the answers.

Usage:  uv run python scripts/qa_eval.py [model-key ...]      (default: the model in config.toml)
Run scripts/make_fake_meeting.py first.
"""

from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_fake_meeting import QUESTIONS  # noqa: E402

from meeting_assistant.ask import ask  # noqa: E402
from meeting_assistant.config import Settings  # noqa: E402
from meeting_assistant.providers import make_llm  # noqa: E402
from meeting_assistant.meeting import MeetingRecord  # noqa: E402


def main() -> None:
    settings = Settings.load()
    record = next((r for r in MeetingRecord.list_all(settings.meetings_dir) if r.id.endswith("_fake-q3-planning")), None)
    if record is None:
        sys.exit("fake meeting not found — run scripts/make_fake_meeting.py first")
    models = sys.argv[1:] or [settings.llm.model]
    for model in models:
        llm = make_llm(replace(settings.llm, model=model))
        llm.ensure_loaded(lambda m: print(f"  ({m})", flush=True))
        print(f"\n{'=' * 78}\nMODEL {model}  —  {record.id} ({record.words} words)\n{'=' * 78}")
        passed, total_time = 0, 0.0
        for question, expected in QUESTIONS:
            t = time.perf_counter()
            answer = ask(llm, record, question)
            dt = time.perf_counter() - t
            total_time += dt
            ok = any(e in answer.lower() for e in expected)
            passed += ok
            print(f"\n{'PASS' if ok else 'FAIL'} ({dt:.1f}s)  Q: {question}\n            A: {answer}")
            if not ok:
                print(f"            expected one of: {expected}")
        print(f"\nSCORE {passed}/{len(QUESTIONS)}  —  avg {total_time / len(QUESTIONS):.1f}s per answer")


if __name__ == "__main__":
    main()
