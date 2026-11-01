from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

from .providers import Provider
from .meeting import attendee_note

BRIEF_SYSTEM = """You are a discreet in-ear assistant for {user}, who is in a live meeting and may have
lost the thread. Someone may have just said their name or asked them something. Using the running
notes and the last few minutes of machine transcript, tell them what is going on and what they
could say — fast. Transcript words may be mis-heard; infer the likely meaning silently.
Lines may start with who said them ("Dev: …", "Speaker 2" = not yet named); use the name when
you say who asked. Never invent project facts, numbers, dates or commitments. If the context does not contain
the answer, make option 1 an honest holding line ("Let me confirm and come back to you right
after this") and keep the clarifying question.

Output exactly this markdown, nothing else, under 130 words total:

**Right now:** <one line — the topic and where it is heading>
**Last said:** <the last 1–2 things said, paraphrased; if a question was put to {user}, quote it>
**You could say:**
- <option 1: a direct, safe answer or contribution in {user}'s voice>
- <option 2>
- <if the context truly does not answer the question: a clarifying question to buy time>
"""

BRIEF_PROMPT = """Trigger: {reason}

RUNNING NOTES:
{notes}

TRANSCRIPT, LAST {window:.0f} SECONDS (newest last; "[live]" is still being transcribed):
{recent}

Respond now."""


def brief_stream(
    llm: Provider,
    *,
    user_names: list[str],
    attendees: list[str] | None = None,
    notes: str,
    recent: str,
    window_seconds: float,
    reason: str,
    max_tokens: int,
    session_dir: Path | None = None,
) -> Iterator[str]:
    """Yield the brief as it is generated; append the finished brief to briefs.md."""
    user = user_names[0] if user_names else "the user"
    system = BRIEF_SYSTEM.format(user=user)
    prompt = attendee_note(attendees or [], user) + BRIEF_PROMPT.format(
        reason=reason,
        notes=notes,
        window=window_seconds,
        recent=recent or "(nothing transcribed yet)",
    )
    parts: list[str] = []
    for delta in llm.stream(system, prompt, max_tokens=max_tokens):
        parts.append(delta)
        yield delta
    if session_dir:
        stamp = datetime.now().strftime("%H:%M:%S")
        with (session_dir / "briefs.md").open("a") as f:
            f.write(f"## {stamp} — {reason}\n\n{''.join(parts).strip()}\n\n")
