from __future__ import annotations

from .providers import Provider
from .meeting import MeetingRecord, attendee_note, fmt_duration

SUMMARY_SYSTEM = """You write the wrap-up for a meeting that has just ended, from its running notes and the
last part of an automatic transcript (mis-heard words are possible; infer the meaning silently).
Lines may start with who said them ("Dev: …"; "Speaker 2" = a voice never named). Attribute
nothing to a person the transcript does not name; write "unassigned" instead.
Output ONLY markdown in exactly this shape, under 300 words, no commentary:

# <a short, specific title for the meeting — max 8 words, no quotes>
**TL;DR:** <2–3 sentences>
## Decisions
- <or "none">
## Action items
- <owner — task — when; prefix with ⚠ only items explicitly for {names}, never unassigned ones>
## Open questions
- <or "none">
## Topics covered
- <3–8 bullets, in order>
"""

SUMMARY_PROMPT = """Meeting started {date} and lasted {duration}.

RUNNING NOTES (final state):
{notes}

TRANSCRIPT, LAST PART:
{transcript}

Write the wrap-up."""

MAX_TRANSCRIPT_WORDS = 1200


def write_summary(llm: Provider, record: MeetingRecord, notes: str, transcript: str, names: list[str], max_tokens: int = 600) -> str:
    """Generate summary.md for the meeting; returns the model's suggested title."""
    words = transcript.split()
    if len(words) > MAX_TRANSCRIPT_WORDS:
        transcript = "… " + " ".join(words[-MAX_TRANSCRIPT_WORDS:])
    prompt = attendee_note(record.attendees, names[0] if names else "") + SUMMARY_PROMPT.format(
        date=record.date_str, duration=fmt_duration(record.duration), notes=notes, transcript=transcript
    )
    text = llm.complete(SUMMARY_SYSTEM.format(names=" / ".join(names) or "the user"), prompt, max_tokens=max_tokens).text.strip()
    title, body = split_title(text)
    heading = f"# {record.title or title or 'Meeting'}\n_{record.date_str} · {fmt_duration(record.duration)}_\n\n"
    record.path("summary.md").write_text(heading + body.strip() + "\n")
    return title


def split_title(markdown: str) -> tuple[str, str]:
    """Pull the leading '# Title' line off the model's output."""
    lines = markdown.strip().splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip().strip('"'), "\n".join(lines[1:])
    return "", markdown
