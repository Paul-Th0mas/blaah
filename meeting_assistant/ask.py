from __future__ import annotations

import json
import re

from .providers import Provider
from .meeting import MeetingRecord, attendee_note
from .speakers import load_namer
from .transcript import Segment, TranscriptStore

ASK_SYSTEM = """You answer questions about one meeting, using only its transcript (automatic, so words may be
mis-heard — infer the likely meaning) and its summary if one is given.
Lines may start with who said them ("Dev: …"); "Speaker 2" is a voice that was never named.
Rules:
- Answer in one to three sentences, specific: names, dates, numbers, owners.
- Cite the time of the transcript line you relied on in brackets, e.g. [10:02:15].
- If the meeting does not contain the answer, say "Not discussed in this meeting." and stop. Never guess.
"""

ASK_PROMPT = """MEETING: {title} ({date})

SUMMARY:
{summary}

TRANSCRIPT{scope}:
{transcript}

QUESTION: {question}"""

MAX_WORDS = 2200  # whole transcript fits a 4k context below this; above it we send an excerpt
STOPWORDS = set(
    "a an the and or of to in on for with by at from is are was were be been being do does did "
    "what when who whom whose which why how where did any about we our us they them it its this that "
    "these those there here have has had will would should could can may might must not no yes".split()
)


def load_segments(record: MeetingRecord) -> list[Segment]:
    path = record.path("transcript.jsonl")
    return [Segment(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]


def ask(llm: Provider, record: MeetingRecord, question: str, max_tokens: int = 300) -> str:
    segments = load_segments(record)
    store = TranscriptStore()
    store.namer = load_namer(record.dir)
    total_words = sum(len(s.text.split()) for s in segments)
    if total_words <= MAX_WORDS:
        transcript, scope = store.render(segments), ""
    else:
        transcript, scope = relevant_excerpt(segments, question, MAX_WORDS, store), " (relevant excerpts)"
    summary_path = record.path("summary.md")
    prompt = attendee_note(record.attendees) + ASK_PROMPT.format(
        title=record.title or "(untitled)",
        date=record.date_str,
        summary=summary_path.read_text().strip() if summary_path.exists() else "(none)",
        scope=scope,
        transcript=transcript or "(empty)",
        question=question.strip(),
    )
    return llm.complete(ASK_SYSTEM, prompt, max_tokens=max_tokens).text.strip()


def relevant_excerpt(segments: list[Segment], question: str, max_words: int, store: TranscriptStore | None = None) -> str:
    """Keyword-scored selection of lines (plus their neighbours), in meeting order, for long transcripts."""
    store = store or TranscriptStore()
    terms = {w for w in re.findall(r"[a-z0-9]+", question.lower()) if w not in STOPWORDS and len(w) > 2}
    scored = []
    for i, seg in enumerate(segments):
        words = set(re.findall(r"[a-z0-9]+", (store.who(seg) + " " + seg.text).lower()))
        hits = sum(1 for t in terms if t in words or any(w.startswith(t) for w in words))
        if hits:
            scored.append((hits, i))
    keep: set[int] = set()
    budget = max_words
    for _, i in sorted(scored, reverse=True):
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(segments) and j not in keep:
                cost = len(segments[j].text.split())
                if cost <= budget:
                    keep.add(j)
                    budget -= cost
    lines, last = [], None
    for j in sorted(keep):
        if last is not None and j != last + 1:
            lines.append("…")
        lines.append(store.line(segments[j]))
        last = j
    return "\n".join(lines)
