from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from .config import TrackerCfg
from .providers import Provider
from .meeting import attendee_note
from .transcript import TranscriptStore

NOTES_SYSTEM = """You are a silent note-taker following a live meeting through an automatic transcript.
You maintain one compact set of running notes. Each time you are given the current notes plus
new transcript, rewrite the notes so they stay current: merge, update or drop stale items,
keep what still matters. The transcript is machine-generated and may contain mis-heard words;
infer the likely meaning and never comment on transcription quality.

Rules:
- Output ONLY the notes in exactly this markdown structure, no preamble or commentary.
- Be concrete: names, numbers, dates, systems. No filler.
- Keep the whole thing under 250 words.
- Lines may start with who said them ("Dev: …"); "Speaker 2" is a voice not yet named. Speaker
  labels come from voice matching and can be wrong; prefer names said in the words themselves.
- Attribute an ask or action to a person only when the transcript names them (as speaker or in
  the words); otherwise write "unassigned". Never guess that something is for {names}.
- Mark anything explicitly asked of, or assigned to, {names} with the prefix ⚠ — never an unassigned item.

## Now discussing
<one line: the current topic and where the discussion is heading>
## Key points
- <3-7 bullets, most recent last>
## Decisions
- <or "none yet">
## Open questions & asks
- <who asked what; unanswered first>
## Action items
- <owner — task — when>
"""

NOTES_PROMPT = """CURRENT NOTES:
{notes}

NEW TRANSCRIPT ({span}):
{transcript}

Rewrite the notes."""

EMPTY_NOTES = "(no notes yet — the meeting has just started)"
MAX_TRANSCRIPT_WORDS = 1200  # keeps the prompt + reply comfortably inside a 4k context


class Tracker(threading.Thread):
    """Background summariser: every interval, folds new transcript into the running notes."""

    def __init__(
        self,
        cfg: TrackerCfg,
        llm: Provider,
        store: TranscriptStore,
        user_names: list[str],
        max_tokens: int,
        session_dir: Path | None = None,
        on_update: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        defer: Callable[[], str] | None = None,
    ):
        super().__init__(name="tracker", daemon=True)
        self.cfg = cfg
        self.llm = llm
        self.store = store
        self.max_tokens = max_tokens
        self.set_user_names(user_names)
        self.attendees: list[str] = []  # set by the session; may change mid-meeting
        self.session_dir = session_dir
        self.on_update = on_update or (lambda _n: None)
        self.on_status = on_status or (lambda _s: None)
        # Why scheduled refreshes should wait ("" = don't), e.g. transcription is behind and the
        # GPU goes to speech-to-text first. A forced refresh still runs.
        self.defer = defer or (lambda: "")
        self.notes = EMPTY_NOTES
        self.updated_at: float | None = None
        self.last_latency = 0.0
        self._cursor = 0
        self._backlog = False  # transcript left over from the last update, folded in next
        self._kick = threading.Event()
        self._stopped = threading.Event()  # not `_stop`: that shadows Thread._stop(), which join() calls
        self._busy = threading.Lock()
        # Set by Session while a brief is being generated: a brief must never wait
        # behind a notes refresh, so an in-flight update aborts and retries later.
        self.yield_to_brief = threading.Event()

    def set_user_names(self, user_names: list[str]) -> None:
        self.system = NOTES_SYSTEM.format(names=" / ".join(user_names) or "the user")
        self.user = user_names[0] if user_names else ""

    def stop(self) -> None:
        self._stopped.set()
        self._kick.set()

    def refresh_now(self) -> None:
        self._kick.set()

    def run(self) -> None:
        while not self._stopped.is_set():
            forced = self._kick.wait(1.0 if self._backlog else self.cfg.interval_seconds)
            self._kick.clear()
            if self._stopped.is_set():
                break
            self.update(force=forced)

    def update(self, force: bool = False) -> bool:
        """Fold new transcript into the notes. Returns True if more is left for another update."""
        if self.yield_to_brief.is_set():
            return False  # a brief is running; the next interval will catch up
        wait = "" if force else self.defer()
        if wait:
            self.on_status(f"notes waiting — {wait}")
            return False
        if not self._busy.acquire(blocking=force):
            return False  # periodic tick while an update runs; a forced one waits instead
        try:
            new, _ = self.store.since_index(self._cursor)
            # A long backlog (notes failed or waited for a while) goes in pieces, oldest first,
            # so no stretch of the meeting is skipped: take what fits now, the rest next time.
            batch: list = []
            words = 0
            for seg in new:
                n = len(seg.text.split())
                if batch and words + n > MAX_TRANSCRIPT_WORDS:
                    break
                batch.append(seg)
                words += n
            if not batch or (words < self.cfg.min_new_words and not force):
                self._backlog = False
                return False
            new, cursor, more = batch, self._cursor + len(batch), len(batch) < len(new)
            transcript = self.store.render(new)
            if words > MAX_TRANSCRIPT_WORDS:  # one giant segment (a hand-made transcript): keep its end
                transcript = "… " + " ".join(transcript.split()[-MAX_TRANSCRIPT_WORDS:])
            span = f"{new[0].clock}–{new[-1].clock}, {words} words"
            prompt = attendee_note(self.attendees, self.user) + NOTES_PROMPT.format(
                notes=self.notes, span=span, transcript=transcript
            )

            self.on_status("updating notes…")
            t0 = time.perf_counter()
            parts: list[str] = []
            gen = self.llm.stream(self.system, prompt, max_tokens=self.max_tokens)
            try:
                for delta in gen:
                    if self.yield_to_brief.is_set():
                        self.on_status("notes paused for brief")
                        return False  # closing the stream makes LM Studio abort generation
                    parts.append(delta)
            except Exception as e:  # LLMError, or whatever a provider plugin raises
                # Notes and cursor stay as they were, so this stretch is retried next time.
                self.on_status(f"notes failed: {e}")
                return False
            finally:
                gen.close()
            self.last_latency = time.perf_counter() - t0
            text = "".join(parts).strip()
            if not text:
                return False
            self.notes = text
            self.updated_at = time.time()
            self._cursor = cursor
            self._backlog = more
            self._save()
            self.on_update(self.notes)
            tps = float(self.llm.last_stats.get("tokens_per_second", 0.0))
            self.on_status(f"notes updated in {self.last_latency:.1f}s ({tps:.0f} tok/s)" + (", more to fold in" if more else ""))
            return more
        finally:
            self._busy.release()

    def _save(self) -> None:
        if self.session_dir:
            (self.session_dir / "notes.md").write_text(self.notes + "\n")
