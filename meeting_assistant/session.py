from __future__ import annotations

import json
import shutil
import threading
from typing import Callable, Iterator

from .audio import FileCapture, MicCapture, MixedCapture, make_capture
from .brief import brief_stream
from .config import Settings
from .llm import LLMError
from .providers import make_llm
from .meeting import MeetingRecord
from .notify import notify
from .speakers import SpeakerTracker, VoiceEmbedder, load_voiceprints, save_voiceprint
from .summary import write_summary
from .tracker import Tracker
from .transcriber import Transcriber
from .transcript import Segment, TranscriptStore
from .triggers import NameDetector

BEHIND_DEFER_SECONDS = 5.0  # notes refreshes wait while transcription is further behind than this
BRIEF_WAIT_SECONDS = 60.0  # wrap-up waits this long for a running brief before the final notes


class Session:
    """Wires mic → Parakeet → transcript → name detector / tracker → briefs, for one meeting.

    UI-agnostic: a front end assigns the `on_*` callbacks (all invoked from
    background threads), calls `brief()` when it wants one, and ends the meeting
    with `wrap_up()` then `complete(title)`.
    """

    def __init__(
        self,
        settings: Settings,
        audio: MicCapture | FileCapture | None = None,
        title: str = "",
        attendees: list[str] | None = None,
    ):
        self.settings = settings
        self.record = MeetingRecord.create(
            settings.meetings_dir, title=title, stt_model=settings.stt.model, llm_model=settings.llm.model
        )
        self.session_dir = self.record.dir

        self.store = TranscriptStore(self.session_dir)
        self.audio = audio or make_capture(settings.audio)
        self.llm = make_llm(settings.llm)
        self.llm_ready = False
        self._closed = False
        self.detector = NameDetector(settings.user.names, settings.triggers.name_cooldown_seconds)
        self._live_hits = 0  # mentions of the user's name in the live text at the last check
        self._briefs = 0  # briefs being generated right now
        self._briefs_done = threading.Condition()

        # Front-end hooks. `on_status(component, message)` components: stt, llm, notes, voices.
        self.on_segment: Callable[[Segment], None] = lambda _s: None
        self.on_live: Callable[[str], None] = lambda _t: None
        self.on_notes: Callable[[str], None] = lambda _n: None
        self.on_status: Callable[[str, str], None] = lambda _c, _m: None
        self.on_trigger: Callable[[str], None] = lambda _r: None
        self.on_speakers: Callable[[dict[str, str]], None] = lambda _names: None

        self.speakers: SpeakerTracker | None = None
        if settings.speakers.enabled:
            self.speakers = SpeakerTracker(settings.speakers, settings.user.names, on_change=self._speakers_changed)
            self.store.namer = self.speakers.display
        self._speakers_lock = threading.Lock()

        self.transcriber = Transcriber(
            settings.stt,
            self.audio,
            self.store,
            on_status=lambda m: self.on_status("stt", m),
            on_live=self._live,
            speakers=self.speakers,
        )
        self.tracker = Tracker(
            settings.tracker,
            self.llm,
            self.store,
            user_names=settings.user.names,
            max_tokens=settings.llm.max_notes_tokens,
            session_dir=self.session_dir,
            on_update=lambda n: self.on_notes(n),
            on_status=lambda m: self.on_status("notes", m),
            defer=self._notes_wait,
        )
        self.store.on_segment(self._segment)
        if attendees:
            self.set_attendees(attendees)

    def start(self) -> None:
        """Blocking: starts audio + STT immediately, then brings the model up (may take a while)."""
        try:
            self.audio.start()
        except RuntimeError as e:
            if not isinstance(self.audio, MixedCapture):
                raise
            # System-audio tap unavailable (no toolchain, helper failed): fall back to mic only.
            self.on_status("stt", f"system audio off: {e}")
            self.audio = MicCapture(self.settings.audio)
            self.transcriber.audio = self.audio
            self.audio.start()
        if hasattr(self.audio, "on_switch"):  # headphones in/out, devices reconnecting
            self.audio.on_switch = lambda message: self.on_status("audio", message)
            if getattr(self.audio, "choice_note", ""):
                self.on_status("audio", self.audio.choice_note)
        self._load_voice_model()
        self.transcriber.start()
        self._bring_up_llm()

    def _bring_up_llm(self) -> str | None:
        """Load the configured model; returns the error, or None once it is ready."""
        try:
            self.llm.ensure_loaded(lambda m: self.on_status("llm", m))
        except Exception as e:  # LLMError, or whatever a provider plugin raises
            # Transcription keeps working; notes and briefs report the problem when used.
            self.on_status("llm", f"error: {e}")
            return str(e)
        if self._closed:  # meeting ended or deleted while the model was loading
            return "the meeting has ended"
        self.record.llm_model = self.llm.cfg.model  # may differ if a fallback was used
        self.record.save()
        self.llm_ready = True
        if not self.tracker.is_alive() and not self.tracker.ident:  # threads start once
            self.tracker.start()
        return None

    def switch_model(self, key: str) -> bool:
        """Use another model for the rest of the meeting (blocking: loads it).

        Loading may unload the current model first (memory), so if the new one fails the old
        one is brought back and notes and briefs keep working. Returns whether it switched.
        """
        old = self.llm.cfg.model
        self.llm_ready = False
        self.llm.cfg.model = key
        error = self._bring_up_llm()
        if error is None or self._closed:
            return error is None
        self.llm.cfg.model = old
        if self._bring_up_llm() is None:
            self.on_status("llm", f"ready — {key} didn't load ({error}), still using {old}")
        return False

    def _notes_wait(self) -> str:
        """Why a scheduled notes refresh should wait ("" = go ahead)."""
        if not self.llm_ready:
            return "model loading"
        if self.transcriber.behind_seconds > BEHIND_DEFER_SECONDS:
            return "transcription catching up"
        return ""

    def set_user_names(self, names: list[str]) -> None:
        """Your name(s), e.g. from the first-run prompt: the trigger, notes and voices follow."""
        names = [n.strip() for n in names if n.strip()]
        self.settings.user.names = names
        self.detector = NameDetector(names, self.settings.triggers.name_cooldown_seconds)
        self._live_hits = 0
        self.tracker.set_user_names(names)
        if self.speakers is not None:
            self.speakers.set_user_names(names)
        if self.record.attendees or names:
            self.set_attendees([a for a in self.record.attendees])

    def stop(self) -> None:
        """Quick stop without a wrap-up (files written so far are kept)."""
        self._closed = True
        self.tracker.stop()
        self.transcriber.stop()
        self.audio.stop()
        self.store.close()

    def discard(self) -> None:
        """Stop without saving anything and delete this meeting's folder.

        Waits for the background threads first, so nothing writes into the folder after
        it is gone (an in-flight notes update is aborted, not awaited).
        """
        self.tracker.yield_to_brief.set()  # makes a running notes update bail out
        self.stop()
        for thread in (self.transcriber, self.tracker):
            if thread.is_alive():
                thread.join(timeout=30)
        shutil.rmtree(self.record.dir, ignore_errors=True)

    # ----- ending a meeting -----
    def wrap_up(self) -> str:
        """Stop listening, refresh the notes one last time and write summary.md.

        Returns the model's suggested title ("" if no LLM or nothing was said).
        """
        self._closed = True
        self.audio.stop()
        self.on_status("stt", "finishing the last words…")
        self.transcriber.finish()  # drains, so speech from the final ~20 s is kept
        self._save_speakers()
        self.store.rewrite_markdown()  # final names everywhere in transcript.md
        self._remember_voices()
        self.tracker.stop()
        if self.tracker.is_alive():
            self.tracker.join(timeout=120)
        self.store.close()
        if not (self.llm_ready and len(self.store)):
            return ""
        with self._briefs_done:  # a brief still answering finishes first; then the model is ours
            self._briefs_done.wait_for(lambda: not self._briefs, timeout=BRIEF_WAIT_SECONDS)
        if self._briefs:
            self.on_status("llm", "a brief is still running — final notes skipped")
        else:
            self.on_status("llm", "final notes…")
            for _ in range(20):  # a long backlog goes in pieces
                if not self.tracker.update(force=True):
                    break
        self.on_status("llm", "writing summary…")
        try:
            title = write_summary(
                self.llm, self.record, self.tracker.notes, self.store.render(self.store.segments()), self.settings.user.names
            )
        except Exception as e:  # LLMError, or whatever a provider plugin raises
            self.on_status("llm", f"summary failed: {e}")
            return ""
        self.on_status("llm", "summary written")
        return title

    def complete(self, title: str | None = None) -> MeetingRecord | None:
        """Finalise the record (title, counts, folder name). A meeting with nothing transcribed leaves no folder."""
        if len(self.store) == 0:
            shutil.rmtree(self.record.dir, ignore_errors=True)
            return None
        return self.record.finish(title)

    # ----- on-demand brief -----
    def brief(self, reason: str) -> Iterator[str]:
        if self._closed:
            raise LLMError("the meeting has ended")
        if not self.llm_ready:
            raise LLMError("the model is not ready yet")
        with self._briefs_done:
            self._briefs += 1
            self.tracker.yield_to_brief.set()  # briefs have priority over notes refreshes
        try:
            yield from brief_stream(
                self.llm,
                user_names=self.settings.user.names,
                attendees=self.attendees,
                notes=self.tracker.notes,
                recent=self.store.recent(self.settings.tracker.recent_window_seconds),
                window_seconds=self.settings.tracker.recent_window_seconds,
                reason=reason,
                max_tokens=self.settings.llm.max_brief_tokens,
                session_dir=self.session_dir,
            )
        finally:
            with self._briefs_done:
                self._briefs -= 1
                if not self._briefs:  # notes may run again once the last brief is done
                    self.tracker.yield_to_brief.clear()
                    self._briefs_done.notify_all()

    def trigger(self, reason: str) -> None:
        if self._closed:  # the last words come in during wrap-up; nobody needs a brief then
            return
        if self.settings.triggers.notify:
            notify("Blaah", f"{reason} — brief on its way")
        self.on_trigger(reason)

    # ----- people -----
    @property
    def attendees(self) -> list[str]:
        return list(self.record.attendees)

    def set_attendees(self, names: list[str]) -> None:
        """Who is in the meeting (any time — what was already said is re-read with the new names)."""
        names = _with_user(names, self.settings.user.names)
        self.record.attendees = names
        self.record.save()
        self.tracker.attendees = names
        if self.speakers is not None:
            self.speakers.set_attendees(names)

    def set_speaker_name(self, voice_id: str, name: str) -> None:
        """Name a voice by hand (from the UI); also adds the name to the attendees."""
        if self.speakers is None:
            return
        if name.strip() and name.strip().lower() not in {a.lower() for a in self.record.attendees}:
            self.set_attendees(self.record.attendees + [name.strip()])
        self.speakers.set_name(voice_id, name)

    # ----- speakers -----
    def _load_voice_model(self) -> None:
        """Load (first run: download, 40 MB) the voice model; speaker labels are skipped if it fails."""
        if self.speakers is None:
            return
        cfg = self.settings.speakers
        try:
            self.on_status("voices", "loading voice model…")
            self.speakers.embedder = VoiceEmbedder(cfg.model)
            # Your own enrolled voice always helps; colleagues' only if remembering is on.
            self.speakers.voiceprints = load_voiceprints(
                self.settings.meetings_dir, None if cfg.remember_voices else [self.speakers.user]
            )
            known = f", {len(self.speakers.voiceprints)} known" if self.speakers.voiceprints else ""
            self.on_status("voices", f"ready{known}")
        except Exception as e:
            self.on_status("voices", f"off: {e}")

    def _speakers_changed(self, names: dict[str, str]) -> None:
        """Names learned or revised (transcriber thread): persist and re-render."""
        self._save_speakers()
        self.store.rewrite_markdown()
        self.on_speakers(names)

    def _save_speakers(self) -> None:
        if self.speakers is None or not self.speakers.voices:
            return
        with self._speakers_lock:
            (self.session_dir / "speakers.json").write_text(json.dumps(self.speakers.snapshot(), indent=1) + "\n")

    def _remember_voices(self) -> None:
        if self.speakers is None or not self.settings.speakers.remember_voices:
            return
        for name, emb in self.speakers.named_prints().items():
            save_voiceprint(self.settings.meetings_dir, name, emb)

    # ----- internal plumbing (transcriber thread) -----
    def _segment(self, seg: Segment) -> None:
        self.on_segment(seg)
        # The live text showed this sentence while it was being said. If a trigger fired since
        # the sentence began, it was for this mention; only one the live text missed fires now.
        if self.detector.last_fired < seg.start - 1.0:
            self._check(seg.text)

    def _live(self, text: str) -> None:
        self.on_live(text)
        # A mention stays in the live text as long as its sentence runs (draft, then final but
        # unfinished), often past the cooldown: fire on a new mention, not the same one again.
        hits = self.detector.count(text)
        if hits > self._live_hits:
            self._check(text)
        self._live_hits = hits

    def _check(self, text: str) -> None:
        hit = self.detector.check(text)
        if hit:
            self.trigger(f'heard "{hit}"')


def _with_user(names: list[str], user_names: list[str]) -> list[str]:
    """Clean the list (trim, dedupe, keep order) and make sure the user is on it."""
    out: list[str] = []
    for n in list(names) + user_names[:1]:
        n = n.strip()
        if n and n.lower() not in {o.lower() for o in out} and n.lower() not in {u.lower() for u in user_names[1:]}:
            out.append(n)
    return out
