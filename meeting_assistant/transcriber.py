from __future__ import annotations

import collections
import queue
import threading
import time
from typing import TYPE_CHECKING, Callable, NamedTuple

import mlx.core as mx
import numpy as np
from parakeet_mlx import from_pretrained
from parakeet_mlx.parakeet import ParakeetRNNT, ParakeetTDT

from .audio import MicCapture
from .config import SttCfg
from .transcript import Segment, TranscriptStore

if TYPE_CHECKING:
    from .speakers import SpeakerTracker

SENTENCE_END = (".", "?", "!")
# Emit an unterminated sentence once no new words have become final for this long.
STALL_SECONDS = 4.0
# ...or once it gets this long without punctuation (rambling monologue).
MAX_PENDING_WORDS = 60
RING_SECONDS = 120.0  # recent audio kept for cutting each sentence's voice sample
MAX_VOICE_SECONDS = 15.0  # longest stretch of a sentence used for its voiceprint
# When audio has queued up (GPU busy with video calls or the LLM), feed it to the model in
# one call instead of second by second: each call re-encodes the look-ahead window, so one
# big call costs little more than a 1 s one. This is what keeps transcription from drifting
# minutes behind under load.
MAX_BATCH_SECONDS = 30.0


class Timed(NamedTuple):
    token: object  # parakeet_mlx AlignedToken
    t: float  # seconds since the stream started, when the token was spoken


class FrameClock:
    """Maps the streaming decoder's window-relative token times onto the stream timeline.

    Parakeet decodes over a rolling window of mel frames, and reports token times
    relative to that window. We record the stream time of every mel frame as it is made
    (the STFT's centre padding yields hop+1 frames per chunk, so frame index × 10 ms
    drifts ~1 %/s and can't be used directly) and where each window starts. Accurate
    to ~0.2 s, measured against a voiced meeting with known line boundaries.

    Uses parakeet-mlx's StreamingParakeet internals (`audio_buffer`, `mel_buffer`);
    if those ever change, `ok` turns False and the caller falls back to coarse dating.
    """

    def __init__(self, hop: int, sample_rate: int):
        self.hop, self.sr = hop, sample_rate
        self.times: collections.deque[float] = collections.deque()
        self.base = 0  # global frame index of times[0]
        self.window = 0  # global frame index of the current window's first frame
        self.ok = True
        self._fed = 0
        self._before: tuple[int, int] = (0, 0)

    def before(self, stream) -> None:
        try:
            mel = stream.mel_buffer.shape[1] if stream.mel_buffer is not None else 0
            self._before = (mel, len(stream.audio_buffer))
        except AttributeError:
            self.ok = False

    def after(self, stream, fed: int) -> None:
        if not self.ok:
            return
        try:
            mel_before, buf_before = self._before
            consumed = buf_before + fed - len(stream.audio_buffer)
        except AttributeError:
            self.ok = False
            return
        start = (self._fed - buf_before) / self.sr
        n_new = consumed // self.hop + 1 if consumed else 0
        self.times.extend(start + j * self.hop / self.sr for j in range(n_new))
        self._fed += fed
        total = self.base + len(self.times)
        self.window = total - n_new - mel_before
        while self.base < self.window - 1000:  # keep a margin behind the window
            self.times.popleft()
            self.base += 1

    def at(self, window_seconds: float) -> float:
        g = self.window + int(round(window_seconds * self.sr / self.hop)) - self.base
        g = min(max(g, 0), len(self.times) - 1)
        return self.times[g]


class Transcriber(threading.Thread):
    """Pulls audio chunks from the mic and streams them through Parakeet.

    Parakeet keeps a right-context window, so tokens become *final* roughly
    right_context * 80 ms after they are spoken (~20 s at the default 256).
    Finalised sentences go to the store — each with its speaker, when a SpeakerTracker
    is attached; the trailing final-but-unterminated text plus Parakeet's draft tokens
    are published as `store.live` so the UI and the name detector see speech within ~1 s.
    """

    def __init__(
        self,
        cfg: SttCfg,
        audio: MicCapture,
        store: TranscriptStore,
        on_status: Callable[[str], None] | None = None,
        on_live: Callable[[str], None] | None = None,
        speakers: SpeakerTracker | None = None,
    ):
        super().__init__(name="transcriber", daemon=True)
        self.cfg = cfg
        self.audio = audio
        self.store = store
        self.speakers = speakers
        self.on_status = on_status or (lambda _s: None)
        self.on_live = on_live or (lambda _t: None)
        self._stopped = threading.Event()  # not `_stop`: that shadows Thread._stop(), which join() calls
        self._flush = threading.Event()
        self.last_chunk_ms = 0.0  # processing time of the latest model call
        self.speed = 0.0  # processing time per second of audio (smoothed); > 1 means falling behind
        self.fed_seconds = 0.0  # audio processed so far
        self.ready = threading.Event()
        self.error: Exception | None = None
        self.stream_start: float | None = None  # wall-clock time of stream second 0
        self._ring: collections.deque[np.ndarray] = collections.deque()
        self._ring_start = 0  # stream sample index of _ring[0]
        self._sr = 16000

    def stop(self) -> None:
        """Stop now; whatever is still in Parakeet's draft window is dropped."""
        self._stopped.set()

    @property
    def behind_seconds(self) -> float:
        """How far transcription lags the live audio (0 when keeping up)."""
        queued = self.audio.chunks.qsize() * getattr(self.audio, "chunk_frames", self._sr) / self._sr
        return max(0.0, queued)

    def finish(self, timeout: float = 90.0) -> None:
        """Stop after draining: remaining audio, then enough silence to finalise the last words.

        Call after the audio source has stopped. Takes a few seconds (the model has to
        see ~20 s of audio past the last word before it commits it).
        """
        self._flush.set()
        self.join(timeout=timeout)

    def run(self) -> None:
        try:
            self._run()
        except Exception as e:  # keep the error visible instead of a silently dead thread
            self.error = e
            self.on_status(f"error: {e}")

    # ----- recent audio, for voiceprints -----
    def _remember(self, chunk: np.ndarray) -> None:
        self._ring.append(chunk)
        limit = int(RING_SECONDS * self._sr)
        while sum(len(c) for c in self._ring) - len(self._ring[0]) > limit:
            self._ring_start += len(self._ring.popleft())

    def audio_between(self, t0: float, t1: float) -> np.ndarray:
        """Audio from stream second t0 to t1 (clipped to what is still in the ring buffer)."""
        a, b = int(max(t0, 0) * self._sr), int(t1 * self._sr)
        out, pos = [], self._ring_start
        for chunk in self._ring:
            lo, hi = max(a, pos), min(b, pos + len(chunk))
            if lo < hi:
                out.append(chunk[lo - pos : hi - pos])
            pos += len(chunk)
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def _segment(self, t0: float, t1: float, text: str) -> Segment:
        """A finished sentence with its speaker, source (mic/call) and loudness readings."""
        seg = Segment(start=self.stream_start + t0, end=self.stream_start + t1, text=text)
        try:
            hint = getattr(self.audio, "source_hint", lambda _a, _b: None)(t0, t1)
            levels = getattr(self.audio, "levels", lambda _a, _b: None)(t0, t1)
            seg.source = hint or ""
            seg.levels = [round(levels[0], 4), round(levels[1], 4)] if levels else None
            if self.speakers is not None:
                mid = (t0 + t1) / 2
                half = min(t1 - t0, MAX_VOICE_SECONDS) / 2
                audio = self.audio_between(max(t0 - 0.1, mid - half), min(t1 + 0.1, mid + half))
                seg.speaker = self.speakers.assign(audio, text, hint)
        except Exception as e:  # a speaker problem must never stop transcription
            self.on_status(f"speaker id error: {e}")
        return seg

    # ----- main loop -----
    def _run(self) -> None:
        self.on_status(f"loading {self.cfg.model.split('/')[-1]}…")
        model = from_pretrained(self.cfg.model)
        sr = self._sr = model.preprocessor_config.sample_rate
        if sr != self.audio.sample_rate:
            raise RuntimeError(f"STT model wants {sr} Hz but audio is {self.audio.sample_rate} Hz")
        ctx = (self.cfg.left_context, self.cfg.right_context)
        hop = model.preprocessor_config.hop_length
        # One encoder frame = mel hop × subsampling (80 ms); the finalised region
        # trails the audio fed so far by right_context frames.
        finalize_lag = self.cfg.right_context * hop / sr * model.encoder_config.subsampling_factor

        want_draft, decode_calls = _skippable_drafts(model)

        # First call compiles kernels (~3 s); do it on a throwaway stream so it
        # neither delays nor pollutes the real timeline.
        with model.transcribe_stream(context_size=ctx) as warm:
            warm.add_audio(mx.zeros(sr))
        self.ready.set()
        self.on_status("listening")

        with model.transcribe_stream(context_size=ctx) as stream:
            clock = FrameClock(hop, sr)
            fed_seconds = 0.0  # audio fed so far
            n_taken = 0  # finalized tokens already moved into `pending`
            pending: list[Timed] = []  # final tokens not yet emitted as a sentence
            last_final_at = 0.0  # fed_seconds when a token last became final

            def process(chunk: np.ndarray, draft: bool = True) -> None:
                nonlocal fed_seconds, n_taken, pending, last_final_at
                if self.stream_start is None:
                    self.stream_start = time.time() - len(chunk) / sr
                self._remember(chunk)

                t0 = time.perf_counter()
                clock.before(stream)
                want_draft[0], decode_calls[0] = draft, 0
                stream.add_audio(mx.array(chunk))
                clock.after(stream, len(chunk))
                took = time.perf_counter() - t0
                self.last_chunk_ms = took * 1000
                per_second = took / (len(chunk) / sr)
                self.speed = per_second if self.speed == 0 else 0.8 * self.speed + 0.2 * per_second
                fed_seconds += len(chunk) / sr
                self.fed_seconds = fed_seconds

                finalized = stream.finalized_tokens
                if len(finalized) > n_taken:
                    for tok in finalized[n_taken:]:
                        t = clock.at(tok.start) if clock.ok else fed_seconds - finalize_lag
                        pending.append(Timed(tok, t))
                    n_taken = len(finalized)
                    last_final_at = fed_seconds

                sentences = _split_sentences(pending)
                if sentences:
                    last = sentences[-1]
                    stalled = fed_seconds - last_final_at > STALL_SECONDS
                    too_long = len(_text(last).split()) >= MAX_PENDING_WORDS
                    if _text(last).endswith(SENTENCE_END) or stalled or too_long:
                        pending = []
                    else:
                        sentences = sentences[:-1]
                        pending = last
                    for group in sentences:
                        t_start = group[0].t
                        t_end = max(group[-1].t + float(getattr(group[-1].token, "duration", 0.0)), t_start)
                        text = _text(group)
                        if any(ch.isalnum() for ch in text):  # skip stray punctuation like "."
                            self.store.add(self._segment(t_start, t_end, text))

                live = (_text(pending) + " " + "".join(t.text for t in stream.draft_tokens)).strip()
                self.store.set_live(live)
                self.on_live(live)

            max_batch = int(MAX_BATCH_SECONDS * sr)
            while not self._stopped.is_set():
                try:
                    chunk = self.audio.chunks.get(timeout=0.5)
                except queue.Empty:
                    if self._flush.is_set():
                        break  # source has stopped and everything queued is processed
                    continue
                batch = [chunk]
                size = len(chunk)
                while size < max_batch:  # take whatever else has piled up meanwhile
                    try:
                        more = self.audio.chunks.get_nowait()
                    except queue.Empty:
                        break
                    batch.append(more)
                    size += len(more)
                caught_up = len(batch) == 1 and self.audio.chunks.empty()
                process(batch[0] if len(batch) == 1 else np.concatenate(batch), draft=caught_up)

            if self._flush.is_set() and not self._stopped.is_set() and self.stream_start is not None:
                # Drain in two calls: enough silence to finalise the look-ahead window, then
                # enough for the last unterminated sentence to stall out and be emitted.
                process(np.zeros(int((finalize_lag + 2) * sr), dtype=np.float32), draft=False)
                process(np.zeros(int((STALL_SECONDS + 1) * sr), dtype=np.float32), draft=False)
                self.store.set_live("")
                self.on_live("")


def _skippable_drafts(model) -> tuple[list[bool], list[int]]:
    """Let the caller skip Parakeet's throwaway draft decode: ([want_draft], [decode_calls]).

    Each add_audio decodes twice: the newly finalised region, then a "draft" of the whole
    look-ahead window (~40 % of the call) that only feeds the live text. While catching up
    nobody needs live text for stale audio, so the draft can go. Only for TDT/RNNT models:
    their decode returns (tokens, states), which the stub mimics; a CTC model returns tokens
    alone, so it keeps its drafts.
    """
    want_draft, decode_calls = [True], [0]
    if isinstance(model, (ParakeetTDT, ParakeetRNNT)):
        real_decode = model.decode

        def decode(*args, **kwargs):
            decode_calls[0] += 1
            if decode_calls[0] == 2 and not want_draft[0]:
                return [[]], [None]  # empty draft, same shape as the real result
            return real_decode(*args, **kwargs)

        model.decode = decode
    return want_draft, decode_calls


def _text(items: list[Timed]) -> str:
    return "".join(d.token.text for d in items).strip()


def _split_sentences(items: list[Timed]) -> list[list[Timed]]:
    """Group tokens into sentences on terminal punctuation; last group may be unterminated."""
    groups: list[list[Timed]] = []
    current: list[Timed] = []
    for d in items:
        current.append(d)
        if d.token.text.strip().endswith(SENTENCE_END):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups
