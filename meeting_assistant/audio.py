from __future__ import annotations

import collections
import ctypes
import queue
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from .config import AudioCfg


# ----- the system's default input device, straight from CoreAudio -----
# PortAudio only sees the devices that existed when it started, so plugging in headphones
# is invisible to it; asking CoreAudio directly is cheap and always current.
class _PropertyAddress(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


def _fourcc(code: str) -> int:
    return int.from_bytes(code.encode("ascii"), "big")


_coreaudio = None


_corefoundation = None


def _ca():
    global _coreaudio
    if _coreaudio is None:
        _coreaudio = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreAudio.framework/CoreAudio")
        for fn in ("AudioObjectGetPropertyData", "AudioObjectGetPropertyDataSize"):
            getattr(_coreaudio, fn).restype = ctypes.c_int32
    return _coreaudio


def _get_u32(obj: int, selector: str, scope: str = "glob") -> int | None:
    address = _PropertyAddress(_fourcc(selector), _fourcc(scope), 0)
    value, size = ctypes.c_uint32(0), ctypes.c_uint32(4)
    status = _ca().AudioObjectGetPropertyData(ctypes.c_uint32(obj), ctypes.byref(address), ctypes.c_uint32(0), None,
                                              ctypes.byref(size), ctypes.byref(value))
    return value.value if status == 0 else None


def default_input_device_id() -> int | None:
    """CoreAudio id of the current default input device (changes when headphones connect)."""
    try:
        return _get_u32(1, "dIn ")
    except OSError:
        return None


def device_transport(device_id: int | None) -> str:
    """'blue' (Bluetooth), 'bltn' (built-in), 'usb ', … for a CoreAudio device."""
    if device_id is None:
        return ""
    try:
        t = _get_u32(device_id, "tran")
        return t.to_bytes(4, "big").decode("ascii", "replace") if t else ""
    except OSError:
        return ""


def _device_name(device_id: int) -> str:
    global _corefoundation
    if _corefoundation is None:
        _corefoundation = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        _corefoundation.CFStringGetCString.restype = ctypes.c_bool
    address = _PropertyAddress(_fourcc("lnam"), _fourcc("glob"), 0)
    ref, size = ctypes.c_void_p(), ctypes.c_uint32(ctypes.sizeof(ctypes.c_void_p))
    if _ca().AudioObjectGetPropertyData(ctypes.c_uint32(device_id), ctypes.byref(address), ctypes.c_uint32(0), None,
                                        ctypes.byref(size), ctypes.byref(ref)) != 0 or not ref.value:
        return ""
    buf = ctypes.create_string_buffer(256)
    ok = _corefoundation.CFStringGetCString(ref, buf, 256, 0x08000100)  # UTF-8
    _corefoundation.CFRelease(ref)
    return buf.value.decode() if ok else ""


def builtin_input_name() -> str | None:
    """Name of the Mac's built-in microphone, if there is one."""
    try:
        address = _PropertyAddress(_fourcc("dev#"), _fourcc("glob"), 0)
        size = ctypes.c_uint32(0)
        if _ca().AudioObjectGetPropertyDataSize(ctypes.c_uint32(1), ctypes.byref(address), ctypes.c_uint32(0), None, ctypes.byref(size)) != 0:
            return None
        ids = (ctypes.c_uint32 * (size.value // 4))()
        if _ca().AudioObjectGetPropertyData(ctypes.c_uint32(1), ctypes.byref(address), ctypes.c_uint32(0), None,
                                            ctypes.byref(size), ctypes.byref(ids)) != 0:
            return None
        for dev in ids:
            streams = _PropertyAddress(_fourcc("stm#"), _fourcc("inpt"), 0)
            n = ctypes.c_uint32(0)
            has_input = _ca().AudioObjectGetPropertyDataSize(ctypes.c_uint32(dev), ctypes.byref(streams), ctypes.c_uint32(0), None, ctypes.byref(n)) == 0 and n.value > 0
            if has_input and device_transport(dev) == "bltn":
                return _device_name(dev) or None
    except OSError:
        return None
    return None


def list_input_devices() -> list[tuple[int, str, int]]:
    """(index, name, default_samplerate) for every device with an input channel."""
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] > 0:
            out.append((i, d["name"], int(d["default_samplerate"])))
    return out


def resolve_device(name_substring: str) -> int | None:
    """Pick the first input device whose name contains the substring; None = system default."""
    if not name_substring:
        return None
    wanted = name_substring.lower()
    for idx, name, _ in list_input_devices():
        if wanted in name.lower():
            return idx
    raise ValueError(
        f"No input device matching {name_substring!r}. Available: "
        + ", ".join(name for _, name, _ in list_input_devices())
    )


class MicCapture:
    """Streams mono float32 audio at the target rate into `chunks`, one block per chunk_seconds.

    Pausing keeps feeding silence so downstream timestamps stay aligned with wall-clock.

    A watchdog follows device changes: when the default input switches (headphones plugged
    in, AirPods connecting) or the stream goes quiet without an error (Bluetooth headsets
    switching mode when a call opens their mic), the mic is reopened on the current device.
    Everything downstream runs on this stream's clock, so without this a dead stream
    silently stops the whole transcript.
    """

    WATCH_SECONDS = 1.0
    # One pair of Bluetooth earbuds sent ±1–2 bits of noise (peak 0.003) while their wearer
    # talked. A Bluetooth mic that never once gets above this in its first 15 s is that case.
    # Only Bluetooth, and only "never": a good USB mic in a quiet room also peaks below this
    # while you just listen, and must not be swapped out for that.
    DEAD_PEAK = 0.005
    DEAD_SECONDS = 15.0

    def __init__(self, cfg: AudioCfg):
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.chunk_frames = int(cfg.sample_rate * cfg.chunk_seconds)
        self.pinned = bool(cfg.input_device.strip())  # a configured device is kept, not followed
        self.device = resolve_device(cfg.input_device)
        self.choice_note = ""  # why a device other than the default was picked, for the UI
        self._silent_devices: set[str] = set()  # mics that delivered only zeros this session
        self._opened_at = 0.0
        self._loudest = 0.0  # highest peak since the stream was opened
        self.chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=300)
        self.level = 0.0  # RMS of the most recent block, for a VU meter
        self.paused = False
        self.dropped = 0
        self.reopened = 0
        self.on_switch: Callable[[str], None] = lambda _message: None
        self.stall_seconds = max(3.0, 2.5 * cfg.chunk_seconds)
        self._stream: sd.InputStream | None = None
        self._resampler = None
        self._device_name = ""
        self._default_id: int | None = None
        self._last_audio = 0.0
        self._running = False
        self._lock = threading.Lock()

    @property
    def device_name(self) -> str:
        return self._device_name

    def start(self) -> None:
        with self._lock:
            self._open_stream()
            self._running = True
        threading.Thread(target=self._watch, name="mic-watchdog", daemon=True).start()

    def current_choice(self) -> tuple[int | None, str]:
        """The device a meeting would record from right now, and why (used by `enroll`, so
        your voiceprint comes from the same mic your meetings do)."""
        self._default_id = default_input_device_id()
        return self._choose_device(), self.choice_note

    def _choose_device(self) -> int | None:
        """PortAudio device to record from (None = system default).

        Bluetooth headset mics are avoided by default: they are narrowband call audio, opening
        one switches the buds into call mode (worse listening for you), and some send pure
        silence to anything but the call app (seen with Bluetooth earbuds). With headphones on, the
        built-in mic hears only you — the others are in your ears — which is what the app needs.
        """
        if self.pinned:
            return resolve_device(self.cfg.input_device)
        default_name = _device_name(self._default_id) if self._default_id is not None else ""
        builtin = builtin_input_name()
        if builtin and default_name != builtin:
            if self.cfg.prefer_builtin_mic and device_transport(self._default_id) == "blue":
                self.choice_note = f"using {builtin} — the {default_name} mic is call-quality Bluetooth"
                return resolve_device(builtin)
            if default_name in self._silent_devices:
                self.choice_note = f"using {builtin} — the {default_name} mic picks up almost nothing"
                return resolve_device(builtin)
        self.choice_note = ""
        return None

    def _open_stream(self) -> None:
        self._resampler = None
        self._default_id = default_input_device_id()
        self.device = self._choose_device()
        try:
            stream = self._open(self.sample_rate, self.chunk_frames)
        except sd.PortAudioError:
            # Device refuses 16 kHz: capture at its native rate and resample on the fly.
            native = int(sd.query_devices(self.device, "input")["default_samplerate"])
            import soxr

            self._resampler = soxr.ResampleStream(native, self.sample_rate, 1, dtype="float32")
            native_block = int(self.chunk_frames * native / self.sample_rate)
            stream = self._open(native, native_block)
        info = sd.query_devices(self.device) if self.device is not None else sd.query_devices(kind="input")
        self._device_name = str(info["name"])
        self._last_audio = self._opened_at = time.time()
        self._loudest = 0.0
        stream.start()
        self._stream = stream

    def _watch(self) -> None:
        while self._running:
            time.sleep(self.WATCH_SECONDS)
            if not self._running:
                return
            current = default_input_device_id()
            switched = not self.pinned and current is not None and current != self._default_id
            stalled = time.time() - self._last_audio > self.stall_seconds
            silent = self._looks_dead()
            if silent:
                self._silent_devices.add(self._device_name)
            if switched or stalled or silent:
                try:
                    self._reopen()
                    what = "switched to" if switched or silent else "reconnected:"
                    self.on_switch(self.choice_note or f"mic {what} {self._device_name}")
                except Exception as e:  # device mid-change; try again next tick
                    self.on_switch(f"mic unavailable ({e}) — retrying")

    def _looks_dead(self) -> bool:
        """True for a followed Bluetooth mic that hasn't been audible once since it opened."""
        return (
            not self.pinned
            and self.device is None  # the system default, not the built-in mic we chose
            and self._loudest < self.DEAD_PEAK
            and time.time() - self._opened_at > self.DEAD_SECONDS
            and device_transport(self._default_id) == "blue"
        )

    def _reopen(self) -> None:
        with self._lock:
            if not self._running:  # stop() won while this watchdog tick was deciding
                return
            old, self._stream = self._stream, None
            if old is not None:
                try:
                    old.abort()
                    old.close()
                except Exception:
                    pass
            # PortAudio lists devices once, at start: restart it so new headphones exist.
            sd._terminate()
            sd._initialize()
            self._open_stream()
            self.reopened += 1

    def _open(self, rate: int, blocksize: int) -> sd.InputStream:
        return sd.InputStream(
            samplerate=rate,
            blocksize=blocksize,
            device=self.device,
            channels=1,
            dtype="float32",
            callback=self._callback,
        )

    def _callback(self, indata: np.ndarray, frames: int, _time, status) -> None:
        self._last_audio = time.time()
        self._loudest = max(self._loudest, float(np.abs(indata).max()) if len(indata) else 0.0)
        mono = indata[:, 0].copy()
        if self._resampler is not None:
            mono = self._resampler.resample_chunk(mono).astype(np.float32)
        mono = self._mix(mono)
        self.level = float(np.sqrt(np.mean(mono * mono))) if len(mono) else 0.0
        if self.paused:
            mono = np.zeros_like(mono)
        try:
            self.chunks.put_nowait(mono)
        except queue.Full:
            self.dropped += 1

    def _mix(self, mono: np.ndarray) -> np.ndarray:
        """Hook for subclasses to blend other sources into the mic clock."""
        return mono

    def stop(self) -> None:
        with self._lock:
            self._running = False
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None


class MixedCapture(MicCapture):
    """Mic plus what the Mac is playing (remote participants), summed on the mic's clock.

    Also keeps 100 ms loudness readings of both sources, so a sentence can be traced to
    the mic (you, or someone in your room) or the speakers (someone on the call).
    """

    LEVEL_SECONDS = 0.1
    LEVEL_HISTORY = 3000  # 5 minutes of readings
    # Call audio at speech level. Meet's background (other people's room noise, comfort noise)
    # sits below this; the old 0.005 counted that as remote speech, so the user never
    # qualified as "mic only" (seen in a real Meet call: every voice got "not you").
    REMOTE_SPEECH_RMS = 0.01
    REMOTE_MIN_SECONDS = 10.0  # remote speech heard before mic/speaker hints are trusted
    MINE_RATIO = 2.0  # the mic must be this much louder than the call for a block to be yours

    def __init__(self, cfg: AudioCfg):
        super().__init__(cfg)
        from .systemaudio import SystemAudioCapture

        self.system = SystemAudioCapture(cfg.sample_rate, cfg.system_audio)
        self.system.on_restart = lambda message: self.on_switch(message)
        self.mic_gain = cfg.mic_gain
        self.system_gain = cfg.system_gain
        self._block = int(cfg.sample_rate * self.LEVEL_SECONDS)
        self._levels: collections.deque[tuple[float, float]] = collections.deque(maxlen=self.LEVEL_HISTORY)
        self._levels_total = 0  # readings ever taken; the reading at index i is stream time i × 0.1 s
        self._remote_blocks = 0

    @property
    def device_name(self) -> str:
        return f"{super().device_name} + system audio [{self.system.source}]"

    def start(self) -> None:
        self.system.start()  # first, so its buffer is filling by the time the mic clock ticks
        super().start()

    def _mix(self, mono: np.ndarray) -> np.ndarray:
        mic = mono * self.mic_gain
        system = self.system.take(len(mono)) * self.system_gain
        for i in range(0, len(mic) - self._block + 1, self._block):
            m, s = _rms(mic[i : i + self._block]), _rms(system[i : i + self._block])
            self._levels.append((m, s))
            self._levels_total += 1
            self._remote_blocks += s > self.REMOTE_SPEECH_RMS
        return np.clip(mic + system, -1.0, 1.0).astype(np.float32)

    def _window(self, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray] | None:
        first = self._levels_total - len(self._levels)
        a = int(t0 / self.LEVEL_SECONDS) - first
        b = int(t1 / self.LEVEL_SECONDS) + 1 - first
        if a < 0 or b <= a or b > len(self._levels):
            return None
        window = list(self._levels)[a:b]
        return np.array([m for m, _ in window]), np.array([s for _, s in window])

    def levels(self, t0: float, t1: float) -> tuple[float, float] | None:
        """Mean (mic, call) loudness over stream seconds t0–t1, kept per sentence for tuning."""
        w = self._window(t0, t1)
        return (float(w[0].mean()), float(w[1].mean())) if w else None

    def source_hint(self, t0: float, t1: float) -> str | None:
        """'me' if stream seconds t0–t1 came from your mic, 'others' if from the call.

        Compares the two sources block by block: yours when the mic hears speech clearly
        louder than whatever the call is playing (also true with laptop speakers, whose echo
        reaches the mic much quieter than the call's own signal); theirs when the call plays
        speech at least as loud as the mic hears. Only used in calls — in a room-only meeting
        everyone is on the mic, so it says nothing.
        """
        if self._remote_blocks * self.LEVEL_SECONDS < self.REMOTE_MIN_SECONDS:
            return None
        w = self._window(t0, t1)
        if w is None:
            return None
        mic, system = w
        # The mic's own noise floor. A low percentile: in calls the mic hears speaker echo most
        # of the time, so a median-ish estimate would mistake echo for background. A copy
        # first: the audio callback appends to the deque meanwhile.
        floor = float(np.percentile([m for m, _ in list(self._levels)], 5))
        talking = mic > max(0.01, 3 * floor)
        mine = float(np.mean(talking & (mic > self.MINE_RATIO * system)))
        theirs = float(np.mean((system > self.REMOTE_SPEECH_RMS) & (system >= mic)))
        if mine >= 0.4 and mine > theirs:
            return "me"
        if theirs >= 0.4:
            return "others"
        return None

    def stop(self) -> None:
        super().stop()
        self.system.stop()


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x))) if len(x) else 0.0


def make_capture(cfg: AudioCfg) -> MicCapture:
    return MixedCapture(cfg) if cfg.system_audio.strip() else MicCapture(cfg)


class FileCapture:
    """Drop-in replacement for MicCapture that plays a WAV/FLAC/MP3 file as if it were the mic.

    Used for testing without talking. Paces chunks in real time, then keeps feeding
    silence after the file ends so the transcriber can finalise its trailing words.
    """

    def __init__(self, path: str | Path, cfg: AudioCfg, realtime: bool = True):
        import soundfile as sf

        self.path = Path(path)
        self.sample_rate = cfg.sample_rate
        self.chunk_frames = int(cfg.sample_rate * cfg.chunk_seconds)
        self.realtime = realtime
        audio, sr = sf.read(str(self.path), dtype="float32")
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            import soxr

            audio = soxr.resample(audio, sr, self.sample_rate).astype(np.float32)
        self.audio = audio
        self.duration = len(audio) / self.sample_rate
        self.chunks: queue.Queue[np.ndarray] = queue.Queue(maxsize=300)
        self.level = 0.0
        self.paused = False
        self.dropped = 0
        self.finished = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._feed, name="file-audio", daemon=True)

    @property
    def device_name(self) -> str:
        return f"file:{self.path.name}"

    def start(self) -> None:
        self._thread.start()

    def _feed(self) -> None:
        pace = self.chunk_frames / self.sample_rate if self.realtime else 0.0
        for i in range(0, len(self.audio), self.chunk_frames):
            if self._stop.is_set():
                return
            chunk = self.audio[i : i + self.chunk_frames]
            if len(chunk) < self.chunk_frames:
                chunk = np.pad(chunk, (0, self.chunk_frames - len(chunk)))
            self.level = float(np.sqrt(np.mean(chunk * chunk)))
            self.chunks.put(chunk)
            if pace:
                time.sleep(pace)
        self.finished.set()
        silence = np.zeros(self.chunk_frames, dtype=np.float32)
        while not self._stop.is_set():
            self.level = 0.0
            self.chunks.put(silence)
            time.sleep(pace or 0.05)

    def stop(self) -> None:
        self._stop.set()
