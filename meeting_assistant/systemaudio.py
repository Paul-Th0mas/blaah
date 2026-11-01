"""Capture what the Mac is playing, via the native Core Audio process-tap helper.

The helper (helpers/audiotap/main.swift) is compiled on first use with the Command Line
Tools' swiftc. It writes one header line to stderr ("rate <Hz>") and then raw mono float32
samples to stdout; we resample to the pipeline rate and keep a short ring buffer that the
mic callback drains, so mic and system audio stay time-aligned to within a chunk.

macOS asks once for "System Audio Recording" permission. If it is denied, Core Audio keeps
delivering silence without any error, so `silent_too_long` is how we notice.
"""

from __future__ import annotations

import collections
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .config import PROJECT_ROOT

HELPER_DIR = PROJECT_ROOT / "helpers" / "audiotap"
HELPER_SRC = HELPER_DIR / "main.swift"
HELPER_PLIST = HELPER_DIR / "Info.plist"
HELPER_BIN = HELPER_DIR / "build" / "audiotap"

BUFFER_SECONDS = 6.0  # more than this queued = drift; drop the oldest (must exceed the mic chunk)
SILENCE_GRACE = 10.0  # seconds of all-zero audio before we flag a permission problem


def ensure_helper_built() -> Path:
    """Compile the Swift helper if the binary is missing or older than its sources."""
    sources = (HELPER_SRC, HELPER_PLIST)
    if HELPER_BIN.exists() and HELPER_BIN.stat().st_mtime >= max(p.stat().st_mtime for p in sources):
        return HELPER_BIN
    HELPER_BIN.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "swiftc", "-O", str(HELPER_SRC), "-o", str(HELPER_BIN),
        "-framework", "CoreAudio", "-framework", "Foundation",
        # Embed Info.plist so the binary has a bundle id + usage description for TCC.
        "-Xlinker", "-sectcreate", "-Xlinker", "__TEXT", "-Xlinker", "__info_plist", "-Xlinker", str(HELPER_PLIST),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
        subprocess.run(["codesign", "-s", "-", "-f", str(HELPER_BIN)], check=True, capture_output=True, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"{e.filename} not found — install Xcode Command Line Tools (xcode-select --install)") from None
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"building audiotap failed: {(e.stderr or e.stdout).strip()[-400:]}") from None
    return HELPER_BIN


class SystemAudioCapture:
    """Mono float32 system audio at `target_rate`, pulled with take(n) by whoever owns the clock."""

    def __init__(self, target_rate: int, source: str = "all"):
        self.target_rate = target_rate
        self.source = (source or "all").strip()
        self.rate: int | None = None
        self.level = 0.0
        self.frames_received = 0
        self.peak_ever = 0.0
        self.started_at = 0.0
        self.last_message = ""  # latest info/warn/error line from the helper
        self.restarts = 0
        self.on_restart: Callable[[str], None] = lambda _message: None
        self._proc: subprocess.Popen | None = None
        self._args: list[str] = []
        self._stopping = False
        self._buf: collections.deque[np.ndarray] = collections.deque()
        self._buffered = 0
        self._lock = threading.Lock()
        self._proc_lock = threading.Lock()  # _proc vs stop(): a helper must never outlive stop()
        self._resampler = None
        self._remainder = b""

    # ----- lifecycle -----
    def start(self) -> None:
        binary = ensure_helper_built()
        self._args = [str(binary), "--all"] if self.source.lower() == "all" else [str(binary), "--bundle", self.source]
        self._stopping = False
        self._launch()  # the first launch reports errors (e.g. permission) to the caller
        self.started_at = time.time()
        threading.Thread(target=self._supervise, name="audiotap-supervisor", daemon=True).start()

    def _launch(self) -> None:
        proc = subprocess.Popen(self._args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        # The helper reports progress ("info: …", "warn: …") before its "rate <Hz>" line;
        # the permission prompt can keep it there until the user answers.
        while True:
            line = proc.stderr.readline().decode(errors="replace").strip()
            if line.startswith("rate "):
                rate = int(line.split()[1])
                break
            if line:
                self.last_message = line.split(": ", 1)[-1]
            if not line or line.startswith("error:"):
                _end(proc)
                raise RuntimeError(self.last_message or "audiotap helper produced no output")
        self.rate, self._remainder = rate, b""
        if rate != self.target_rate:
            import soxr

            self._resampler = soxr.ResampleStream(rate, self.target_rate, 1, dtype="float32")
        else:
            self._resampler = None
        with self._proc_lock:
            if self._stopping:  # stop() came while the helper was starting up
                _end(proc)
                return
            self._proc = proc
        threading.Thread(target=self._read_stderr, args=(proc,), name="audiotap-stderr", daemon=True).start()

    def _supervise(self) -> None:
        """Read the helper's audio; when it exits (output device changed) launch it again."""
        while not self._stopping:
            proc = self._proc
            if proc is not None:
                self._read_stdout(proc)  # returns when the helper exits
            if self._stopping:
                return
            time.sleep(0.5)  # let the new device settle
            if self._stopping:
                return
            try:
                self._launch()
                if self._stopping:
                    return
                self.restarts += 1
                self.last_message = ""
                self.on_restart(f"system audio follows the new output device ({self.rate} Hz)")
            except RuntimeError as e:
                self._proc = None
                self.last_message = f"reconnecting… ({e})"
                time.sleep(2.0)

    def stop(self) -> None:
        with self._proc_lock:
            self._stopping = True
            proc, self._proc = self._proc, None
        if proc is not None:
            _end(proc)

    @property
    def silent_too_long(self) -> bool:
        """True once the tap has run a while and never produced a non-zero sample (permission denied)."""
        return self.started_at > 0 and time.time() - self.started_at > SILENCE_GRACE and self.peak_ever == 0.0

    @property
    def status(self) -> str:
        """One short phrase for a status bar."""
        if self.peak_ever > 0:
            return "system audio ✓"
        if self.silent_too_long:
            return "⚠ system audio silent — allow it in System Settings › Privacy & Security › Screen & System Audio Recording"
        return f"system audio: {self.last_message}" if self.last_message else "system audio: starting"

    # ----- consumer side (mic callback) -----
    def take(self, n: int) -> np.ndarray:
        """Up to n frames of system audio, zero-padded if fewer are queued."""
        out = np.zeros(n, dtype=np.float32)
        filled = 0
        with self._lock:
            while filled < n and self._buf:
                chunk = self._buf[0]
                room = n - filled
                if len(chunk) <= room:
                    out[filled : filled + len(chunk)] = chunk
                    filled += len(chunk)
                    self._buf.popleft()
                else:
                    out[filled:] = chunk[:room]
                    self._buf[0] = chunk[room:]
                    filled = n
                self._buffered -= min(len(chunk), room)
        return out

    # ----- producer side (helper process) -----
    def _push(self, samples: np.ndarray) -> None:
        if len(samples) == 0:
            return
        peak = float(np.abs(samples).max())
        self.peak_ever = max(self.peak_ever, peak)
        self.level = float(np.sqrt(np.mean(samples * samples)))
        self.frames_received += len(samples)
        with self._lock:
            self._buf.append(samples)
            self._buffered += len(samples)
            limit = int(BUFFER_SECONDS * self.target_rate)
            while self._buffered > limit and self._buf:  # bound drift between the two clocks
                dropped = self._buf.popleft()
                self._buffered -= len(dropped)

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert self.rate is not None
        block = (self.rate // 10) * 4  # ~100 ms of float32 per read
        while True:
            data = proc.stdout.read(block)
            if not data:
                break
            data = self._remainder + data
            usable = len(data) - len(data) % 4
            self._remainder = data[usable:]
            samples = np.frombuffer(data[:usable], dtype=np.float32)
            if self._resampler is not None:
                samples = self._resampler.resample_chunk(samples).astype(np.float32)
            self._push(samples)

    def _read_stderr(self, proc: subprocess.Popen) -> None:
        for raw in proc.stderr:
            line = raw.decode(errors="replace").strip()
            if line:
                self.last_message = line.split(": ", 1)[-1]


def _end(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
