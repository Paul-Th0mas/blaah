"""Mic handling without a real device: dead-mic rule, stop vs the watchdog, device choice."""

import time

import numpy as np

from meeting_assistant import audio
from meeting_assistant.config import AudioCfg


def _mic(monkeypatch) -> audio.MicCapture:
    monkeypatch.setattr(audio, "resolve_device", lambda _name: None)
    return audio.MicCapture(AudioCfg())


def _block(level: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, level, (32000, 1)).astype(np.float32)


def test_only_a_bluetooth_mic_that_is_never_audible_counts_as_dead(monkeypatch):
    mic = _mic(monkeypatch)
    mic.device, mic._default_id, mic._opened_at = None, 5, time.time() - 20
    mic._callback(_block(0.001), 32000, None, None)  # room noise at -60 dBFS: peak ~0.004
    monkeypatch.setattr(audio, "device_transport", lambda _dev: "usb ")
    assert not mic._looks_dead()  # a good USB mic while you just listen
    monkeypatch.setattr(audio, "device_transport", lambda _dev: "blue")
    assert mic._looks_dead()  # the earbuds that send ±1–2 bits
    mic._callback(_block(0.05), 32000, None, None)  # it was audible once
    assert not mic._looks_dead()


def test_the_watchdog_cannot_reopen_the_mic_after_stop(monkeypatch):
    mic = _mic(monkeypatch)
    opened = []
    mic._open_stream = lambda: opened.append(1)
    monkeypatch.setattr(audio.sd, "_terminate", lambda: None)
    monkeypatch.setattr(audio.sd, "_initialize", lambda: None)
    mic._running = True
    mic.stop()  # the meeting ends while a watchdog tick that saw a device switch is in flight
    mic._reopen()
    assert opened == []


def test_enroll_uses_the_mic_meetings_record_from(monkeypatch):
    mic = _mic(monkeypatch)
    monkeypatch.setattr(audio, "default_input_device_id", lambda: 77)
    monkeypatch.setattr(audio, "device_transport", lambda dev: "blue" if dev == 77 else "bltn")
    monkeypatch.setattr(audio, "_device_name", lambda _dev: "Paul's AirPods")
    monkeypatch.setattr(audio, "builtin_input_name", lambda: "MacBook Pro Microphone")
    monkeypatch.setattr(audio, "resolve_device", lambda name: {"": None, "MacBook Pro Microphone": 3}[name])
    device, why = mic.current_choice()
    assert device == 3 and "Bluetooth" in why
