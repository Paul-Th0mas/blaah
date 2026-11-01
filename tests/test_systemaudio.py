import numpy as np

from meeting_assistant.systemaudio import BUFFER_SECONDS, SystemAudioCapture


def _cap():
    return SystemAudioCapture(target_rate=100, source="all")  # tiny rate keeps the maths readable


def test_take_returns_queued_audio_and_zero_pads():
    cap = _cap()
    cap._push(np.array([1, 2, 3], dtype=np.float32))
    cap._push(np.array([4, 5], dtype=np.float32))
    out = cap.take(4)
    assert out.tolist() == [1, 2, 3, 4]
    assert cap.take(4).tolist() == [5, 0, 0, 0]  # remainder of the split chunk, then silence
    assert cap.take(2).tolist() == [0, 0]


def test_buffer_is_bounded_to_limit_drift():
    cap = _cap()
    for _ in range(10):
        cap._push(np.ones(100, dtype=np.float32))  # 10 s queued at 100 Hz
    assert cap._buffered <= BUFFER_SECONDS * 100
    assert cap.take(1).tolist() == [1.0]  # newest audio survived, oldest was dropped


def test_peak_tracking_detects_signal():
    cap = _cap()
    cap._push(np.zeros(50, dtype=np.float32))
    assert cap.peak_ever == 0.0
    cap._push(np.array([0.0, -0.4, 0.1], dtype=np.float32))
    assert cap.peak_ever == np.float32(0.4)


def test_mixed_capture_sums_and_clips(monkeypatch):
    from meeting_assistant import audio as audio_mod
    from meeting_assistant.config import AudioCfg

    monkeypatch.setattr(audio_mod, "resolve_device", lambda _name: None)
    cfg = AudioCfg(system_audio="all", mic_gain=1.0, system_gain=0.5)
    mixed = audio_mod.MixedCapture(cfg)
    mixed.system._push(np.array([1.0, 1.0, -1.0], dtype=np.float32))
    out = mixed._mix(np.array([0.9, 0.2, -0.9], dtype=np.float32))
    np.testing.assert_allclose(out, [1.0, 0.7, -1.0], atol=1e-6)  # 0.9+0.5 clipped, 0.2+0.5, -0.9-0.5 clipped
    assert out.dtype == np.float32


def test_the_helper_is_not_relaunched_after_stop(monkeypatch):
    import threading
    import time

    cap = _cap()
    launches = []

    class Exited:
        def poll(self):
            return 0

    monkeypatch.setattr(cap, "_read_stdout", lambda _proc: None)  # the helper exited: output device changed
    monkeypatch.setattr(cap, "_launch", lambda: launches.append(1))
    cap._proc = Exited()
    supervisor = threading.Thread(target=cap._supervise)
    supervisor.start()
    time.sleep(0.1)
    cap.stop()  # quit while the supervisor waits for the new device to settle
    supervisor.join(timeout=2)
    assert launches == [] and not supervisor.is_alive()


def test_a_helper_starting_up_during_stop_is_ended(monkeypatch):
    import io

    from meeting_assistant import systemaudio

    cap = _cap()
    cap._args = ["audiotap"]
    ended = []

    class Starting:
        stderr = io.BytesIO(b"rate 100\n")

        def poll(self):
            return None

        def terminate(self):
            ended.append(1)

        def wait(self, timeout=None):
            return 0

    def popen(*_a, **_k):
        cap.stop()  # stop() lands while the helper is starting
        return Starting()

    monkeypatch.setattr(systemaudio.subprocess, "Popen", popen)
    cap._launch()
    assert ended == [1] and cap._proc is None
