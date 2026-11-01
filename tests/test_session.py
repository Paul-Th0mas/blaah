"""Session behaviour: one brief per mention, briefs vs notes, wrap-up, switching models."""

import subprocess

import pytest

from meeting_assistant.config import LlmCfg, Settings, StorageCfg
from meeting_assistant.llm import LLMError, LMStudio
from meeting_assistant.session import Session
from meeting_assistant.transcript import Segment


def _session(tmp_path) -> Session:
    settings = Settings(storage=StorageCfg(meetings_dir=str(tmp_path)))
    settings.user.names = ["Paul"]
    settings.triggers.notify = False
    return Session(settings)


class _Clock:
    def __init__(self, t: float):
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_one_mention_triggers_one_brief_however_long_the_sentence(tmp_path, monkeypatch):
    s = _session(tmp_path)
    fired = []
    s.on_trigger = fired.append
    clock = _Clock(1000.0)
    monkeypatch.setattr("meeting_assistant.triggers.time.time", clock)
    # The timeline of a real Parakeet run: a 45-word question that opens with the name.
    s._live("Paul, could you")  # the draft shows the name ~2 s after it is said
    clock.t += 30.2  # past the cooldown, and the sentence still hasn't ended
    s._live("Paul, could you walk us through where the database migration")
    clock.t += 5
    s._segment(Segment(start=998.0, end=1012.0, text="Paul, could you walk us through … once Friday's release is out?"))
    assert fired == ['heard "Paul"']
    s._live("Thanks.")  # next sentence
    clock.t += 40
    s._live("Thanks. And Paul, one more thing")  # a new mention
    assert len(fired) == 2


def test_a_mention_only_the_final_text_caught_still_triggers(tmp_path, monkeypatch):
    s = _session(tmp_path)
    fired = []
    s.on_trigger = fired.append
    clock = _Clock(1000.0)
    monkeypatch.setattr("meeting_assistant.triggers.time.time", clock)
    s._live("Pole, could you")  # the draft misheard the name
    clock.t += 22
    s._segment(Segment(start=999.0, end=1003.0, text="Paul, could you take this one?"))
    assert fired == ['heard "Paul"']


class _SlowLLM:
    name, cfg, last_stats = "slow", LlmCfg(), {}

    def stream(self, system, prompt, *, max_tokens=512, temperature=None):
        yield from ("Right ", "now: ", "the ", "cutover.")


def test_notes_wait_until_the_last_of_overlapping_briefs_is_done(tmp_path):
    s = _session(tmp_path)
    s.llm, s.llm_ready = _SlowLLM(), True
    first = s.brief("requested")
    next(first)
    second = s.brief('heard "Paul"')  # a trigger while the first brief is still writing
    next(second)
    first.close()  # the app cancels the older brief
    assert s.tracker.yield_to_brief.is_set()  # the second still has the model
    list(second)
    assert not s.tracker.yield_to_brief.is_set()


def test_no_briefs_or_triggers_once_the_meeting_is_ending(tmp_path):
    s = _session(tmp_path)
    s.llm, s.llm_ready = _SlowLLM(), True
    fired = []
    s.on_trigger = fired.append
    s._closed = True  # wrap-up has begun; the last words are still being transcribed
    s.trigger('heard "Paul"')
    assert fired == []
    with pytest.raises(LLMError, match="ended"):
        next(s.brief("requested"))


def test_a_model_that_fails_to_load_leaves_the_old_one_working(tmp_path, monkeypatch):
    loaded = {"qwen/qwen3.5-4b"}
    calls = []

    def models(_self):
        return [{"key": k, "type": "llm", "loaded_instances": [{"id": k, "config": {"context_length": 4096}}] if k in loaded else []}
                for k in ("qwen/qwen3.5-4b", "qwen/qwen3.5-9b")]

    def run(cmd, **_kw):
        calls.append(" ".join(cmd[1:3]))
        if cmd[1] == "unload":
            loaded.discard(cmd[2])
        if cmd[1] == "load" and cmd[2] == "qwen/qwen3.5-9b":
            return subprocess.CompletedProcess(cmd, 1, "", "Error: not enough memory to load model")
        if cmd[1] == "load":
            loaded.add(cmd[2])
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(LMStudio, "server_up", lambda _self: True)
    monkeypatch.setattr(LMStudio, "models", models)
    monkeypatch.setattr(LMStudio, "complete", lambda *_a, **_k: None)
    monkeypatch.setattr("meeting_assistant.llm.subprocess.run", run)
    s = _session(tmp_path)
    statuses = []
    s.on_status = lambda _component, message: statuses.append(message)
    s.llm_ready = True
    try:
        assert s.switch_model("qwen/qwen3.5-9b") is False
        assert calls == ["unload qwen/qwen3.5-4b", "load qwen/qwen3.5-9b", "load qwen/qwen3.5-4b"]
        assert s.llm.cfg.model == "qwen/qwen3.5-4b" and s.llm_ready and loaded == {"qwen/qwen3.5-4b"}
        assert "didn't load" in statuses[-1] and "still using qwen/qwen3.5-4b" in statuses[-1]
    finally:
        s.tracker.stop()


def test_the_model_choice_is_saved_only_once_the_switch_worked(monkeypatch):
    from types import SimpleNamespace

    from meeting_assistant.app import MeetingApp

    saved = []
    monkeypatch.setattr("meeting_assistant.config.save_user_settings", saved.append)
    for works in (False, True):
        app = SimpleNamespace(session=SimpleNamespace(switch_model=lambda _key, works=works: works), _ui=lambda *_a, **_k: None, notify=None)
        MeetingApp._switch_model(app, "qwen/qwen3.5-9b")
    assert saved == [{"llm": {"model": "qwen/qwen3.5-9b"}}]
