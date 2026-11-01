"""Settings layers, the provider plugin registry, model fallback, and `blaah models`."""

import tomllib
from types import SimpleNamespace

import pytest

from meeting_assistant import __main__ as cli
from meeting_assistant import providers
from meeting_assistant.config import LlmCfg, Settings, save_user_settings
from meeting_assistant.llm import LLMError, LMStudio, pick_fallback
from meeting_assistant.providers import ModelInfo, Provider, make_llm, provider_names


# ----- settings layers -----
def test_personal_settings_override_shared_ones(tmp_path, monkeypatch):
    shared = tmp_path / "config.toml"
    shared.write_text('[user]\nnames = []\n[llm]\nmodel = "shared/model"\ntemperature = 0.5\n')
    personal = tmp_path / "me.toml"
    personal.write_text('[llm]\nmodel = "my/model"\n')
    monkeypatch.setattr("meeting_assistant.config.DEFAULT_CONFIG_PATH", shared)
    s = Settings.load(user_path=personal)
    assert s.llm.model == "my/model" and s.llm.temperature == 0.5 and s.user.names == []
    assert Settings.load(shared).llm.model == "shared/model"  # an explicit file is used alone


def test_save_user_settings_merges_and_round_trips(tmp_path):
    path = tmp_path / "blaah" / "config.toml"
    save_user_settings({"user": {"names": ['Pri "P"', "Priya"]}}, path)
    save_user_settings({"llm": {"model": "qwen/qwen3.5-9b", "auto_load": False, "temperature": 0.3}}, path)
    data = tomllib.loads(path.read_text())
    assert data == {"user": {"names": ['Pri "P"', "Priya"]},
                    "llm": {"model": "qwen/qwen3.5-9b", "auto_load": False, "temperature": 0.3}}


# ----- registry and plugins -----
def test_lmstudio_is_built_in():
    assert provider_names()["lmstudio"] == "meeting_assistant.llm:LMStudio"
    llm = make_llm(LlmCfg())
    assert isinstance(llm, LMStudio) and isinstance(llm, Provider)


def test_unknown_provider_lists_the_available_ones():
    with pytest.raises(ValueError, match="available: lmstudio"):
        make_llm(LlmCfg(provider="nope"))


class FakeProvider:
    """A minimal third-party provider, as a plugin package would ship it."""

    name = "fake"

    def __init__(self, cfg):
        self.cfg, self.last_stats = cfg, {}

    def ensure_loaded(self, on_status=lambda _s: None):
        on_status("ready")

    def complete(self, system, prompt, *, max_tokens=512, temperature=None):
        return SimpleNamespace(text="ok")

    def stream(self, system, prompt, *, max_tokens=512, temperature=None):
        yield "ok"

    def list_models(self):
        return [ModelInfo("fake/model", "Fake", 0.1, True)]

    def download(self, key, on_line=print):
        raise NotImplementedError


def test_plugins_register_through_entry_points(monkeypatch):
    ep = SimpleNamespace(name="fake", value=f"{__name__}:FakeProvider")
    monkeypatch.setattr(providers, "entry_points", lambda group: [ep] if group == "blaah.providers" else [])
    assert "fake" in provider_names() and "lmstudio" in provider_names()
    llm = make_llm(LlmCfg(provider="fake", model="fake/model"))
    assert isinstance(llm, FakeProvider) and isinstance(llm, Provider)
    assert llm.complete("", "hi").text == "ok"


# ----- fallback when the configured model isn't downloaded -----
M = lambda key, size, loaded=False, kind="llm": {"key": key, "type": kind, "size_bytes": size,  # noqa: E731
                                                 "loaded_instances": [{"id": key}] if loaded else []}


def test_pick_fallback_prefers_loaded_then_recommended_then_smallest():
    assert pick_fallback([M("big", 9e9), M("small", 1e9), M("qwen/qwen3.5-4b", 3e9), M("x", 5e9, loaded=True)])["key"] == "x"
    assert pick_fallback([M("big", 9e9), M("qwen/qwen3.5-4b", 3e9), M("small", 1e9)])["key"] == "qwen/qwen3.5-4b"
    assert pick_fallback([M("big", 9e9), M("small", 1e9), M("embed", 1e8, kind="embedding")])["key"] == "small"
    assert pick_fallback([M("embed", 1e8, kind="embedding")]) is None


def _offline_lmstudio(monkeypatch, models):
    llm = LMStudio(LlmCfg(model="missing/model"))
    monkeypatch.setattr(llm, "server_up", lambda: True)
    monkeypatch.setattr(llm, "models", lambda: models)
    monkeypatch.setattr(llm, "complete", lambda *a, **k: SimpleNamespace(text="OK"))
    monkeypatch.setattr("meeting_assistant.llm.subprocess.run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    return llm


def test_missing_model_falls_back_and_says_so(monkeypatch):
    llm = _offline_lmstudio(monkeypatch, [M("qwen/qwen3.5-9b", 6e9)])
    said = []
    llm.ensure_loaded(said.append)
    assert llm.cfg.model == "qwen/qwen3.5-9b"
    assert any("missing/model isn't downloaded — using qwen/qwen3.5-9b" in s for s in said)


def test_no_chat_model_at_all_explains_how_to_get_one(monkeypatch):
    llm = _offline_lmstudio(monkeypatch, [M("embed", 1e8, kind="embedding")])
    with pytest.raises(LLMError, match="blaah models get qwen/qwen3.5-4b"):
        llm.ensure_loaded()


# ----- `blaah models use` -----
def test_models_use_saves_only_downloaded_models(monkeypatch, tmp_path, capsys):
    saved = {}
    fake = FakeProvider(LlmCfg(provider="fake"))
    monkeypatch.setattr(cli, "_llm", lambda settings: fake)
    monkeypatch.setattr(cli, "save_user_settings", lambda updates: saved.update(updates) or tmp_path / "c.toml")
    settings = Settings.load(tmp_path / "none.toml")
    assert cli.cmd_models(settings, ["use", "fake/model"]) == 0 and saved == {"llm": {"model": "fake/model"}}
    saved.clear()
    assert cli.cmd_models(settings, ["use", "not/downloaded"]) == 1 and saved == {}
    assert "blaah models get not/downloaded" in capsys.readouterr().out
