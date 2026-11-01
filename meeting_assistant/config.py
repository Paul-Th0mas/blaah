from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
# Personal settings (your name, your model) live outside the repo and override config.toml.
USER_CONFIG_PATH = Path(os.environ.get("BLAAH_CONFIG", "~/.config/blaah/config.toml")).expanduser()


class ConfigError(ValueError):
    """A config file that can't be read, or a value of the wrong kind."""


@dataclass
class UserCfg:
    names: list[str] = field(default_factory=list)  # set with `blaah setup` or the first-run prompt


@dataclass
class AudioCfg:
    input_device: str = ""
    # "" = mic only; "all" = everything the Mac plays; or part of an app's bundle id ("brave",
    # "chrome", "zoom", "teams", "slack") to tap just that app. Mixed with the mic.
    system_audio: str = ""
    system_gain: float = 1.0
    mic_gain: float = 1.0
    prefer_builtin_mic: bool = True
    sample_rate: int = 16000
    chunk_seconds: float = 2.0


@dataclass
class SttCfg:
    model: str = "mlx-community/parakeet-tdt-0.6b-v3"
    left_context: int = 256
    right_context: int = 256


@dataclass
class LlmCfg:
    provider: str = "lmstudio"  # see meeting_assistant/providers.py; plugins can add more
    base_url: str = "http://localhost:1234"
    model: str = "qwen/qwen3.5-4b"
    context_length: int = 4096
    parallel: int = 1
    auto_load: bool = True
    unload_others: bool = True
    temperature: float = 0.2
    max_notes_tokens: int = 700
    max_brief_tokens: int = 350
    # Any other [llm] keys, for the provider: a plugin's own settings (api_key, keep_alive, …).
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackerCfg:
    interval_seconds: float = 45.0
    min_new_words: int = 40
    recent_window_seconds: float = 180.0


@dataclass
class TriggerCfg:
    name_cooldown_seconds: float = 30.0
    notify: bool = True
    auto_brief: bool = True


@dataclass
class SpeakersCfg:
    enabled: bool = True
    model: str = "models/nemo_en_titanet_small.onnx"
    threshold: float = 0.35
    remember_voices: bool = False
    people: list[str] = field(default_factory=list)


@dataclass
class StorageCfg:
    meetings_dir: str = "meetings"


@dataclass
class Settings:
    user: UserCfg = field(default_factory=UserCfg)
    audio: AudioCfg = field(default_factory=AudioCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    llm: LlmCfg = field(default_factory=LlmCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    triggers: TriggerCfg = field(default_factory=TriggerCfg)
    speakers: SpeakersCfg = field(default_factory=SpeakersCfg)
    storage: StorageCfg = field(default_factory=StorageCfg)
    path: Path | None = None

    @classmethod
    def load(cls, path: Path | None = None, user_path: Path | None = USER_CONFIG_PATH) -> "Settings":
        """Defaults, overlaid by the shared config.toml, overlaid by your personal config.

        An explicit `path` loads only that file (tests, `--config`); unknown keys are ignored.
        """
        files = [path] if path else [DEFAULT_CONFIG_PATH] + ([user_path] if user_path else [])
        settings = cls(path=path or DEFAULT_CONFIG_PATH)
        for file in files:
            try:
                data: dict[str, Any] = tomllib.loads(file.read_text()) if file.exists() else {}
            except tomllib.TOMLDecodeError as e:
                raise ConfigError(f"{file}: {e}") from None
            for f in fields(cls):
                if f.name != "path" and isinstance(data.get(f.name), dict):
                    _overlay(getattr(settings, f.name), data[f.name], f"{file.name} [{f.name}]")
        return settings

    @property
    def meetings_dir(self) -> Path:
        p = Path(self.storage.meetings_dir).expanduser()
        return p if p.is_absolute() else PROJECT_ROOT / p


def _overlay(section: Any, raw: dict[str, Any], where: str) -> None:
    defaults = {f.name: getattr(section, f.name) for f in fields(section) if f.name != "extra"}
    for key, value in raw.items():
        if key in defaults:
            setattr(section, key, _checked(value, defaults[key], f"{where} {key}"))
        elif hasattr(section, "extra"):
            section.extra[key] = value  # provider-specific; the provider checks it
        # other unknown keys are ignored (older or newer config files)


def _checked(value: Any, default: Any, name: str) -> Any:
    """`value` if it is the same kind as the default. A lone name where a list is expected
    ("Paul") becomes ["Paul"], and a whole number where a decimal is expected is fine."""
    if isinstance(default, bool):
        ok, kind = isinstance(value, bool), "true or false"
    elif isinstance(default, list):
        value = [value] if isinstance(value, str) else value
        ok, kind = isinstance(value, list) and all(isinstance(v, str) for v in value), 'a list of names, e.g. ["Paul"]'
    elif isinstance(default, (int, float)):
        ok = isinstance(value, (int, float)) and not isinstance(value, bool)
        kind = "a number"
        if ok and isinstance(default, float):
            value = float(value)
        elif ok and not isinstance(value, int):
            ok, kind = False, "a whole number"
    else:
        ok, kind = isinstance(value, str), "text in quotes"
    if not ok:
        raise ConfigError(f"{name} should be {kind}, got {value!r}")
    return value


def save_user_settings(updates: dict[str, dict[str, Any]], path: Path = USER_CONFIG_PATH) -> Path:
    """Merge {section: {key: value}} into your personal config file and write it."""
    data: dict[str, dict[str, Any]] = tomllib.loads(path.read_text()) if path.exists() else {}
    for section, values in updates.items():
        data.setdefault(section, {}).update(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Your personal Blaah settings — override config.toml. Written by `blaah setup` / `blaah models use`.\n"]
    for section, values in data.items():
        lines.append(f"\n[{section}]\n")
        lines += [f"{key} = {_toml_value(value)}\n" for key, value in values.items()]
    path.write_text("".join(lines))
    return path


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if isinstance(value, dict):  # e.g. a provider plugin's table, kept as you wrote it
        return "{" + ", ".join(f"{json.dumps(str(k))} = {_toml_value(v)}" for k, v in value.items()) + "}"
    raise TypeError(f"can't write {type(value).__name__} to TOML")
