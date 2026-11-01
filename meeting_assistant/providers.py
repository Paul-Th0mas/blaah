"""Model providers: where Blaah's notes, briefs and summaries are generated.

Built in: "lmstudio" (LM Studio, required for now). Choose one with `[llm] provider` in a
config file; choose its model with `blaah models use <key>`, the `m` key in the app, or
`[llm] model`.

Writing a provider plugin
-------------------------
A plugin is any installed Python package that exposes a provider class through the
entry-point group "blaah.providers":

    # in the plugin's pyproject.toml
    [project.entry-points."blaah.providers"]
    ollama = "blaah_ollama:OllamaProvider"

The class is built with the `[llm]` settings (an `LlmCfg`) and implements `Provider` below.
Then `provider = "ollama"` in a config file selects it. Settings of the plugin's own go in
`[llm]` too (`keep_alive = "10m"`); whatever Blaah doesn't know arrives in `cfg.extra`.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Callable, Iterator, NamedTuple, Protocol, runtime_checkable

from .config import LlmCfg

ENTRY_POINT_GROUP = "blaah.providers"
BUILTIN = {"lmstudio": "meeting_assistant.llm:LMStudio"}

# Shown by `blaah models` and `blaah setup` when choosing a model. Measured on a 16 GB M5.
RECOMMENDED = [
    ("qwen/qwen3.5-4b", "3 GB — the default: fast (~4 s briefs), 13/13 on the meeting Q&A test"),
    ("qwen/qwen3.5-9b", "6.5 GB — better judgement, about twice as slow on a 16 GB Mac"),
]


class ModelInfo(NamedTuple):
    key: str  # what goes in `[llm] model`
    name: str  # human-readable
    size_gb: float
    loaded: bool
    params: str = ""


@runtime_checkable
class Provider(Protocol):
    """What Blaah needs from a model backend."""

    name: str
    cfg: LlmCfg  # the active model is cfg.model; the app changes it to switch models
    last_stats: dict  # timing info from the latest call (tokens_per_second, …), may be empty

    def ensure_loaded(self, on_status: Callable[[str], None] = ...) -> None:
        """Make cfg.model ready to answer (start servers, load, warm up). Raise LLMError if not."""

    def complete(self, system: str, prompt: str, *, max_tokens: int = ..., temperature: float | None = ...): ...

    def stream(self, system: str, prompt: str, *, max_tokens: int = ..., temperature: float | None = ...) -> Iterator[str]:
        """Yield the reply as it is generated."""

    def list_models(self) -> list[ModelInfo]:
        """Chat models this provider can use right now."""

    def download(self, key: str, on_line: Callable[[str], None] = ...) -> None:
        """Fetch a model so it can be used (may raise NotImplementedError)."""


def provider_names() -> dict[str, str]:
    """name -> "module:Class" for built-in providers and installed plugins."""
    found = dict(BUILTIN)
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        found.setdefault(ep.name, ep.value)
    return found


def make_llm(cfg: LlmCfg) -> Provider:
    """The provider named in `[llm] provider`, set up with the `[llm]` settings."""
    available = provider_names()
    target = available.get(cfg.provider)
    if target is None:
        raise ValueError(f"unknown provider {cfg.provider!r}; available: {', '.join(sorted(available))}")
    module, _, attr = target.partition(":")
    cls = getattr(importlib.import_module(module), attr)
    return cls(cfg)
