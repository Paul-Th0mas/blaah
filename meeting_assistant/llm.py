from __future__ import annotations

import http.client
import json
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterator

from .config import LlmCfg
from .providers import RECOMMENDED, ModelInfo


class LLMError(RuntimeError):
    pass


LMS_MISSING = (
    "LM Studio's `lms` command isn't on your PATH — install LM Studio (https://lmstudio.ai), "
    "open it once, then run ~/.lmstudio/bin/lms bootstrap"
)
PROGRESS_SECONDS = 2.0  # `lms get` redraws its progress bar many times a second; show one line per this


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    ttft: float = 0.0  # time to first token, seconds


class LMStudio:
    """The built-in provider: LM Studio's native REST API (`/api/v1`).

    Used instead of LM Studio's OpenAI-compatible endpoint because it lets us turn a
    model's reasoning/thinking off per request (essential for fast briefs with
    Qwen/Gemma thinking models) and reports timing stats with every response.
    """

    name = "lmstudio"

    def __init__(self, cfg: LlmCfg):
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")
        self.reasoning_off = True  # refined by ensure_loaded() from the model's capabilities
        self.last_stats: dict = {}

    # ----- lifecycle -----
    def ensure_server(self, on_status: Callable[[str], None] = lambda _s: None) -> None:
        if self.server_up():
            return
        on_status("starting LM Studio server…")
        self._lms("server", "start", timeout=60)
        for _ in range(30):
            if self.server_up():
                return
            time.sleep(1)
        raise LLMError(f"LM Studio server not reachable at {self.base}; start it with `lms server start`")

    def ensure_loaded(self, on_status: Callable[[str], None] = lambda _s: None) -> None:
        """Start the server, load the configured model and warm it up, as needed.

        If the configured model isn't downloaded, falls back to one that is (and says so),
        so a fresh install works with whatever chat model is already in LM Studio.
        """
        self.ensure_server(on_status)
        models = self.models()
        info = next((m for m in models if m.get("key") == self.cfg.model), None)
        if info is None:
            fallback = pick_fallback(models)
            if fallback is None:
                raise LLMError(
                    f"no chat model in LM Studio — download one: blaah models get {RECOMMENDED[0][0]}"
                )
            on_status(f"{self.cfg.model} isn't downloaded — using {fallback['key']} (change with `blaah models`)")
            self.cfg.model, info = fallback["key"], fallback
        if self.cfg.unload_others and self.cfg.auto_load:
            for other in models:
                if other.get("type") == "llm" and other.get("key") != self.cfg.model:
                    for inst in other.get("loaded_instances") or []:
                        on_status(f"unloading {other['key']}…")
                        self._lms("unload", inst.get("id", other["key"]), timeout=120)
        reasoning = (info.get("capabilities") or {}).get("reasoning") or {}
        self.reasoning_off = "off" in (reasoning.get("allowed_options") or [])

        instances = info.get("loaded_instances") or []
        if instances and self.cfg.auto_load and self._needs_reload(instances[0]):
            # Extra parallel slots cost real speed on a memory-constrained Mac (measured
            # ~2× on 16 GB), and a too-small context would truncate our prompts.
            on_status("reloading model with the app's settings…")
            self._lms("unload", instances[0].get("id", self.cfg.model), timeout=120)
            instances = []

        if not instances:
            if not self.cfg.auto_load:
                raise LLMError(f"{self.cfg.model} is not loaded; run `lms load {self.cfg.model}`")
            on_status(f"loading {self.cfg.model} into LM Studio…")
            proc = self._lms(
                "load", self.cfg.model,
                "--context-length", str(self.cfg.context_length),
                "--parallel", str(self.cfg.parallel),
                "-y",
                timeout=900,
            )
            if proc.returncode != 0:
                raise LLMError(f"lms load failed: {(proc.stderr or proc.stdout).strip()[-300:]}")

        on_status("warming up…")
        self.complete("", "Reply with OK.", max_tokens=3)
        on_status("ready")

    def _needs_reload(self, instance: dict) -> bool:
        """True when the loaded instance has more parallel slots than configured or a
        context too small for our prompts. A *larger* context is left alone: LM Studio
        reports its own figure for MLX models (KV cache is allocated lazily there)."""
        loaded = instance.get("config") or {}
        return (
            int(loaded.get("context_length", 0)) < self.cfg.context_length
            or int(loaded.get("parallel", 1)) > self.cfg.parallel
        )

    def _lms(self, *args: str, timeout: float) -> subprocess.CompletedProcess:
        """Run the `lms` CLI. A missing CLI or a hang is an LLMError like any other LM Studio
        problem, so callers (and the app) report it instead of crashing on it."""
        try:
            return subprocess.run(["lms", *args], capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            raise LLMError(LMS_MISSING) from None
        except subprocess.TimeoutExpired:
            raise LLMError(f"`lms {' '.join(args[:2])}` didn't finish within {timeout:.0f} s") from None

    def server_up(self) -> bool:
        try:
            self._request("GET", "/api/v1/models", timeout=3).read()
            return True
        except LLMError:
            return False

    def models(self) -> list[dict]:
        with self._request("GET", "/api/v1/models", timeout=10) as r:
            return json.loads(r.read()).get("models", [])

    def model_info(self) -> dict | None:
        return next((m for m in self.models() if m.get("key") == self.cfg.model), None)

    def list_models(self) -> list[ModelInfo]:
        """Chat models downloaded in LM Studio (starts the server if needed)."""
        self.ensure_server()
        return [
            ModelInfo(
                key=m["key"],
                name=m.get("display_name") or m["key"],
                size_gb=round(m.get("size_bytes", 0) / 1e9, 1),
                loaded=bool(m.get("loaded_instances")),
                params=m.get("params_string") or "",
            )
            for m in self.models()
            if m.get("type") == "llm"
        ]

    def download(self, key: str, on_line: Callable[[str], None] = print) -> None:
        """Download a model from the LM Studio catalog (`lms get`)."""
        try:
            proc = subprocess.Popen(["lms", "get", key, "-y"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        except FileNotFoundError:
            raise LLMError(LMS_MISSING) from None
        last, shown, shown_at = "", "", 0.0
        for line in proc.stdout:
            line = line.strip().replace("\x1b", "")
            if not line or line == last:
                continue
            last = line
            if time.monotonic() - shown_at >= PROGRESS_SECONDS:
                on_line(line)
                shown, shown_at = line, time.monotonic()
        if last != shown:
            on_line(last)  # the final state ("Download complete" or the error)
        if proc.wait() != 0:
            raise LLMError(f"`lms get {key}` failed: {last}")

    # ----- inference -----
    def complete(
        self, system: str, prompt: str, *, max_tokens: int = 512, temperature: float | None = None
    ) -> LLMResult:
        body = self._body(system, prompt, max_tokens, temperature, stream=False)
        with self._request("POST", "/api/v1/chat", body, timeout=300) as r:
            try:
                return self._result(json.loads(r.read()))
            except (OSError, http.client.HTTPException, ValueError) as e:
                raise LLMError(f"LM Studio stopped mid-reply: {e or type(e).__name__}") from None

    def stream(
        self, system: str, prompt: str, *, max_tokens: int = 512, temperature: float | None = None
    ) -> Iterator[str]:
        """Yield text deltas as they are generated; stats land in `last_stats` at the end.

        Raises LLMError if the reply stops before LM Studio says it is complete (server quit or
        crashed), so a cut-off reply is never taken for a whole one.
        """
        body = self._body(system, prompt, max_tokens, temperature, stream=True)
        ended = False
        with self._request("POST", "/api/v1/chat", body, timeout=300) as r:
            try:
                for raw in r:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    if kind == "message.delta":
                        yield event.get("content", "")
                    elif kind == "chat.end":
                        self.last_stats = (event.get("result") or {}).get("stats") or {}
                        ended = True
                    elif kind == "error":
                        raise LLMError(str(event.get("error") or event))
            except (OSError, http.client.HTTPException) as e:
                raise LLMError(f"LM Studio stopped mid-reply: {e or type(e).__name__}") from None
        if not ended:
            raise LLMError("LM Studio closed the connection before the reply was complete")

    # ----- internals -----
    def _body(self, system: str, prompt: str, max_tokens: int, temperature: float | None, stream: bool) -> dict:
        body: dict = {
            "model": self.cfg.model,
            "input": prompt,
            "max_output_tokens": max_tokens,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "stream": stream,
        }
        if system:
            body["system_prompt"] = system
        if self.reasoning_off:
            body["reasoning"] = "off"
        return body

    def _result(self, data: dict) -> LLMResult:
        parts = []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            content = item.get("content", "")
            if isinstance(content, list):  # multimodal shape: [{"type": "text", "text": ...}]
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            parts.append(content)
        stats = data.get("stats") or {}
        self.last_stats = stats
        return LLMResult(
            text="".join(parts),
            input_tokens=int(stats.get("input_tokens", 0)),
            output_tokens=int(stats.get("total_output_tokens", 0)),
            tokens_per_second=float(stats.get("tokens_per_second", 0.0)),
            ttft=float(stats.get("time_to_first_token_seconds", 0.0)),
        )

    def _request(self, method: str, path: str, body: dict | None = None, timeout: float = 10):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            return urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:500]
            try:
                detail = json.loads(detail)["error"]["message"]
            except (ValueError, KeyError, TypeError):
                pass
            raise LLMError(f"LM Studio HTTP {e.code}: {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise LLMError(f"cannot reach LM Studio at {self.base}: {e}") from None


LLM = LMStudio  # older name, kept for scripts and tests


def pick_fallback(models: list[dict]) -> dict | None:
    """A downloaded chat model to use when the configured one is missing: the loaded one if
    any, else a recommended one, else the smallest (fastest)."""
    llms = [m for m in models if m.get("type") == "llm"]
    loaded = [m for m in llms if m.get("loaded_instances")]
    if loaded:
        return loaded[0]
    for key, _ in RECOMMENDED:
        match = next((m for m in llms if m.get("key") == key), None)
        if match:
            return match
    return min(llms, key=lambda m: m.get("size_bytes", float("inf")), default=None)
