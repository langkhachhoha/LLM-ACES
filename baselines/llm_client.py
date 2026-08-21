"""Minimal OpenAI-compatible chat client (OpenRouter by default).

Every LLM-touching part of the reproduction -- LLM-only, LLM-ODE, LLM-ACES and
the GPT-4o-mini symbolic-accuracy judge -- goes through this one client so that
API keys, retries, rate limiting and call accounting are handled in a single
place.

Credentials are read from ``.env`` in the repo root (or the environment):

    OPENAI_API_KEY=sk-or-v1-...        # an OpenRouter key works as-is
    OPENAI_BASE_URL=https://openrouter.ai/api/v1
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_ENV_LOADED = False


def load_dotenv(path: str | Path | None = None) -> None:
    """Populate os.environ from a .env file (does not override existing vars)."""
    global _ENV_LOADED
    if _ENV_LOADED and path is None:
        return
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    if path is None:
        _ENV_LOADED = True


def resolve_api_key() -> str:
    load_dotenv()
    key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
           or os.environ.get("API_KEY"))
    if not key:
        raise EnvironmentError(
            "No API key found. Put OPENAI_API_KEY=... in .env or export it."
        )
    return key


def resolve_base_url() -> str:
    load_dotenv()
    return (os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
            or DEFAULT_BASE_URL).rstrip("/")


class LLMClient:
    """Chat-completions client with retries and a call counter."""

    def __init__(self, model: str, temperature: float = 1.0, max_tokens: int = 1024,
                 base_url: str | None = None, api_key: str | None = None,
                 max_retries: int = 5, timeout: float = 180.0, log_path: str | Path | None = None):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = (base_url or resolve_base_url()).rstrip("/")
        self.api_key = api_key or resolve_api_key()
        self.max_retries = max_retries
        self.timeout = timeout
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_completion_tokens = 0
        self.log_path = Path(log_path) if log_path else None

    # -- low level --------------------------------------------------------
    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/scientific-discovery/LLM-ACES",
                "X-Title": "LLM-ACES reproduction",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _log(self, prompt, responses) -> None:
        if self.log_path is None:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "call": self.n_calls, "model": self.model,
                    "prompt": prompt, "responses": responses,
                }) + "\n")
        except Exception:
            pass

    # -- public -----------------------------------------------------------
    def chat(self, prompt: str, n: int = 1, temperature: float | None = None,
             system: str | None = None, max_tokens: int | None = None) -> list[str]:
        """Return ``n`` completions for ``prompt``. Counts as ``1`` LLM call."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if n > 1:
            payload["n"] = n

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                data = self._post(payload)
                if "error" in data and data.get("error"):
                    raise RuntimeError(f"API error: {data['error']}")
                choices = data.get("choices") or []
                outs = [(c.get("message") or {}).get("content") or "" for c in choices]
                if not outs:
                    raise RuntimeError(f"Empty response: {str(data)[:400]}")
                usage = data.get("usage") or {}
                self.n_prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.n_completion_tokens += int(usage.get("completion_tokens") or 0)
                self.n_calls += 1
                self._log(prompt, outs)
                # Some providers ignore `n`; fall back to sequential sampling.
                while len(outs) < n:
                    outs.extend(self.chat(prompt, n=1, temperature=temperature,
                                          system=system, max_tokens=max_tokens))
                return outs[:n]
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_exc}")

    def stats(self) -> dict:
        return {
            "llm_calls": self.n_calls,
            "prompt_tokens": self.n_prompt_tokens,
            "completion_tokens": self.n_completion_tokens,
            "model": self.model,
        }


# Convenience: the two model ids used throughout this reproduction.
GPT_MODEL = "openai/gpt-4o-mini-2024-07-18"
QWEN_MODEL = "qwen/qwen3-30b-a3b-instruct-2507"
