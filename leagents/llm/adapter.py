"""Provider-agnostic LLM adapter (DESIGN.md §4, §8 M1).

leagents must run regardless of model provider: agents depend only on the
LLMClient protocol. Provider SDKs are import-guarded optional extras, and
NullLLM keeps every flow functional with no LLM configured at all —
consumers treat an empty completion as "fall back to deterministic logic".
"""

from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        """Return the model's text reply, or "" when no model is available."""
        ...


class NullLLM:
    """No-LLM default: always returns "" so callers use their fallbacks."""

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        return ""


class AnthropicLLM:
    def __init__(self, model: str, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as exc:
            raise RuntimeError(
                "AnthropicLLM requires the 'anthropic' package: pip install anthropic"
            ) from exc
        kwargs = {"api_key": api_key} if api_key else {}
        self._client = anthropic.Anthropic(**kwargs)
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        kwargs = {"system": system} if system else {}
        message = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", "") == "text"
        )


class OpenAICompatibleLLM:
    """OpenAI SDK against any OpenAI-compatible endpoint (OpenAI, vLLM, Ollama, ...)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "OpenAICompatibleLLM requires the 'openai' package: pip install openai"
            ) from exc
        # explicit key > OPENAI_API_KEY env > keyless (local Ollama/vLLM)
        key = api_key or os.environ.get("OPENAI_API_KEY") or "not-needed"
        self._client = openai.OpenAI(base_url=base_url, api_key=key)
        self.model = model

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 1024) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        reply = self._client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_tokens
        )
        return reply.choices[0].message.content or ""


# Google's OpenAI-compatible endpoint for the Gemini API (free tier available).
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def make_llm(spec: str | None) -> LLMClient:
    """Build a client from a config spec string.

    None                                        -> NullLLM (deterministic fallbacks)
    "anthropic:claude-sonnet-5"                 -> Anthropic API (ANTHROPIC_API_KEY)
    "gemini:gemini-2.5-flash"                   -> Gemini API (GEMINI_API_KEY)
    "openai:gpt-5.2"                            -> OpenAI API (OPENAI_API_KEY)
    "openai:qwen3@http://localhost:11434/v1"    -> any OpenAI-compatible server
    """
    if not spec:
        return NullLLM()
    provider, _, rest = spec.partition(":")
    if not rest:
        raise ValueError(f"llm spec must be 'provider:model[@base_url]', got {spec!r}")
    model, _, base_url = rest.partition("@")
    if provider == "anthropic":
        return AnthropicLLM(model)
    if provider == "gemini":
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("llm 'gemini:*' needs GEMINI_API_KEY (or GOOGLE_API_KEY) set")
        return OpenAICompatibleLLM(model, base_url=base_url or _GEMINI_OPENAI_BASE,
                                   api_key=key)
    if provider == "openai":
        return OpenAICompatibleLLM(model, base_url=base_url or None)
    raise ValueError(
        f"unknown LLM provider {provider!r} (use 'anthropic', 'gemini', or 'openai')"
    )
