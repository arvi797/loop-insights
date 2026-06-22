"""LLM provider abstraction.

A thin seam over OpenAI and Gemini so the narrative logic doesn't care which model
is configured, and so the eval harness can swap a fake provider in. Both real
providers are asked for structured output matching the NarrativeDraft schema, which
is derived directly from the Pydantic model (no hand-maintained schema to drift).
"""

from __future__ import annotations

from typing import Protocol

from app.config import Settings
from app.models import NarrativeDraft


class LLMProvider(Protocol):
    async def draft_narrative(self, system: str, user: str) -> NarrativeDraft:
        """Return a structured narrative draft. Implementations enforce the schema."""
        ...


class OpenAIProvider:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def draft_narrative(self, system: str, user: str) -> NarrativeDraft:
        # .parse() uses OpenAI structured outputs (strict json_schema derived from the
        # Pydantic model) and returns a parsed instance — schema enforced at decode.
        kwargs: dict = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": NarrativeDraft,
        }
        # Reasoning models (gpt-5.x) only allow the default temperature and reject an
        # explicit value. Send it for models that accept it; otherwise omit it.
        if self._temperature is not None and not self._is_fixed_temperature_model():
            kwargs["temperature"] = self._temperature

        resp = await self._client.chat.completions.parse(**kwargs)
        parsed = resp.choices[0].message.parsed
        if parsed is None:  # model refused or produced no parseable content
            raise ValueError("OpenAI returned no parseable narrative.")
        return parsed

    def _is_fixed_temperature_model(self) -> bool:
        """gpt-5.x and o-series reasoning models pin temperature to its default."""
        m = self._model.lower()
        return m.startswith(("gpt-5", "o1", "o3", "o4"))


class GeminiProvider:
    def __init__(self, api_key: str, model: str, temperature: float = 0.2):
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._temperature = temperature

    async def draft_narrative(self, system: str, user: str) -> NarrativeDraft:
        from google.genai import types

        # Gemini enforces the schema at decode via response_schema; passing the
        # Pydantic model directly keeps it in sync with OpenAI and the validator.
        resp = await self._client.aio.models.generate_content(
            model=self._model,
            contents=f"{system}\n\n{user}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=NarrativeDraft,
                temperature=self._temperature,
            ),
        )
        # The SDK parses to the model when response_schema is a type; fall back to
        # validating the raw JSON text if `parsed` is unavailable.
        if getattr(resp, "parsed", None) is not None:
            return resp.parsed
        return NarrativeDraft.model_validate_json(resp.text or "{}")


class LLMNotConfiguredError(RuntimeError):
    """Raised when the narrative endpoint is called without a usable LLM key."""


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMNotConfiguredError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set."
            )
        return OpenAIProvider(
            settings.openai_api_key, settings.openai_model, settings.llm_temperature
        )
    if settings.llm_provider == "gemini":
        if not settings.google_api_key:
            raise LLMNotConfiguredError(
                "LLM_PROVIDER=gemini but GOOGLE_API_KEY is not set."
            )
        return GeminiProvider(
            settings.google_api_key, settings.gemini_model, settings.llm_temperature
        )
    raise LLMNotConfiguredError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")
