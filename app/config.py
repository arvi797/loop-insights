"""Application configuration, loaded from environment / .env.

Secrets live only in the environment — never in code or the repo. See .env.example.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # GitHub
    github_token: str | None = Field(default=None)
    max_pages: int = Field(default=10, ge=1, le=100)

    # LLM
    llm_provider: Literal["openai", "gemini"] = "openai"
    openai_api_key: str | None = None
    # gpt-5.5: strongest reasoning/instruction-following for the grounding +
    # calibration task. Slower than gpt-4.1; swap back via OPENAI_MODEL if latency
    # matters for the demo. Structured outputs supported.
    openai_model: str = "gpt-5.5"
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    # Low temperature: the narrative should be stable and faithful, not creative.
    llm_temperature: float = 0.2

    # App
    default_repo: str = "fastapi/fastapi"
    database_url: str = "sqlite:///./data/loop.db"
    log_level: str = "INFO"

    @property
    def sqlite_path(self) -> str:
        """Extract the filesystem path from a sqlite:/// URL."""
        prefix = "sqlite:///"
        if self.database_url.startswith(prefix):
            return self.database_url[len(prefix) :]
        return self.database_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
