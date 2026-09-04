"""Application Configuration using Pydantic Settings."""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System-wide configuration settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Datalab OCR API
    DATALAB_API_KEY: Optional[str] = None
    DATALAB_API_URL: str = "https://www.datalab.to/api/v1/convert"
    DATALAB_TIMEOUT: float = 60.0

    # OpenRouter LLM / Vision Provider Settings
    OPENROUTER_API_KEY: Optional[str] = None
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    LLM_MODEL: str = "anthropic/claude-3.5-sonnet"
    TEMPERATURE: float = 0.1

    # Retry and Loop Policy
    MAX_RETRIES: int = 3
    OCR_MOCK_FALLBACK: bool = True

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

    @property
    def openrouter_api_key(self) -> Optional[str]:
        return self.OPENROUTER_API_KEY

    @property
    def openrouter_base_url(self) -> str:
        return self.OPENROUTER_BASE_URL

    @property
    def llm_model(self) -> str:
        return self.LLM_MODEL


settings = Settings()
