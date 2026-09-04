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
    DATALAB_API_URL: str = "https://api.datalab.to/v1/marker"
    DATALAB_TIMEOUT: float = 60.0

    # Vision & LLM Model Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    SOLVER_MODEL: str = "gpt-4o"
    VISION_MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.1

    # Retry and Loop Policy
    MAX_RETRIES: int = 3
    OCR_MOCK_FALLBACK: bool = True

    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent


settings = Settings()
