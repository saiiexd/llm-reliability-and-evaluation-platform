"""
Configuration settings for the FastAPI backend application.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ApiSettings(BaseSettings):
    """Configuration settings for the LLM Reliability API server."""

    model_config = SettingsConfigDict(
        env_prefix="LLM_RELIABILITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    api_v1_prefix: str = "/api/v1"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )
    persistence_dir: Path = Field(default_factory=lambda: Path("experiments"))
    environment: str = "development"
    debug: bool = False
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    allow_external_inference: bool = False

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> list[str]:
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if item]
                except json.JSONDecodeError:
                    pass
            return [part.strip() for part in value.split(",") if part.strip()]
        if isinstance(value, (list, tuple)):
            return [str(item).strip() for item in value if item]
        return ["http://localhost:5173", "http://127.0.0.1:5173"]

    @field_validator("persistence_dir", mode="before")
    @classmethod
    def parse_persistence_dir(cls, value: object) -> Path:
        if isinstance(value, Path):
            return value
        return Path(str(value))
