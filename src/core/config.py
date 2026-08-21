import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_root(marker: str = "pyproject.toml") -> Path:
    env_root = os.environ.get("APP_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent

    raise FileNotFoundError(
        f"Project root marker '{marker}' not found in any parent of {current}. "
        "Set the APP_ROOT environment variable to specify the project root."
    )


ROOT_DIR = _find_root()
ENV_FILE = ROOT_DIR / ".env"


class Environment(StrEnum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    llm_api_key: SecretStr = SecretStr("")
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-120b"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 3
    llm_retry_initial_delay: float = 0.5
    llm_retry_backoff_factor: float = 2.0
    llm_max_steps: int = 50

    worker_concurrency: int = 4
    task_queue_max_size: int = 100
    shutdown_drain_timeout_seconds: float = 30.0

    @model_validator(mode="after")
    def _validate_settings(self) -> Settings:
        if self.environment is Environment.PRODUCTION and self.debug:
            raise ValueError("DEBUG must not be enabled in PRODUCTION environment.")
        if (
            self.environment in (Environment.PRODUCTION, Environment.STAGING)
            and not self.llm_api_key.get_secret_value()
        ):
            raise ValueError(
                "LLM_API_KEY is required in PRODUCTION and STAGING environments."
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
