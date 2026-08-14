import os
from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_root(marker: str = "pyproject.toml") -> Path:
    env_root = os.environ.get("APP_ROOT")
    if env_root:
        return Path(env_root).resolve()

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / marker).exists():
            return parent

    return current.parents[2] if len(current.parents) > 2 else current.parent


ROOT_DIR = _find_root()
ENV_FILE = ROOT_DIR / ".env"


class Environment(str, Enum):
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

    @model_validator(mode="after")
    def _validate_debug_in_production(self) -> Settings:
        if self.environment is Environment.PRODUCTION and self.debug:
            raise ValueError("DEBUG must not be enabled in PRODUCTION environment.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
