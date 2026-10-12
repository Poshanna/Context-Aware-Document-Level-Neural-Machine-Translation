"""Application-wide settings loaded from environment variables.

All credentials MUST be provided via environment variables or a ``.env``
file.  No secrets are hardcoded here — placeholders only.
"""

from __future__ import annotations

from typing import List

from pydantic import AnyHttpUrl, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --- General --------------------------------------------------------
    PROJECT_NAME: str = "context-aware-nmt"
    ENVIRONMENT: str = Field("development", pattern="^(development|staging|production)$")
    DEBUG: bool = True

    # --- Backend --------------------------------------------------------
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- Security -------------------------------------------------------
    JWT_SECRET_KEY: SecretStr = Field(..., description="Source: .env only.")
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    PASSWORD_HASH_SCHEME: str = Field("argon2", pattern="^(argon2|bcrypt)$")

    # --- Postgres -------------------------------------------------------
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "context_aware_nmt"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: SecretStr = Field(..., description="Source: .env only.")
    DATABASE_URL: str | None = Field(
        default=None,
        description=(
            "Fully-qualified DB URL. If not provided, one is constructed "
            "from the individual POSTGRES_* fields."
        ),
    )

    def build_database_url(self) -> str:
        if self.DATABASE_URL:
            return self.DATABASE_URL
        pw = self.POSTGRES_PASSWORD.get_secret_value()
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{pw}"
            f"@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # --- Uploads --------------------------------------------------------
    UPLOAD_DIR: str = "/app/uploads"
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: List[str] = ["txt", "pdf", "docx"]

    # --- Hugging Face / Runtime ----------------------------------------
    HF_HOME: str = "/app/.cache/hf"
    TRANSFORMERS_OFFLINE: int = 0
    FORCE_DEVICE: str = Field("auto", pattern="^(auto|cpu|cuda)$")
    DEMO_MODE: int = 1  # 1 => allow CPU fallback with warning banner

    # --- Translation model names (CHECKPOINTS ONLY — not auto-downloaded)
    MODEL_INDICTRANS2_EN_TE: str = "ai4bharat/indictrans2-en-te-1B"
    MODEL_INDICTRANS2_TE_EN: str = "ai4bharat/indictrans2-te-en-1B"
    MODEL_NLLB200: str = "facebook/nllb-200-distilled-600M"
    MODEL_DEFAULT: str = Field("nllb200", pattern="^(indictrans2|nllb200)$")
    MODEL_NLLB200_LOCAL_PATH: str | None = None
    TRANSLATION_SOURCE_LANGUAGE: str = Field("en", pattern="^(en|te)$")
    TRANSLATION_TARGET_LANGUAGE: str = Field("te", pattern="^(en|te)$")
    TRANSLATION_MAX_INPUT_LENGTH: int = Field(512, ge=8, le=2048)
    TRANSLATION_MAX_OUTPUT_LENGTH: int = Field(256, ge=1, le=1024)
    TRANSLATION_SEED: int = 42

    # --- Context retrieval ---------------------------------------------
    CONTEXT_STRATEGY: str = Field("sbert", pattern="^(sbert|lastn)$")
    CONTEXT_K: int = Field(3, ge=0, le=20)
    CONTEXT_WINDOW: int = Field(20, ge=1, le=200)
    CONTEXT_MIN_SCORE: float = Field(0.2, ge=0.0, le=1.0)
    SENTENCE_TRANSFORMER_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- Evaluation -----------------------------------------------------
    ENABLE_COMET: int = Field(0, ge=0, le=1)
    COMET_MODEL: str = "Unbabel/wmt22-comet-da"
    BERT_SCORE_MODEL: str = "bert-base-multilingual-cased"
    EVAL_SEED: int = 42
    EVAL_ALLOW_LOCAL_PATHS: int = Field(0, ge=0, le=1)

    # --- Dataset --------------------------------------------------------
    DATASET_SEED: int = 42
    SAMANANTAR_DATA_PATH: str | None = Field(
        default=None,
        description=(
            "Absolute or project-relative path to the local Samanantar en-te "
            "extracted folder. Leave None until you manually download the "
            "tarball per docs/dataset.md. Scripts fail fast with a clear "
            "message when this path points to a missing location."
        ),
    )


_settings_singleton: Settings | None = None


def get_settings() -> Settings:
    """Return the cached ``Settings`` instance.

    The pydantic-settings machinery loads once from env / .env.  Callers
    should always go through this helper so that tests can monkeypatch a
    single location.
    """
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = Settings()  # type: ignore[call-arg]
    return _settings_singleton
