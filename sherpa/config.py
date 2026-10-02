"""Runtime configuration, loaded once from `.env` (see `.env.example`)."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": .env may also hold Kaggle-side values (NGROK_*) the app doesn't use
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM endpoint ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_basic_auth: SecretStr | None = None
    chat_model: str = "qwen2.5:7b"
    embed_model: str = "nomic-embed-text"
    embed_base_url: str | None = None
    embed_basic_auth: SecretStr | None = None
    num_ctx: int = 8192
    request_timeout_s: float = 180.0
    llm_max_retries: int = Field(default=3, ge=1)

    # --- temperatures ---
    temp_planner: float = 0.2
    temp_explainer: float = 0.3
    temp_quiz_gen: float = 0.4
    temp_quiz_grade: float = 0.1
    temp_coach: float = 0.5

    # --- curriculum ---
    min_topics: int = 3
    max_topics: int = 7
    max_plan_revisions: int = 5

    # --- explainer ---
    explainer_max_tool_rounds: int = 5
    retrieval_top_k: int = 4
    retrieval_min_similarity: float = 0.35
    fetch_max_chars: int = 6000
    web_max_results: int = 5

    # --- quiz / progress ---
    quiz_num_questions: int = 5
    quiz_min_questions: int = 3
    quiz_verify_key: bool = True
    pass_threshold: float = 0.7
    max_attempts_per_topic: int = Field(default=3, ge=1)

    # --- paths ---
    notes_dir: Path = Path("./notes")
    data_dir: Path = Path("./data")

    @field_validator("ollama_base_url", "embed_base_url", mode="before")
    @classmethod
    def _clean_url(cls, v):
        if v is None:
            return None
        v = str(v).strip().rstrip("/")
        if not v:
            return None
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"must start with http:// or https:// (got {v[:12]!r}...)")
        return v

    @field_validator("ollama_basic_auth", "embed_basic_auth", mode="before")
    @classmethod
    def _clean_auth(cls, v):
        if v is None:
            return None
        v = str(v).strip()
        if not v:
            return None
        if ":" not in v:
            raise ValueError("must be in the form user:password")
        return v

    @property
    def effective_embed_base_url(self) -> str:
        return self.embed_base_url or self.ollama_base_url

    @property
    def effective_embed_basic_auth(self) -> SecretStr | None:
        # Separate embed endpoint without its own auth → assume no auth (e.g. local Ollama)
        if self.embed_base_url:
            return self.embed_basic_auth
        return self.ollama_basic_auth

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"

    @property
    def checkpoints_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"


@lru_cache
def get_settings() -> Settings:
    return Settings()
