"""Phase 2 runtime configuration.

Single source of truth for all settings. Env-driven via Pydantic settings.
Values default to docker-compose's expectations; override via .env or env vars.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    # API keys
    anthropic_api_key: str = Field(...)
    openai_api_key: str = Field(...)

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "knowledgebot"
    postgres_password: str = "knowledgebot"
    postgres_db: str = "knowledgebot"

    # Model IDs (held constant with Java repo for cross-stack comparison)
    chat_model: str = "claude-haiku-4-5"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # RAG params
    chunk_size: int = 800
    chunk_overlap: int = 0
    top_k: int = 4

    # Vector store — LlamaIndex prepends "data_" so the actual SQL table is "data_knowledge_bot"
    vector_table_name: str = "knowledge_bot"

    # Corpus
    corpus_path: Path = Path("corpus/ncc-2022-vol2.pdf")
    corpus_id: str = "ncc-2022-vol2"

    @property
    def vector_table_full(self) -> str:
        """Actual SQL table name as created by LlamaIndex's PGVectorStore."""
        return f"data_{self.vector_table_name}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
