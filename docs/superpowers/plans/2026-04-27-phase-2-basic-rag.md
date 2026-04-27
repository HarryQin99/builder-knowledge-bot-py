# Phase 2: Basic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the full 312-page NCC corpus queryable via `POST /ask` — ingest into pgvector via a CLI script, retrieve via LlamaIndex's `RetrieverQueryEngine`, return answer + token/cost metrics. Cleanly mirrors Java's Phase 2 scope and parameters.

**Architecture:** Two pipelines sharing one pgvector store. Ingestion (one-shot CLI): PDF → pymupdf → SentenceSplitter (800 tok, no overlap) → SHA-256 chunk IDs → SQL pre-check → OpenAI embed → pgvector. Query (per request): question → LlamaIndex `RetrieverQueryEngine` (top-K=4, cosine) → claude-haiku-4-5 → AnswerResponse with token/cost metrics.

**Tech Stack:** Python 3.12, FastAPI, LlamaIndex (`llama-index-core`, `llama-index-vector-stores-postgres`, `llama-index-llms-anthropic`, `llama-index-embeddings-openai`), psycopg v3, SQLAlchemy, Postgres + pgvector, Docker Compose.

**Spec:** `docs/superpowers/specs/2026-04-27-phase-2-basic-rag-design.md`

---

## File Structure

| Path | Action | Purpose |
|---|---|---|
| `pyproject.toml` | Modify | Add 6 new deps via `uv add`; lockfile regenerates |
| `.env.example` | Modify | Add Postgres env var placeholders |
| `src/knowledge_bot/config.py` | Create | Pydantic-settings: DB URL, model IDs, chunk size, top-K |
| `src/knowledge_bot/models.py` | Create | Pydantic models: AskRequest, AnswerMetrics, AnswerResponse |
| `src/knowledge_bot/pricing.py` | Create | ModelPricing dataclass + `cost_for()` function |
| `src/knowledge_bot/query.py` | Create | `build_query_engine()` factory — wires PGVectorStore + retriever + LLM |
| `src/knowledge_bot/ask_service.py` | Create | `AskService.answer()` — measures latency, captures tokens, computes cost |
| `src/knowledge_bot/routes/__init__.py` | Create | Empty package init |
| `src/knowledge_bot/routes/ask.py` | Create | `POST /ask` FastAPI router |
| `src/knowledge_bot/main.py` | Modify | Add lifespan that builds AskService; include ask router |
| `src/knowledge_bot/ingest.py` | Create | Ingestion CLI — runnable as `python -m knowledge_bot.ingest` |
| `tests/test_ask.py` | Create | TestClient + mocked AskService — 3 tests (200, 422, 503) |
| `Dockerfile` | Modify | Entrypoint script that runs ingest then uvicorn |
| `docker-compose.yml` | Modify | Pass `OPENAI_API_KEY` correctly; ensure healthcheck timing |
| `docs/observations/phase-2-findings.md` | Create | Pipeline numbers, sample query results, observed failure modes |
| `README.md` | Modify | Mark Phase 2 done; update phase table to show 8 phases |

Each Python file has one clear responsibility. Files mirror Java's controller/service/model/util/config layout.

---

## Task 1: Add Phase 2 dependencies

**Files:**
- Modify: `pyproject.toml`, `uv.lock`

- [ ] **Step 1: Add the LlamaIndex stack and Postgres drivers**

Run:
```bash
uv add llama-index-core llama-index-vector-stores-postgres llama-index-llms-anthropic llama-index-embeddings-openai 'psycopg[binary]' sqlalchemy pydantic-settings
```

Expected: `Resolved N packages`, `Installed M packages`. Six top-level adds.

- [ ] **Step 2: Verify imports**

Run:
```bash
uv run python -c "
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore
import psycopg
import sqlalchemy
from pydantic_settings import BaseSettings
print('ok')
"
```

Expected: prints `ok`. Any `ImportError` means a package name is wrong; STOP and report.

- [ ] **Step 3: Verify Phase 0 + Phase 1 tests still pass**

Run:
```bash
uv run pytest -v
```

Expected: `1 passed` (test_health_returns_ok); no failures.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "Phase 2: add LlamaIndex stack and Postgres deps

Adds llama-index-core, llama-index-vector-stores-postgres,
llama-index-llms-anthropic, llama-index-embeddings-openai,
psycopg v3, sqlalchemy, and pydantic-settings."
```

---

## Task 2: Update `.env.example` with Postgres vars

**Files:**
- Modify: `.env.example`

- [ ] **Step 1: Replace `.env.example` content**

Write `.env.example` with this content (overwrite):

```
# API keys
ANTHROPIC_API_KEY=sk-ant-api03-...your-key-here...
OPENAI_API_KEY=sk-proj-...your-key-here...

# Postgres (defaults match docker-compose.yml; override for non-Docker runs)
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=knowledgebot
POSTGRES_PASSWORD=knowledgebot
POSTGRES_DB=knowledgebot
```

- [ ] **Step 2: Update local `.env` (don't commit)**

The existing `.env` (copied from the Java repo earlier) has both API keys but no Postgres vars. Add them:

```bash
cat >> .env <<'EOF'
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=knowledgebot
POSTGRES_PASSWORD=knowledgebot
POSTGRES_DB=knowledgebot
EOF
```

Expected: silent success.

- [ ] **Step 3: Verify `.env` is still gitignored**

Run:
```bash
git status --short .env
```

Expected: empty output.

- [ ] **Step 4: Commit `.env.example` only**

```bash
git add .env.example
git commit -m "Phase 2: add Postgres env vars to .env.example

Defaults match docker-compose's host port 5433 mapping."
```

---

## Task 3: Create `config.py` (Pydantic settings)

**Files:**
- Create: `src/knowledge_bot/config.py`

- [ ] **Step 1: Write the config module**

Create `src/knowledge_bot/config.py` with this content:

```python
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
```

- [ ] **Step 2: Verify it imports and instantiates**

Run:
```bash
uv run python -c "from knowledge_bot.config import get_settings; s = get_settings(); print(s.chat_model, s.postgres_port, s.vector_table_full)"
```

Expected: `claude-haiku-4-5 5433 data_knowledge_bot`. If you get a validation error about missing API keys, your `.env` lost them — restore from the Java repo.

- [ ] **Step 3: Lint**

Run:
```bash
uv run ruff check src/knowledge_bot/config.py
```

Expected: `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_bot/config.py
git commit -m "Phase 2: add config.py (Pydantic settings)

Single source of runtime config: Postgres connection, model IDs,
chunk size, top-K, vector table name. Env-driven via .env."
```

---

## Task 4: Create `models.py` (Pydantic models)

**Files:**
- Create: `src/knowledge_bot/models.py`

- [ ] **Step 1: Write the models**

Create `src/knowledge_bot/models.py`:

```python
"""Phase 2 API contract: request and response shapes for /ask.

Mirrors Java's record shapes (snake_case in JSON because Python convention).
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)


class AnswerMetrics(BaseModel):
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float


class AnswerResponse(BaseModel):
    answer: str
    metrics: AnswerMetrics
```

- [ ] **Step 2: Verify**

Run:
```bash
uv run python -c "
from knowledge_bot.models import AskRequest, AnswerResponse, AnswerMetrics
req = AskRequest(question='hi')
print(req.question)
resp = AnswerResponse(answer='ok', metrics=AnswerMetrics(input_tokens=1, output_tokens=2, latency_ms=3, cost_usd=0.0001))
print(resp.model_dump_json())
"
```

Expected: prints `hi` then `{"answer":"ok","metrics":{"input_tokens":1,"output_tokens":2,"latency_ms":3,"cost_usd":0.0001}}`.

- [ ] **Step 3: Lint**

Run:
```bash
uv run ruff check src/knowledge_bot/models.py
```

Expected: `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_bot/models.py
git commit -m "Phase 2: add API request/response models

AskRequest, AnswerMetrics, AnswerResponse mirror Java's record shapes."
```

---

## Task 5: Create `pricing.py`

**Files:**
- Create: `src/knowledge_bot/pricing.py`

- [ ] **Step 1: Write the module**

Create `src/knowledge_bot/pricing.py`:

```python
"""Phase 2 cost calculator.

Per-token prices for the models we use. Unknown model raises loudly
to prevent silent miscalculation. Mirrors Java's ModelPricing enum.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float


# Anthropic Claude Haiku 4.5: $1 per million input tokens, $5 per million output tokens
# OpenAI text-embedding-3-small: $0.02 per million input tokens
MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(input_per_1k=0.001, output_per_1k=0.005),
    "text-embedding-3-small": ModelPricing(input_per_1k=0.00002, output_per_1k=0.0),
}


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD cost of a call. Raises if the model is not in MODEL_PRICING."""
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        raise ValueError(
            f"No pricing for model {model!r}; add it to MODEL_PRICING in pricing.py."
        )
    return (input_tokens * pricing.input_per_1k + output_tokens * pricing.output_per_1k) / 1000
```

- [ ] **Step 2: Verify**

Run:
```bash
uv run python -c "
from knowledge_bot.pricing import cost_for
print(cost_for('claude-haiku-4-5', 1000, 500))
"
```

Expected: `0.0035` (1000 × 0.001 / 1000 + 500 × 0.005 / 1000 = 0.001 + 0.0025 = 0.0035).

- [ ] **Step 3: Lint**

Run:
```bash
uv run ruff check src/knowledge_bot/pricing.py
```

Expected: `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_bot/pricing.py
git commit -m "Phase 2: add pricing.py

ModelPricing dataclass + cost_for() function. Mirrors Java's
ModelPricing enum. Unknown model raises ValueError."
```

---

## Task 6: Create `query.py` (query engine factory)

**Files:**
- Create: `src/knowledge_bot/query.py`

- [ ] **Step 1: Write the factory**

Create `src/knowledge_bot/query.py`:

```python
"""Phase 2 query engine factory.

Builds the LlamaIndex RetrieverQueryEngine once at startup with explicit
named components: PGVectorStore + OpenAIEmbedding + Anthropic LLM.
Returned alongside a TokenCountingHandler so the AskService can read
per-query token usage.
"""
from __future__ import annotations

from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic
from llama_index.vector_stores.postgres import PGVectorStore

from knowledge_bot.config import get_settings


def build_query_engine() -> tuple[RetrieverQueryEngine, TokenCountingHandler]:
    """Wire the RAG query path. Returns (engine, token_counter)."""
    settings = get_settings()

    embed_model = OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    llm = Anthropic(
        model=settings.chat_model,
        api_key=settings.anthropic_api_key,
    )

    pg_store = PGVectorStore.from_params(
        database=settings.postgres_db,
        host=settings.postgres_host,
        password=settings.postgres_password,
        port=settings.postgres_port,
        user=settings.postgres_user,
        table_name=settings.vector_table_name,
        embed_dim=settings.embedding_dim,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )

    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])

    index = VectorStoreIndex.from_vector_store(
        vector_store=pg_store,
        embed_model=embed_model,
        callback_manager=callback_manager,
    )
    retriever = index.as_retriever(similarity_top_k=settings.top_k)

    engine = RetrieverQueryEngine.from_args(
        retriever=retriever,
        llm=llm,
        callback_manager=callback_manager,
    )
    return engine, token_counter
```

- [ ] **Step 2: Lint and syntax-check (don't run end-to-end — postgres may not be up)**

Run:
```bash
uv run ruff check src/knowledge_bot/query.py
uv run python -c "from knowledge_bot.query import build_query_engine; print('imports ok')"
```

Expected: ruff passes; `imports ok` printed.

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_bot/query.py
git commit -m "Phase 2: add query engine factory

build_query_engine() wires PGVectorStore + OpenAIEmbedding + Anthropic
LLM into a RetrieverQueryEngine with explicit named components.
Returns the engine plus a TokenCountingHandler for per-query metrics."
```

---

## Task 7: Create `ask_service.py`

**Files:**
- Create: `src/knowledge_bot/ask_service.py`

- [ ] **Step 1: Write the service**

Create `src/knowledge_bot/ask_service.py`:

```python
"""Phase 2 service layer: takes a question, returns AnswerResponse with metrics.

Pure service — no FastAPI imports. Mockable in tests by replacing the
RetrieverQueryEngine and TokenCountingHandler with stubs.
"""
from __future__ import annotations

import time

from llama_index.core.callbacks import TokenCountingHandler
from llama_index.core.query_engine import RetrieverQueryEngine

from knowledge_bot.config import get_settings
from knowledge_bot.models import AnswerMetrics, AnswerResponse
from knowledge_bot.pricing import cost_for


class EmptyVectorStoreError(RuntimeError):
    """Raised when retrieval returns no source nodes (vector store is empty)."""


class AskService:
    def __init__(
        self,
        query_engine: RetrieverQueryEngine,
        token_counter: TokenCountingHandler,
    ):
        self._query_engine = query_engine
        self._token_counter = token_counter
        self._chat_model = get_settings().chat_model

    def answer(self, question: str) -> AnswerResponse:
        self._token_counter.reset_counts()
        t0 = time.perf_counter()
        response = self._query_engine.query(question)
        latency_ms = int((time.perf_counter() - t0) * 1000)

        if not response.source_nodes:
            raise EmptyVectorStoreError(
                "Vector store returned no chunks. "
                "Run `uv run python -m knowledge_bot.ingest` first."
            )

        input_tokens = self._token_counter.prompt_llm_token_count
        output_tokens = self._token_counter.completion_llm_token_count
        cost = cost_for(self._chat_model, input_tokens, output_tokens)

        return AnswerResponse(
            answer=str(response),
            metrics=AnswerMetrics(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                cost_usd=round(cost, 6),
            ),
        )
```

- [ ] **Step 2: Lint and syntax-check**

Run:
```bash
uv run ruff check src/knowledge_bot/ask_service.py
uv run python -c "from knowledge_bot.ask_service import AskService, EmptyVectorStoreError; print('ok')"
```

Expected: ruff passes; `ok` printed.

- [ ] **Step 3: Commit**

```bash
git add src/knowledge_bot/ask_service.py
git commit -m "Phase 2: add AskService

Service layer that wraps RetrieverQueryEngine: resets the token
counter, measures wall-clock latency, captures token counts via
TokenCountingHandler, computes USD cost via pricing.cost_for()."
```

---

## Task 8: Create `routes/__init__.py` + `routes/ask.py`

**Files:**
- Create: `src/knowledge_bot/routes/__init__.py`
- Create: `src/knowledge_bot/routes/ask.py`

- [ ] **Step 1: Create the routes package**

Run:
```bash
mkdir -p src/knowledge_bot/routes
```

Create `src/knowledge_bot/routes/__init__.py` (empty):

```python
```

(One blank line; an empty file is fine.)

- [ ] **Step 2: Write the ask router**

Create `src/knowledge_bot/routes/ask.py`:

```python
"""POST /ask FastAPI router."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status

from knowledge_bot.ask_service import AskService, EmptyVectorStoreError
from knowledge_bot.models import AnswerResponse, AskRequest

router = APIRouter()


def get_ask_service(request: Request) -> AskService:
    svc = getattr(request.app.state, "ask_service", None)
    if svc is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="AskService not initialised. Check Postgres and run `python -m knowledge_bot.ingest`.",
        )
    return svc


@router.post("/ask", response_model=AnswerResponse)
def ask(req: AskRequest, service: AskService = Depends(get_ask_service)) -> AnswerResponse:
    try:
        return service.answer(req.question)
    except EmptyVectorStoreError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
```

- [ ] **Step 3: Lint**

Run:
```bash
uv run ruff check src/knowledge_bot/routes/
```

Expected: `All checks passed!`.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_bot/routes/
git commit -m "Phase 2: add /ask router

Thin FastAPI router. Validates AskRequest, delegates to AskService,
returns AnswerResponse. EmptyVectorStoreError -> 503."
```

---

## Task 9: Wire lifespan + ask router into `main.py`

**Files:**
- Modify: `src/knowledge_bot/main.py`

- [ ] **Step 1: Replace `main.py`**

Overwrite `src/knowledge_bot/main.py` with:

```python
"""Phase 2 FastAPI app entrypoint.

The lifespan builds the LlamaIndex query engine once at startup and stashes
the AskService on app.state. If startup fails (e.g., Postgres unreachable),
the service is None and /ask returns 503 — useful for tests that override
the dependency without needing a live Postgres.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from knowledge_bot.ask_service import AskService
from knowledge_bot.query import build_query_engine
from knowledge_bot.routes.ask import router as ask_router

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        engine, counter = build_query_engine()
        app.state.ask_service = AskService(engine, counter)
        log.info("AskService initialised.")
    except Exception as e:  # noqa: BLE001
        log.warning("AskService init failed; /ask will return 503. Reason: %s", e)
        app.state.ask_service = None
    yield


app = FastAPI(title="Knowledge Bot", lifespan=lifespan)
app.include_router(ask_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
```

- [ ] **Step 2: Lint**

Run:
```bash
uv run ruff check src/knowledge_bot/main.py
```

Expected: `All checks passed!`.

- [ ] **Step 3: Verify the existing `/health` test still passes**

Run:
```bash
uv run pytest tests/test_health.py -v
```

Expected: `1 passed`. The lifespan will fail to connect to Postgres (it's not running), log the warning, and `/health` continues to work since it doesn't touch AskService.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_bot/main.py
git commit -m "Phase 2: add lifespan and include /ask router

Lifespan builds AskService at startup, tolerates failure (so tests run
without Postgres). /health unchanged. /ask 503s if AskService is None."
```

---

## Task 10: Create `tests/test_ask.py`

**Files:**
- Create: `tests/test_ask.py`

- [ ] **Step 1: Write the tests**

Create `tests/test_ask.py`:

```python
"""Phase 2 /ask endpoint tests.

Uses FastAPI's TestClient with the AskService dependency overridden
to a mock — no Postgres, no live LlamaIndex calls, no real LLMs.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from knowledge_bot.ask_service import AskService, EmptyVectorStoreError
from knowledge_bot.main import app
from knowledge_bot.models import AnswerMetrics, AnswerResponse
from knowledge_bot.routes.ask import get_ask_service


def _make_response() -> AnswerResponse:
    return AnswerResponse(
        answer="Mocked answer.",
        metrics=AnswerMetrics(
            input_tokens=100, output_tokens=50, latency_ms=42, cost_usd=0.000350
        ),
    )


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_ask_returns_answer_with_metrics(client):
    fake = MagicMock(spec=AskService)
    fake.answer.return_value = _make_response()
    app.dependency_overrides[get_ask_service] = lambda: fake
    try:
        response = client.post("/ask", json={"question": "What is the minimum ceiling height?"})
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "Mocked answer."
        assert body["metrics"]["input_tokens"] == 100
        assert body["metrics"]["output_tokens"] == 50
        assert body["metrics"]["latency_ms"] == 42
        assert body["metrics"]["cost_usd"] == 0.00035
        fake.answer.assert_called_once_with("What is the minimum ceiling height?")
    finally:
        app.dependency_overrides.clear()


def test_ask_missing_question_returns_422(client):
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_empty_store_returns_503(client):
    fake = MagicMock(spec=AskService)
    fake.answer.side_effect = EmptyVectorStoreError("Vector store returned no chunks.")
    app.dependency_overrides[get_ask_service] = lambda: fake
    try:
        response = client.post("/ask", json={"question": "anything"})
        assert response.status_code == 503
        assert "no chunks" in response.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()
```

- [ ] **Step 2: Run the tests**

Run:
```bash
uv run pytest tests/test_ask.py -v
```

Expected: `3 passed`. If the lifespan-startup warning shows up in the output, that's fine — the lifespan tolerates Postgres being unreachable.

- [ ] **Step 3: Run the full test suite**

Run:
```bash
uv run pytest -v
```

Expected: `4 passed` (1 health + 3 ask).

- [ ] **Step 4: Commit**

```bash
git add tests/test_ask.py
git commit -m "Phase 2: add /ask endpoint tests

Three TestClient tests covering success, missing question (422), and
empty-store (503) cases. Mocks AskService via dependency_overrides;
no Postgres or LLM required to run."
```

---

## Task 11: Create `ingest.py` (ingestion CLI)

**Files:**
- Create: `src/knowledge_bot/ingest.py`

- [ ] **Step 1: Write the CLI**

Create `src/knowledge_bot/ingest.py`:

```python
"""Phase 2 ingestion CLI.

Two-tier idempotency:
  1. md5 the PDF bytes; compare to corpus_metadata. If match -> exit early.
  2. On mismatch: chunk, compute SHA-256 chunk IDs, SELECT existing IDs,
     embed and INSERT only the new chunks. UPSERT corpus_metadata.

Run: uv run python -m knowledge_bot.ingest
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import psycopg
import pymupdf
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.postgres import PGVectorStore

from knowledge_bot.config import Settings, get_settings


CORPUS_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS corpus_metadata (
    corpus_id    TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    chunk_count  INT  NOT NULL,
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def deterministic_chunk_id(corpus_filename: str, page: int, idx: int, text: str) -> str:
    raw = f"{corpus_filename}|{page}|{idx}|{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_documents(corpus_path: Path) -> list[Document]:
    doc = pymupdf.open(str(corpus_path))
    try:
        return [
            Document(
                text=page.get_text(),
                metadata={"page_number": i + 1, "source": corpus_path.name},
            )
            for i, page in enumerate(doc)
        ]
    finally:
        doc.close()


def existing_corpus_hash(conn: psycopg.Connection, corpus_id: str) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT content_hash FROM corpus_metadata WHERE corpus_id = %s", (corpus_id,))
        row = cur.fetchone()
        return row[0] if row else None


def upsert_corpus_metadata(
    conn: psycopg.Connection, corpus_id: str, content_hash: str, chunk_count: int
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO corpus_metadata (corpus_id, content_hash, chunk_count, ingested_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (corpus_id) DO UPDATE
              SET content_hash = EXCLUDED.content_hash,
                  chunk_count = EXCLUDED.chunk_count,
                  ingested_at = EXCLUDED.ingested_at
            """,
            (corpus_id, content_hash, chunk_count),
        )
    conn.commit()


def existing_chunk_ids(conn: psycopg.Connection, ids: list[str], table: str) -> set[str]:
    if not ids:
        return set()
    try:
        with conn.cursor() as cur:
            cur.execute(f'SELECT id FROM "{table}" WHERE id = ANY(%s)', (ids,))
            return {row[0] for row in cur.fetchall()}
    except psycopg.errors.UndefinedTable:
        conn.rollback()
        return set()


def build_pg_store(settings: Settings) -> PGVectorStore:
    return PGVectorStore.from_params(
        database=settings.postgres_db,
        host=settings.postgres_host,
        password=settings.postgres_password,
        port=settings.postgres_port,
        user=settings.postgres_user,
        table_name=settings.vector_table_name,
        embed_dim=settings.embedding_dim,
        hnsw_kwargs={
            "hnsw_m": 16,
            "hnsw_ef_construction": 64,
            "hnsw_ef_search": 40,
            "hnsw_dist_method": "vector_cosine_ops",
        },
    )


def main() -> None:
    settings = get_settings()
    corpus = settings.corpus_path

    if not corpus.exists():
        sys.exit(f"Corpus not found at {corpus}. Place the NCC PDF there first.")

    print(f"Ingest starting. Corpus: {corpus}")
    current_hash = md5_file(corpus)

    conn_kwargs = dict(
        host=settings.postgres_host,
        port=settings.postgres_port,
        user=settings.postgres_user,
        password=settings.postgres_password,
        dbname=settings.postgres_db,
    )

    with psycopg.connect(**conn_kwargs) as conn:
        with conn.cursor() as cur:
            cur.execute(CORPUS_METADATA_DDL)
        conn.commit()

        stored = existing_corpus_hash(conn, settings.corpus_id)
        if stored == current_hash:
            print(f"Corpus unchanged (md5={current_hash[:12]}...). Skipping ingestion.")
            return

        print("Corpus changed or first ingest. Running full pipeline...")

        t0 = time.perf_counter()
        documents = load_documents(corpus)
        print(f"  PDF loaded: {len(documents)} pages in {time.perf_counter() - t0:.2f}s")

        t1 = time.perf_counter()
        splitter = SentenceSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )
        nodes = splitter.get_nodes_from_documents(documents)
        print(f"  Chunked: {len(nodes)} chunks in {time.perf_counter() - t1:.2f}s")

        for i, node in enumerate(nodes):
            page = node.metadata.get("page_number", 0)
            node.id_ = deterministic_chunk_id(corpus.name, page, i, node.text)

        ids = [n.id_ for n in nodes]
        existing = existing_chunk_ids(conn, ids, settings.vector_table_full)
        new_nodes = [n for n in nodes if n.id_ not in existing]
        print(f"  In store: {len(existing)}; new to embed: {len(new_nodes)}")

        if new_nodes:
            t2 = time.perf_counter()
            pg_store = build_pg_store(settings)
            embed_model = OpenAIEmbedding(
                model=settings.embedding_model, api_key=settings.openai_api_key
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store=pg_store, embed_model=embed_model
            )
            index.insert_nodes(new_nodes)
            print(
                f"  Embedded + inserted {len(new_nodes)} chunks "
                f"in {time.perf_counter() - t2:.2f}s"
            )
        else:
            print("  No new chunks to embed; updating metadata only.")

        upsert_corpus_metadata(conn, settings.corpus_id, current_hash, len(nodes))
        print(f"Ingest complete. Total chunks: {len(nodes)}; hash: {current_hash[:12]}...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Lint**

Run:
```bash
uv run ruff check src/knowledge_bot/ingest.py
```

Expected: `All checks passed!`.

- [ ] **Step 3: Syntax check**

Run:
```bash
uv run python -c "import ast; ast.parse(open('src/knowledge_bot/ingest.py').read()); print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add src/knowledge_bot/ingest.py
git commit -m "Phase 2: add ingestion CLI

Standalone CLI runnable as 'python -m knowledge_bot.ingest'. Two-tier
idempotency: md5(pdf) short-circuit via corpus_metadata; chunk-level
SHA-256 + SELECT pre-check via psycopg, embed and insert only new
chunks via VectorStoreIndex.insert_nodes()."
```

---

## Task 12: Update Dockerfile to run ingest before uvicorn

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Replace the `CMD` line with an entrypoint script**

Edit `Dockerfile` to change the last line. Replace:

```dockerfile
CMD ["uvicorn", "knowledge_bot.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

With:

```dockerfile
CMD ["sh", "-c", "python -m knowledge_bot.ingest && exec uvicorn knowledge_bot.main:app --host 0.0.0.0 --port 8000"]
```

The `&&` ensures uvicorn only starts if ingest exits 0. `exec` replaces the shell with uvicorn so signals propagate.

- [ ] **Step 2: Verify the Dockerfile parses**

Run:
```bash
docker build --no-cache=false -t knowledge-bot-py:phase2-test . 2>&1 | tail -20
```

Expected: build completes; if it fails on `python -m knowledge_bot.ingest`, that's expected at build time (it's invoked at runtime, not build time). What we're checking here is that the Dockerfile syntax is valid.

If `docker build` fails for unrelated reasons (e.g., file not found), STOP and report.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "Phase 2: run ingest before uvicorn in container

CMD now runs python -m knowledge_bot.ingest first, then exec uvicorn.
The ingest CLI exits 0 either via 'corpus unchanged' or successful
ingestion. exec ensures signals reach uvicorn for graceful shutdown."
```

---

## Task 13: Update docker-compose.yml

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add the corpus volume and Postgres env vars to the app service**

The current `app` service block needs Postgres connection env vars (so the in-container app talks to the `postgres` service via its docker-network hostname, not localhost). Replace the `app` service in `docker-compose.yml` with:

```yaml
  app:
    build: .
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      POSTGRES_HOST: postgres
      POSTGRES_PORT: 5432
      POSTGRES_USER: knowledgebot
      POSTGRES_PASSWORD: knowledgebot
      POSTGRES_DB: knowledgebot
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY}
      OPENAI_API_KEY: ${OPENAI_API_KEY}
    ports:
      - "8000:8000"
    volumes:
      - ./corpus:/app/corpus:ro
```

The two key changes from the current file:
1. `POSTGRES_HOST=postgres` (the service name, not localhost) and `POSTGRES_PORT=5432` (in-container port, not the 5433 host mapping).
2. Removed the now-unused `DATABASE_URL` line (we build the connection in `config.py`).

- [ ] **Step 2: Verify YAML parses**

Run:
```bash
docker compose config 1>/dev/null
```

Expected: silent success. If you see a YAML error, STOP and report.

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "Phase 2: pass POSTGRES_* env vars to app container

The in-container app needs to reach postgres via the docker-network
hostname 'postgres' on port 5432, not localhost:5433. Drops the
unused DATABASE_URL var (config.py builds the connection from parts)."
```

---

## Task 14: Run docker-compose up and verify ingestion + smoke test

**Files:** none (verification).

- [ ] **Step 1: Verify the local corpus PDF is still in place**

Run:
```bash
ls -lh corpus/ncc-2022-vol2.pdf
```

Expected: ~6.5MB file. If missing, run `cp /Users/harry/Development/build-knowledge-bot/corpus/ncc-2022-vol2.pdf corpus/`.

- [ ] **Step 2: Bring the stack up (foreground, watch logs)**

Run:
```bash
docker compose up --build
```

In the logs, look for:
- `postgres-1 | ... database system is ready to accept connections`
- `app-1     | Ingest starting. Corpus: corpus/ncc-2022-vol2.pdf`
- `app-1     |   PDF loaded: 312 pages in ...`
- `app-1     |   Chunked: ~485 chunks in ...`
- `app-1     |   Embedded + inserted ~485 chunks in ...`
- `app-1     | Ingest complete. Total chunks: ~485`
- `app-1     | Uvicorn running on http://0.0.0.0:8000`

The chunk count should be near Java's 485 (within ±10%).

If ingestion fails, common causes:
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` not set in `.env` → check `.env` exists and has both keys.
- pgvector extension not enabled → the `pgvector/pgvector:pg16` image enables it automatically; if the error mentions `vector` extension missing, ensure the image is `pgvector/pgvector:pg16` not plain `postgres:16`.
- Postgres connection refused → check the `POSTGRES_HOST=postgres` env var is set in the `app` service.

- [ ] **Step 3: Smoke-test /health from another terminal**

Run:
```bash
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`.

- [ ] **Step 4: Smoke-test /ask with one substantive question**

Run:
```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the minimum ceiling height for a habitable room?"}' \
  | tee /tmp/phase2-smoke.json
```

Expected: a JSON response shaped `{"answer":"...","metrics":{"input_tokens":...,"output_tokens":...,"latency_ms":...,"cost_usd":...}}`. Token counts should be roughly 2k–5k input, 100–400 output. Cost should be in the $0.002–$0.006 range.

If the response is 503 with "Vector store returned no chunks", ingestion didn't populate the store — go back to Step 2 logs.

If 422, the request body is malformed.

- [ ] **Step 5: Verify pgvector contents**

In a third terminal:
```bash
docker compose exec postgres psql -U knowledgebot -d knowledgebot -c "SELECT count(*) FROM data_knowledge_bot;"
docker compose exec postgres psql -U knowledgebot -d knowledgebot -c "SELECT * FROM corpus_metadata;"
```

Expected: count is the same number of chunks as logged in Step 2 (~485). The `corpus_metadata` row shows `corpus_id=ncc-2022-vol2`, the md5 hash, the chunk count, and a recent `ingested_at`.

- [ ] **Step 6: Restart and verify idempotency**

Stop the stack (Ctrl+C in terminal from Step 2), then:
```bash
docker compose up
```

In the logs, look for:
- `app-1     | Corpus unchanged (md5=...). Skipping ingestion.`

This confirms the doc-level hash short-circuit works. No second round of OpenAI embedding charges.

- [ ] **Step 7: This task makes NO commit (verification only)**

Working tree should still be clean from Task 13's commit.

---

## Task 15: Run six sample queries (mirroring Java's findings)

**Files:** none (captures stdout for Task 16).

- [ ] **Step 1: With the stack still running, send the six probe queries**

Run each one and tee the output. They mirror Java's Phase 2 sample set.

```bash
mkdir -p /tmp/phase2-samples

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the minimum ceiling height for a habitable room?"}' \
  | tee /tmp/phase2-samples/01-easy-ceiling.json

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the fall rating for an external balcony?"}' \
  | tee /tmp/phase2-samples/02-unknown-term.json

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What are the Bushfire Attack Levels (BAL) defined in NCC and what does BAL-40 mean?"}' \
  | tee /tmp/phase2-samples/03-enumeration-bal.json

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What are the smoke alarm requirements for a Class 1 building in the NCC?"}' \
  | tee /tmp/phase2-samples/04-specific-smoke.json

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"Explain clause H99Z1."}' \
  | tee /tmp/phase2-samples/05-fake-clause-plain.json

curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the minimum ceiling height in the UK building regulations?"}' \
  | tee /tmp/phase2-samples/06-out-of-corpus-uk.json
```

Each command should print a JSON response. Save these for Task 16.

- [ ] **Step 2: Quick sanity check on the metrics**

Run:
```bash
for f in /tmp/phase2-samples/*.json; do
  echo "=== $(basename "$f") ==="
  jq '.metrics' "$f" 2>/dev/null || cat "$f"
  echo
done
```

Expected: each shows non-zero `input_tokens`, `output_tokens`, `latency_ms`, and `cost_usd`. If any are zero, the `TokenCountingHandler` isn't wired correctly — STOP and investigate.

- [ ] **Step 3: This task makes NO commit (verification only)**

---

## Task 16: Write `docs/observations/phase-2-findings.md`

**Files:**
- Create: `docs/observations/phase-2-findings.md`

- [ ] **Step 1: Draft the findings doc**

Use the captured numbers from `/tmp/phase2-samples/*.json` to fill in the table. Create the file with this skeleton; substitute every `<...>` placeholder with the corresponding actual value:

```markdown
# Phase 2 Findings: Basic RAG (Python)

## What was built

Two pipelines sharing a single Postgres+pgvector store:

- **Ingestion** (one-shot CLI: `uv run python -m knowledge_bot.ingest`): NCC PDF → 312 pages via `pymupdf` → `SentenceSplitter` (800 tok, no overlap) → <CHUNK_COUNT> chunks → SHA-256 deterministic IDs → SQL pre-check via `psycopg` → `VectorStoreIndex.insert_nodes()` for new chunks only. Two-tier idempotency: `md5(pdf_bytes)` short-circuits via `corpus_metadata`.
- **Query** (`POST /ask`): question → LlamaIndex `RetrieverQueryEngine(top_k=4)` → cosine search in `data_knowledge_bot` → 4 chunks → Anthropic `claude-haiku-4-5` → `AnswerResponse(answer, metrics)`.

Containerised via docker-compose: `pgvector/pgvector:pg16` + multi-stage Python image. The container's entrypoint runs ingest then uvicorn.

## Pipeline numbers

| Metric | Value |
|---|---|
| Corpus | NCC 2022 Volume 2, 312 pages |
| Chunks stored | <CHUNK_COUNT> |
| Embedding model | `text-embedding-3-small` (1536 dim) |
| Chat model | `claude-haiku-4-5` |
| Vector index | HNSW, cosine distance |
| First-boot ingestion cost | ~$<FIRST_BOOT_COST> (one OpenAI embedding pass per chunk) |
| Re-boot ingestion cost | $0.00 (md5-hash short-circuit fires) |
| Re-boot ingestion latency | ~<REBOOT_SECONDS>s (PDF md5 + corpus_metadata SELECT) |

The `md5(pdf_bytes)` short-circuit is the load-bearing decision. Without it, every `docker compose up` would re-extract the PDF and re-query pgvector. With it, restarts are basically instant until the corpus content changes.

## Per-query cost & latency

Across six probe queries mirroring the Java repo's Phase 2 sample set:

| Question type | Input tok | Output tok | Latency | Cost |
|---|---|---|---|---|
| Easy / specific (ceiling height) | <Q1_IN> | <Q1_OUT> | <Q1_LAT>s | $<Q1_COST> |
| Unknown term (fall rating) | <Q2_IN> | <Q2_OUT> | <Q2_LAT>s | $<Q2_COST> |
| Enumeration (BAL ladder) | <Q3_IN> | <Q3_OUT> | <Q3_LAT>s | $<Q3_COST> |
| Specific topic (smoke alarms) | <Q4_IN> | <Q4_OUT> | <Q4_LAT>s | $<Q4_COST> |
| Fake clause (plain) | <Q5_IN> | <Q5_OUT> | <Q5_LAT>s | $<Q5_COST> |
| Out-of-corpus (UK regulations) | <Q6_IN> | <Q6_OUT> | <Q6_LAT>s | $<Q6_COST> |

Cost is clustered tightly around $<MEDIAN_COST>. Input tokens dominate (4×–20× output), driven by the four retrieved chunks plus the LlamaIndex `text_qa_template` overhead. Latency is dominated by Claude's generation time, not retrieval (pgvector queries return in tens of milliseconds at <CHUNK_COUNT> rows).

## Observed failure modes

<Inspect each query response. Note which exhibited which failure mode. The expected pattern (matching Java's findings) is:>

### 1. <Failure mode 1, e.g. retrieval miss / scope leakage / incidental refusal>

**Question:** *<verbatim question>*

<2–4 sentences describing what the system returned and why it's a failure (or success). Cross-reference Java's findings if behaviour matches.>

### 2. <Failure mode 2 or positive observation>

**Question:** *<verbatim question>*

<As above.>

<Continue for each query worth discussing.>

## Cross-stack note

Mirrors the [Java repo's Phase 2 findings](https://github.com/Harryqin99/build-knowledge-bot/blob/main/docs/observations/phase-2-findings.md):

- **Same models, same parameters, same chunk size, same top-K** — the cross-stack comparison holds.
- **Different orchestration.** Spring AI's `QuestionAnswerAdvisor` wired retrieval + augmentation as a one-line advisor on the `ChatClient`; LlamaIndex's `RetrieverQueryEngine` exposes the retriever, prompt template, and LLM as named components you can swap individually. The Python side reads more explicitly; the Java side reads more declaratively.
- **Different ingestion timing.** Java ingested at `ApplicationReadyEvent` (boot-time event); Python ingests via standalone CLI invoked by the container entrypoint before uvicorn starts. Functionally equivalent for the docker-compose UX; pedagogically different (Python's pipeline is a discrete step you can run, time, and inspect on its own).
- **Same idempotency algorithm.** Both sides use SHA-256 content-hash IDs + a SELECT pre-check; Java's done via `JdbcTemplate`, Python's via `psycopg`. Python adds a second-tier `md5(pdf_bytes)` short-circuit for an extra layer of "skip everything" when the document hasn't changed — useful given that ingest runs on every container start.

## Implications for Phase 3+

<Pull forward concrete failure modes observed in the Python run, mapped to future phases:>

- **Refusal is incidental, not structural** (if observed) → Phase 3 honesty layer.
- **Scope leakage on UK questions** (if observed) → Phase 3 scope check.
- **Boilerplate-chunk pollution / clause-ID split / enumeration miss** (if observed) → Phase 5 structure-aware chunking.

## Net result

Phase 2's `/ask` endpoint works end-to-end for substantive NCC questions. Cost per query is predictable (~$<MEDIAN_COST>) and matches Java's order of magnitude. Failure modes are *named* and *each mapped to a future phase*.

The system as built is good enough to demonstrate cross-stack RAG mechanics, but not good enough to deploy: refusal is incidental (Phase 3), retrieval has known precision gaps (Phase 5/6/7), corpus scope is incomplete (out of scope here). Each gap is documented above with the specific test case that surfaced it.
```

- [ ] **Step 2: Substitute every `<...>` placeholder**

Open the file. Replace each placeholder with the value from `/tmp/phase2-samples/`. Compute `<MEDIAN_COST>` as the median of the six per-query costs. Look at each response's `answer` field to fill in the failure-mode discussion (1–3 sub-sections is fine; not all six queries need a section).

- [ ] **Step 3: Verify no placeholders remain**

Run:
```bash
grep -nE "<[A-Za-z_0-9]+>" docs/observations/phase-2-findings.md
```

Expected: empty output. If matches, fix them.

- [ ] **Step 4: Commit**

```bash
git add docs/observations/phase-2-findings.md
git commit -m "Phase 2: add findings doc

Records pipeline numbers, six probe-query results with cost/latency,
observed failure modes, and a cross-stack note vs. the Java repo's
Phase 2 findings."
```

---

## Task 17: Update `README.md` (8-phase plan + Phase 2 done)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the phases table**

Open `README.md`. Find the phases table around lines 43–50. Replace its body with the new 8-phase version. The full `| Phase | Status | What |` table should now read:

```
| Phase | Status | What |
|---|---|---|
| 0 — Scaffolding | done | uv, FastAPI, Docker, compose, /health |
| 1 — Long-context measurement | [done](docs/observations/phase-1-findings.md) | NCC corpus measures 190K tokens via Anthropic `count_tokens` — 95% of Claude's 200K window, zero headroom; RAG required |
| 2 — Basic RAG | [done](docs/observations/phase-2-findings.md) | LlamaIndex `RetrieverQueryEngine` over pgvector; ingestion CLI; `/ask` returns answer + token/cost metrics |
| 3 — Honesty layer | — | Pydantic-typed structured refusal envelope (`{can_answer, reason, answer, citations}`); scope check |
| 4 — Eval harness | — | golden Q&A set, retrieval@K, faithfulness scoring |
| 5 — Structure-aware chunking | — | regex-split on clause IDs; strip repeating page boilerplate |
| 6 — Reranking | — | retrieve top-N via embeddings → cross-encoder rerank → top-K |
| 7 — Multi-query retrieval | — | LLM expands user question into N variants; merge candidate chunks |
```

- [ ] **Step 2: Verify rendering**

Run:
```bash
sed -n '/^| Phase /,/^$/p' README.md
```

Expected: clean Markdown table with eight rows, no broken pipes.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Phase 2: mark phase complete; expand to 8-phase plan

Adds two new phases (6 reranking, 7 multi-query) discovered during
Phase 2 brainstorming. Phase 2 row links to phase-2-findings.md."
```

---

## Task 18: Final verification

**Files:** none (verification).

- [ ] **Step 1: Working tree is clean**

Run:
```bash
git status
```

Expected: `nothing to commit, working tree clean`.

- [ ] **Step 2: All tests pass**

Run:
```bash
uv run pytest -v
```

Expected: `4 passed` (1 health + 3 ask).

- [ ] **Step 3: Container still serving**

If you stopped docker compose between Tasks 15 and 17, restart:
```bash
docker compose up -d
sleep 5
curl -s http://localhost:8000/health
```

Expected: `{"status":"ok"}`. If you didn't stop it, just curl.

- [ ] **Step 4: One last /ask call**

Run:
```bash
curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question":"What is the minimum ceiling height for a habitable room?"}' \
  | jq '.metrics'
```

Expected: a metrics object with non-zero token counts, sub-10s latency, cost in $0.002–$0.006 range.

- [ ] **Step 5: Bring stack down**

Run:
```bash
docker compose down
```

Expected: containers stopped. Vector data persists in the named volume `pg-data` for next run.

- [ ] **Step 6: Commit log shows the Phase 2 commits**

Run:
```bash
git log --oneline -20
```

Expected: at least 14 Phase 2 commits since `0e0ed31` (the design commit).

---

## Done criteria

Phase 2 is complete when all of the following hold:

1. `docker compose up --build` runs to completion: postgres healthy, ingest produces ~485 chunks (or skips if unchanged), uvicorn listens on 8000.
2. `curl -X POST localhost:8000/ask -d '{"question":"..."}'` returns a substantive JSON answer with non-zero metrics.
3. `uv run pytest -v` shows 4 tests passing.
4. `docs/observations/phase-2-findings.md` exists, has all `<...>` placeholders filled, links to the Java repo's findings.
5. README's Phase 2 row reads `[done](docs/observations/phase-2-findings.md)`; phase table shows 8 phases.
6. `git status` is clean.
7. Restarting the stack (`docker compose down && docker compose up`) shows "Corpus unchanged, skipping ingestion" — md5 short-circuit works, no double-billing.
