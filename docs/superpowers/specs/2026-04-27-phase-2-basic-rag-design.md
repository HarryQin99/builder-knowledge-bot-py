# Phase 2: Basic RAG (Python) — Design

**Status:** Approved 2026-04-27 (pending written-spec review)
**Phase:** 2 of 8 (project expanded from 6 to 8 phases — see "Phase plan update" below)
**Predecessor:** Phase 1 (`docs/observations/phase-1-findings.md`) — corpus measured at 190K tokens, RAG required for headroom
**Sibling:** [Java repo Phase 2 plan + findings](https://github.com/Harryqin99/build-knowledge-bot)

## Goal

Make the full 312-page NCC 2022 Vol 2 corpus queryable via `POST /ask`. Same models, same chunk size, same top-K, and same response shape as the Java repo's Phase 2 — so the cross-stack comparison stays apples-to-apples. Diverges only where Python idioms genuinely differ (CLI ingestion vs. Spring lifecycle event, document-level hash short-circuit on top of chunk-level dedup).

## Why this phase exists

Phase 1 confirmed the corpus is 190K Anthropic tokens — 95% of Claude's window with no headroom. Long-context is structurally infeasible. RAG is the only path forward: chunk the corpus, embed each chunk, retrieve only the relevant passages per question, and stuff *those* into the prompt instead of the whole document.

The point of Phase 2 is not "RAG works at all" — that's a known result. The point is the **cross-stack comparison**: Spring AI's `QuestionAnswerAdvisor` vs. LlamaIndex's `RetrieverQueryEngine`. Same problem, same models, same parameters, different orchestration.

## Architecture: two pipelines, one vector store

```
INGESTION (one-shot CLI: `uv run python -m knowledge_bot.ingest`)
  PDF → pymupdf extract → SentenceSplitter (800 tok, no overlap)
       → SHA-256 chunk IDs → SQL pre-check (skip existing)
       → OpenAI text-embedding-3-small → pgvector INSERT
       → UPSERT corpus_metadata with new content hash

QUERY (per request: `POST /ask`)
  question → LlamaIndex RetrieverQueryEngine
         (retriever: VectorStoreIndex.as_retriever(top_k=4)
          → cosine search in pgvector
          → 4 chunks
          → text_qa_template injects chunks into Claude's prompt
          → claude-haiku-4-5 generates answer)
  → AskService captures token usage via TokenCountingHandler,
    measures latency, computes USD cost, returns AnswerResponse
```

The two pipelines share `data_knowledge_bot` (the LlamaIndex pgvector table). Ingestion writes; query reads. Both pipelines are explicit and named — no `from_documents(...).as_query_engine().query(...)` magic. Each stage (chunking, embedding, retrieval, prompt-stitching, generation, metrics) is a callable function or named LlamaIndex object.

## Design decisions

### Abstraction depth: mid-level explicit pipeline

LlamaIndex offers three depths: high-level magic (`VectorStoreIndex.from_documents().as_query_engine().query()`), mid-level (named components wired together: `IngestionPipeline` + `RetrieverQueryEngine` + explicit `Anthropic` LLM), and low-level (manual chunking + manual SQL + manual SDK calls).

**Mid-level chosen.** Reasoning: high-level hides everything (defeats the learning goal); low-level writes off LlamaIndex (defeats the cross-stack story — README pitches LlamaIndex's `QueryEngine` as the comparison point with Spring AI's `QuestionAnswerAdvisor`). Mid-level keeps the framework comparison while making each stage of the pipeline a named, inspectable, mockable object.

### Ingestion entry point: standalone CLI

Java ran ingestion at boot via `@EventListener(ApplicationReadyEvent)`. Python diverges deliberately: a CLI script (`uv run python -m knowledge_bot.ingest`) is the canonical ingestion path. The FastAPI app assumes the corpus is indexed; if the vector store is empty, `/ask` returns 503 with a message pointing at the CLI.

Reasoning:
- The learning goal benefits from seeing ingestion run as a separate step. `docker-compose up` shouldn't hide it inside FastAPI's lifespan.
- The CLI is independently testable (run with print-debugging, time it, swap a fixture PDF) without spinning up FastAPI.
- The cross-stack note is itself interesting: "Spring AI runs ingestion as a Spring lifecycle event; Python runs it as an explicit CLI" — different framework idioms, same outcome.

The docker-compose flow runs the CLI as a separate `command:` step (or a `depends_on` ordering with a healthcheck) so a fresh container still ends up with a populated store; the *user* doesn't run ingestion manually unless developing locally.

### Idempotency: two-tier check, doc hash short-circuits chunk hash

```
1. Compute md5(pdf_bytes).
2. SELECT content_hash FROM corpus_metadata WHERE corpus_id = 'ncc-2022-vol2'.
3. If match → log "corpus unchanged, skipping ingestion" → exit (no PDF parse, no chunking, no DB chunk lookup, no embedding).
4. If miss → run the full pipeline:
     a. parse PDF → chunks → SHA-256 IDs (one ID per chunk, derived from source + page + chunk_index + text).
     b. SELECT id FROM data_knowledge_bot WHERE id = ANY(:ids) — find which chunks already exist.
     c. embed and INSERT only the genuinely new chunks via VectorStoreIndex.insert_nodes().
     d. UPSERT corpus_metadata with new hash + chunk count + ingested_at timestamp.
```

**Doc-level hash hashes PDF bytes only.** Not the chunking config. Reasoning: simpler. If you ever change chunking parameters, the workaround is a one-time `DELETE FROM corpus_metadata; DELETE FROM data_knowledge_bot;` to force re-ingestion. The cost of forgetting is paid by the engineer, not the user, and re-ingestion is ~$0.012.

**Chunk-level hash mirrors Java exactly.** Same algorithm: SHA-256 of `corpus_filename + "|" + page_number + "|" + chunk_index + "|" + chunk_text`, inserted as the chunk's `node_id` (LlamaIndex) which becomes the `id` column in pgvector. Cross-stack data point: "Java: SHA-256 + JdbcTemplate; Python: SHA-256 + psycopg" — same algorithm, different drivers, same idempotency property.

### Chunking parameters: match Java exactly

`SentenceSplitter(chunk_size=800, chunk_overlap=0)` — sentence-boundary aware, ~800 tokens per chunk, no overlap. Mirrors Spring AI's default `TokenTextSplitter` behaviour (Spring AI 2.0.0-M4 didn't support overlap anyway).

Java's Phase 2 found three chunking-related weaknesses (boilerplate-chunk pollution, header/content split breakage, BAL enumeration retrieval miss) and explicitly deferred them to Phase 5. Python's Phase 2 will reproduce the same weaknesses on purpose — they're the baseline against which Phase 5's structure-aware chunking gets measured. Fixing them now means having nothing to compare against in Phase 5.

### Retrieval: top-K = 4

Mirrors Java exactly. `as_retriever(similarity_top_k=4)`. Cosine similarity in pgvector. HNSW index. Same number of chunks land in the prompt; same input-token cost (~4× chunk_size + template overhead).

### Generation: claude-haiku-4-5 via LlamaIndex's `Anthropic` LLM class

Same model as Java. LlamaIndex's `llama-index-llms-anthropic` package wraps Anthropic's SDK. Token usage flows through LlamaIndex's `TokenCountingHandler` callback for metrics capture.

### Embedding: OpenAI text-embedding-3-small

Same model as Java. 1536 dimensions. `llama-index-embeddings-openai`. The pgvector column is `vector(1536)` — must match.

### Response shape: mirror Java's exactly

```python
class AnswerMetrics(BaseModel):
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float

class AnswerResponse(BaseModel):
    answer: str
    metrics: AnswerMetrics
```

Same fields, same casing (snake_case in Python; Java was camelCase but content is identical). Pydantic v2 validates both directions — request validation on `AskRequest(question: str)`, serialization on `AnswerResponse`.

### Cost calculator: explicit module, mirrors Java's enum

```python
# pricing.py
@dataclass(frozen=True)
class ModelPricing:
    input_per_1k: float
    output_per_1k: float

MODEL_PRICING: dict[str, ModelPricing] = {
    "claude-haiku-4-5": ModelPricing(input_per_1k=0.001, output_per_1k=0.005),  # placeholder values; fill from Anthropic price page
    "text-embedding-3-small": ModelPricing(input_per_1k=0.00002, output_per_1k=0.0),
}

def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        raise ValueError(f"No pricing for model {model!r}; add it to MODEL_PRICING.")
    return (input_tokens * pricing.input_per_1k + output_tokens * pricing.output_per_1k) / 1000
```

Unknown model raises loudly — no silent miscalculation. Mirrors Java's enum-based `forModel(...)` lookup that throws on unknown ids.

### Schema management: LlamaIndex auto-creates vector table; manual CREATE TABLE for corpus_metadata

LlamaIndex's `PGVectorStore(table_name="knowledge_bot", embed_dim=1536, hnsw_kwargs={...})` creates its own table on first use (with HNSW index, cosine distance). LlamaIndex prepends `data_` to the table name, so the actual SQL identifier is `data_knowledge_bot`. No Alembic, no migrations.

The `corpus_metadata` table is one-table-one-purpose, gets a single `CREATE TABLE IF NOT EXISTS` at the top of `ingest.py`. Two tables total, no migration tooling — deliberately under-engineered for the project's size.

```sql
CREATE TABLE IF NOT EXISTS corpus_metadata (
  corpus_id TEXT PRIMARY KEY,
  content_hash TEXT NOT NULL,
  chunk_count INT NOT NULL,
  ingested_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

### Stack lock

| Concern | Choice |
|---|---|
| RAG framework | LlamaIndex (`llama-index-core`) |
| Vector store binding | `llama-index-vector-stores-postgres` (PGVectorStore) |
| Embedding binding | `llama-index-embeddings-openai` (OpenAIEmbedding) |
| Chat LLM binding | `llama-index-llms-anthropic` (Anthropic) |
| Postgres driver | `psycopg[binary]` v3 (modern; for the manual SQL pre-check + corpus_metadata) |
| ORM (for raw SQL only) | `sqlalchemy` (LlamaIndex's PGVectorStore takes a SQLAlchemy URL anyway) |
| Splitter | `SentenceSplitter(chunk_size=800, chunk_overlap=0)` |
| Embedding model | `text-embedding-3-small` (1536 dim) |
| Chat model | `claude-haiku-4-5` |
| Vector index | HNSW, cosine distance |
| Top-K | 4 |
| Vector table | `data_knowledge_bot` |
| Postgres image | `pgvector/pgvector:pg16` (matches Java repo, host port 5433) |

## Components

### File layout

```
src/knowledge_bot/
├── main.py              EXISTING, FastAPI app — gains lifespan + /ask router include
├── config.py            NEW: Pydantic-settings — DB URL, model IDs, chunk size, top-K, corpus path
├── ingest.py            NEW: ingestion CLI entrypoint + pipeline functions
├── query.py             NEW: builds the LlamaIndex RetrieverQueryEngine (singleton, lifespan-cached)
├── ask_service.py       NEW: takes a question, calls query_engine, captures metrics, returns AnswerResponse
├── routes/
│   ├── __init__.py      NEW
│   └── ask.py           NEW: POST /ask FastAPI router; depends-on AskService
├── models.py            NEW: Pydantic — AskRequest, AnswerResponse, AnswerMetrics
└── pricing.py           NEW: ModelPricing dataclass + cost_for() function

scripts/
└── measure_tokens.py    EXISTING (Phase 1) — kept; doesn't conflict with Phase 2

tests/
├── test_health.py       EXISTING
└── test_ask.py          NEW: pytest + FastAPI TestClient against /ask, query_engine mocked

docker-compose.yml       MODIFIED: postgres service activated; ingestion runs as separate step
Dockerfile               MODIFIED: copy scripts/ + entry shell that calls ingest then uvicorn
.env.example             MODIFIED: add OPENAI_API_KEY, POSTGRES_* placeholders
pyproject.toml           MODIFIED: add LlamaIndex packages + psycopg + sqlalchemy
```

### Per-file responsibilities

- **`config.py`** — Single source of truth for all runtime config, env-driven via Pydantic settings (`BaseSettings`). One `Settings` class, one `get_settings()` function (cached). Avoids constants scattered across modules. Same role as `application.properties` in Java.
- **`ingest.py`** — Runnable as `uv run python -m knowledge_bot.ingest`. Holds the chunk-id hashing, the doc-level + chunk-level dedup checks, and the pipeline orchestration. ~120 lines. Exports a `main()` for the CLI and an `ingest()` function for tests/reuse.
- **`query.py`** — Wraps `VectorStoreIndex.from_vector_store(...).as_retriever(...)` + `RetrieverQueryEngine.from_args(...)` into a `build_query_engine() -> RetrieverQueryEngine` factory. Called once per process at FastAPI startup, cached on `app.state.query_engine` via the `lifespan` context manager.
- **`ask_service.py`** — Pure service layer. Doesn't know about FastAPI. Takes a `RetrieverQueryEngine` + a question, attaches a `TokenCountingHandler`, calls `query_engine.query(question)`, reads token counts off the handler, measures wall-clock latency, computes cost via `pricing.cost_for(...)`, returns `AnswerResponse`. Mockable in tests by replacing `RetrieverQueryEngine` with a stub.
- **`routes/ask.py`** — Thin FastAPI router. `@router.post("/ask")`, validates `AskRequest`, delegates to `AskService`, returns `AnswerResponse`. The HTTP layer.
- **`models.py`** — Pydantic v2 models: `AskRequest(question: str)`, `AnswerMetrics(...)`, `AnswerResponse(answer, metrics)`. Same shape as Java's records (snake_case in JSON because Python convention).
- **`pricing.py`** — `MODEL_PRICING: dict[str, ModelPricing]` and a `cost_for(model, input_tok, output_tok)` function. Mirrors Java's `ModelPricing` enum + `CostCalculator`. Unknown model → raises (loud failure, no silent miscalculation).

## Data flow

### Ingestion (CLI)

1. `uv run python -m knowledge_bot.ingest`
2. Load settings (`config.get_settings()` — DB URL, corpus path, model IDs).
3. Open the corpus PDF; compute `md5(pdf_bytes)`.
4. Connect to postgres via psycopg.
5. `CREATE TABLE IF NOT EXISTS corpus_metadata (...)`.
6. `SELECT content_hash FROM corpus_metadata WHERE corpus_id = 'ncc-2022-vol2'`.
   - If matches current md5 → print "Corpus unchanged, nothing to ingest." → exit 0.
   - If empty or mismatched → continue to step 7.
7. Extract text via pymupdf, page by page → list of `Document(text=..., metadata={"page_number": N})`.
8. Chunk via `SentenceSplitter(chunk_size=800, chunk_overlap=0).get_nodes_from_documents(documents)` → list of `TextNode` objects.
9. Assign deterministic `node_id` to each chunk: `sha256(corpus_filename + "|" + page_number + "|" + chunk_index + "|" + text).hexdigest()`. Set on `node.id_`.
10. `SELECT id FROM data_knowledge_bot WHERE id = ANY(:ids)` → set of existing IDs.
11. Filter out already-existing chunks. Log: "N total, M already in store, K to embed."
12. If `K == 0`: skip step 13.
13. `index = VectorStoreIndex.from_vector_store(pg_store, embed_model=openai_embed)`. `index.insert_nodes(new_chunks)` → embeds + inserts in one call.
14. `UPSERT INTO corpus_metadata (corpus_id, content_hash, chunk_count, ingested_at) VALUES (...)`.
15. Print: "Ingested: K new, M skipped, total {K + M} chunks. Cost: ${...}."

### Query (HTTP)

1. FastAPI receives `POST /ask` with body `{"question": "What's the BAL ladder?"}`.
2. `AskRequest` validates the body (Pydantic).
3. `AskService.answer(question)`:
   - `t0 = time.perf_counter()`.
   - Reset `TokenCountingHandler`.
   - `response = query_engine.query(question)` — under the hood: embed the question via OpenAI, search pgvector for top-4 by cosine, format chunks into a prompt template, call Claude.
   - `latency_ms = int((time.perf_counter() - t0) * 1000)`.
   - Read `token_counter.total_llm_token_count_input` and `total_llm_token_count_output`.
   - `cost = cost_for("claude-haiku-4-5", input_tok, output_tok)`.
   - Return `AnswerResponse(answer=str(response), metrics=AnswerMetrics(...))`.
4. FastAPI serializes to JSON, returns 200.

If pgvector is empty (no ingestion ever ran), the retrieval call returns no chunks. The service detects this (`len(response.source_nodes) == 0`) and returns 503 with a message pointing at the CLI: "Vector store is empty. Run `uv run python -m knowledge_bot.ingest` first."

## Testing

**Light strategy (TestClient + smoke).**

### `tests/test_ask.py`

- Spins up FastAPI's `TestClient`.
- Replaces the `RetrieverQueryEngine` dependency with a stub (`FakeQueryEngine.query(question) -> stub Response`).
- Asserts:
  - `POST /ask` with valid body → 200 + `AnswerResponse` shape
  - `POST /ask` with missing question → 422
  - `POST /ask` when query engine raises `EmptyVectorStoreError` → 503 with helpful message
- ~3 tests, ~50 lines. No live API calls, no live database, no live LlamaIndex calls.

### Smoke test

- `docker-compose up --build` boots postgres + app.
- App entrypoint runs `python -m knowledge_bot.ingest` first, then `uvicorn`. Ingest exits 0 either via "corpus unchanged" or via successful ingestion.
- `curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"What is the minimum ceiling height?"}'` → 200 with substantive answer + metrics.
- Six sample queries (mirroring Java's findings doc) drive the Phase 2 findings writeup.

### What's not tested

- The ingestion CLI itself. It's a one-shot pipeline; running it once and seeing 485-ish chunks in pgvector is the verification. Unit-testing the splitter or the SHA-256 hashing teaches you about the libraries, not your code.
- Cost calculator math. It's three lines; if it's wrong it's wrong in the response, which the smoke test inspects.
- LlamaIndex internals. Not our code.

## Verification

1. `docker-compose up --build` runs to completion: postgres healthy, app listening on 8000.
2. First boot: ingestion produces ~485 chunks (within ±5% of Java's 485). Cost printed (~$0.012).
3. Second boot: ingestion logs "Corpus unchanged, nothing to ingest." Cost = $0.
4. `curl localhost:8000/health` → 200 `{"status":"ok"}`.
5. `curl -X POST localhost:8000/ask -d '{"question":"..."}'` → 200 with `{"answer": "...", "metrics": {"input_tokens": ..., "output_tokens": ..., "latency_ms": ..., "cost_usd": ...}}`.
6. Six sample queries (easy / specific / clause-reference / cross-section / fake-clause / out-of-corpus, mirroring Java) all return responses; transcripts captured.
7. `uv run pytest -v` → green (test_health + test_ask).
8. `docs/observations/phase-2-findings.md` exists, contains pipeline numbers, per-query cost & latency, observed failure modes, and a cross-stack note vs. the Java repo.
9. README's Phase 2 row reads `[done](docs/observations/phase-2-findings.md)`. Phase table reflects the new 8-phase plan.

## Risks / decisions

- **LlamaIndex token counting accuracy.** `TokenCountingHandler` reads from LLM call metadata. For Anthropic, this maps to `response.usage.input_tokens` / `output_tokens`. Should be authoritative — same numbers as a direct SDK call. If the handler reports zero, fall back to reading `response.metadata` on the LlamaIndex `Response` object. Resolve at runtime; don't pre-engineer fallback.
- **Ingestion in Docker.** The `command:` step needs the postgres container to be ready. Use a healthcheck on the postgres service + `depends_on: condition: service_healthy` on the app. If the app's entrypoint runs ingest before uvicorn, the postgres healthcheck must finish first.
- **OpenAI ChatModel auto-config.** Java had to disable Spring AI's OpenAI ChatModel auto-config because we only wanted embeddings. LlamaIndex doesn't have this issue — `OpenAIEmbedding` and the chat LLM are separate classes you instantiate explicitly. No analogous problem.
- **Concurrent CLI runs.** Two ingest CLIs racing each other could both pass the doc-hash check, both compute chunk IDs, both try to INSERT. Postgres' UNIQUE constraint on `id` would reject duplicates; one CLI wins, the other gets an integrity error. Acceptable for a basic phase. Not worth fixing now.
- **Schema drift.** No migrations; if we ever change the `corpus_metadata` schema, manual `ALTER TABLE` or `DROP TABLE corpus_metadata` is required. Acceptable at this scale.

## Phase plan update

Phase brainstorming surfaced two additional depth-pick phases. The project goes from 6 phases to 8. New phase table:

| Phase | Topic | Status |
|---|---|---|
| 0 | Scaffolding | done |
| 1 | Long-context measurement | done |
| 2 | Basic RAG | (this design) |
| 3 | Honesty layer | structured `{can_answer, reason, answer, citations}`; scope check |
| 4 | Eval harness | golden Q&A; retrieval@K; faithfulness scoring |
| 5 | Structure-aware chunking | regex-split on clause IDs; strip repeating page boilerplate |
| 6 | Reranking | retrieve top-N → cross-encoder rerank → top-K |
| 7 | Multi-query retrieval | LLM expands user question into N variants; merge candidates |

Order rationale: eval before any tuning (Phase 4 measures everything that follows); chunking before reranking (foundational fix to the source data); reranking before multi-query (smaller change first; multi-query benefits from reranking already in place).

The README's phase table will be updated to reflect this when Phase 2 ships.

## Out of scope (Phases 3+)

- Structured refusal envelope, scope check, `can_answer` bool — Phase 3.
- Eval harness, golden dataset, retrieval@K, faithfulness scoring — Phase 4.
- Structure-aware chunking, header/footer stripping — Phase 5.
- Cross-encoder reranking — Phase 6.
- Multi-query retrieval, query expansion — Phase 7.
- Streaming token-by-token responses — not currently slated.
- Prompt caching for follow-up queries — not currently slated.
