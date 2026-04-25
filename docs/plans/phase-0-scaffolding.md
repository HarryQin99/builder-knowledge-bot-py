# Phase 0: Python Repo Scaffolding

**Goal:** Bootstrap an empty Python repo that boots, serves a FastAPI `/health` endpoint, and runs alongside Postgres+pgvector via docker-compose. Mirror the operational shape of the Java repo (multi-stage Dockerfile, Compose for local dev, `.env` for secrets) so cross-stack comparison stays apples-to-apples.

**Why this exists:** This Python repo is a parallel implementation of the [Java RAG project](https://github.com/Harryqin99/build-knowledge-bot) — same five phases, same problem (NCC 2022 Vol 2 QA), same models and prompts. The variables that differ are language, framework, and ecosystem. The point is to demonstrate cross-stack judgment, not to ship a different product.

---

## Approach

- **Package manager: `uv`.** Modern, fast, lockfile-driven. Replaces pip + venv + poetry.
- **Python 3.12.** Battle-tested current version; FastAPI + LlamaIndex + Pydantic v2 all stable on it.
- **Source layout: `src/knowledge_bot/`.** PEP 8 underscore name. Test discovery via `tests/`.
- **FastAPI + Pydantic v2** for the web layer. `uvicorn` for the dev server.
- **Multi-stage Dockerfile.** Build stage installs deps; runtime stage copies the venv + source. Same multi-stage *principle* as Java (smaller runtime image, no build tools shipped) — but Python is interpreted, so the build stage's job is dependency resolution + (optionally) bytecode pre-compilation, not compilation.
- **docker-compose mirrors Java repo's structure.** Same `pgvector/pgvector:pg16` image, same Postgres credentials shape, same env-var pattern from `.env`. Chunks land in the same shape so the two repos can be diffed at the data layer if needed.
- **Defer LlamaIndex / Anthropic / OpenAI / pgvector deps to Phase 1+.** Phase 0 is purely "FastAPI + Docker + Compose works"; pulling in 30 ML deps now is noise.
- **Cross-link in READMEs at the bottom**, both directions. Don't bury the relationship; don't lead with it either.

### Stack lock (decided in conversation, recorded here)

| Concern | Choice |
|---|---|
| Package manager | `uv` |
| Python | 3.12 |
| Web framework | FastAPI |
| Validation | Pydantic v2 |
| Test runner | `pytest` + `httpx` for FastAPI test client |
| Linter / formatter | `ruff` |
| Vector store (Phase 2+) | pgvector (same as Java for fair comparison) |
| RAG framework (Phase 2+) | LlamaIndex |
| Models (Phase 2+) | OpenAI `text-embedding-3-small` + Anthropic `claude-haiku-4-5` (same as Java) |

---

## TODO

- [x] Create `.gitignore` (Python: `__pycache__/`, `.venv/`, `.env`, `dist/`, `*.egg-info`, `.pytest_cache/`, `.ruff_cache/`, `corpus/*.pdf`)
- [x] `uv init --package knowledge-bot --python 3.12` → produces `pyproject.toml` with `src/` layout
- [x] Add Phase 0 deps via `uv add`: `fastapi`, `uvicorn[standard]`, `pydantic`, `python-dotenv`
- [x] Add dev deps via `uv add --dev`: `pytest`, `httpx`, `ruff`
- [x] Source skeleton: `src/knowledge_bot/__init__.py`, `src/knowledge_bot/main.py` (FastAPI app + `/health` endpoint)
- [x] Test skeleton: `tests/__init__.py`, `tests/test_health.py` (asserts `GET /health` → 200, `{"status": "ok"}`)
- [x] `Dockerfile` (multi-stage: build = install deps into `.venv`; runtime = copy `.venv` + `src/`, run uvicorn)
- [x] `.dockerignore` (exclude `.git`, `.venv`, `__pycache__`, `.pytest_cache`, `corpus/*.pdf`, `.env*`)
- [x] `docker-compose.yml` — `postgres` (`pgvector/pgvector:pg16`, same creds shape as Java) + `app` services. Postgres on host port 5433 (not 5432) so it can run alongside the Java repo's compose
- [x] `.env.example` — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` placeholders
- [x] `README.md` skeleton — stack, phases table (mirroring Java's), Setup section with `uv` + `docker-compose` paths
- [x] Local smoke test: `uv run uvicorn knowledge_bot.main:app --reload` → `curl localhost:8000/health` → 200
- [x] Docker smoke test: `docker-compose up --build` → `curl localhost:8000/health` → 200
- [x] Cross-link from Java repo's `README.md` (separate commit in Java repo, pointing here)
- [x] Cross-link from this repo's `README.md` to the Java repo

---

## Verification

1. `uv run pytest` → `tests/test_health.py` green
2. `uv run uvicorn knowledge_bot.main:app` then `curl localhost:8000/health` → `{"status":"ok"}`
3. `docker-compose up --build` then `curl localhost:8000/health` from host → `{"status":"ok"}`
4. `docker-compose exec postgres psql -U knowledgebot -d knowledgebot -c "\dt"` → empty (no tables yet — pgvector schema is created by Phase 2 via LlamaIndex / SQLAlchemy)
5. Both READMEs link to each other at the bottom

## Out of scope (Phase 1+)

- Token measurement script (`tiktoken`-based): Phase 1
- LlamaIndex pipeline, ingestion, retrieval, `/ask`: Phase 2
- Refusal layer, scope check, structured `can_answer` envelope: Phase 3
- Eval harness with golden Q&A: Phase 4
- Depth pick (hybrid retrieval / re-ranking / streaming / etc.): Phase 5
