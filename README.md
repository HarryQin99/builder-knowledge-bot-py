# Knowledge Bot (Python)

A RAG-based knowledge assistant over the **Australian National Construction Code (NCC) 2022 Volume 2** — residential buildings.

This is a **parallel implementation** of the [Java/Spring AI knowledge-bot](https://github.com/Harryqin99/build-knowledge-bot) project. Same five phases, same corpus, same models, same prompts — different language, framework, and ecosystem. The point is to demonstrate cross-stack engineering judgment by running the same RAG problem through two production-grade stacks and reporting the tradeoffs.

The README *is* the deliverable. This repo + the Java repo together cover both the backend and AI engineering portfolio angles.

## Stack

- Python 3.12 + `uv`
- FastAPI + Pydantic v2
- LlamaIndex (Phase 2+)
- Anthropic Claude via `anthropic` SDK (Phase 2+)
- OpenAI `text-embedding-3-small` (Phase 2+)
- Postgres + pgvector (Phase 2+)

Models, embedding provider, chunk size, top-K, and system prompts are held constant with the Java repo so the cross-stack comparison stays fair.

## Setup

```bash
# 1. Get the corpus
#    → https://ncc.abcb.gov.au, register free, download NCC 2022 Vol 2 PDF
#    → save as corpus/ncc-2022-vol2.pdf

# 2. Set API keys
cp .env.example .env
# edit .env, fill in ANTHROPIC_API_KEY and OPENAI_API_KEY

# 3a. Run locally (Phase 0+)
uv sync
uv run uvicorn knowledge_bot.main:app --reload
curl localhost:8000/health   # {"status":"ok"}

# 3b. Run via docker-compose (Phase 0+, includes Postgres for Phase 2+)
docker-compose up --build
curl localhost:8000/health
```

## Phases

| Phase | Status | What |
|---|---|---|
| 0 — Scaffolding | in progress | uv, FastAPI, Docker, compose, /health |
| 1 — Long-context measurement | — | tiktoken-accurate token count, confirms Java's chars/4 estimate, no full implementation (Java already proved infeasibility — see Java repo's [phase-1-findings](https://github.com/Harryqin99/build-knowledge-bot/blob/main/docs/observations/phase-1-findings.md)) |
| 2 — Basic RAG | — | LlamaIndex ingestion + query, pgvector, FastAPI `/ask` |
| 3 — Honesty layer | — | Pydantic-typed structured refusal envelope, scope check |
| 4 — Eval harness | — | golden Q&A set, retrieval@K, faithfulness scoring |
| 5 — Depth pick | — | one of: hybrid retrieval, re-ranking, streaming, prompt caching |

## Results

_To be filled after each phase._

## Cross-stack reading

This repo's Java sibling: [Harryqin99/build-knowledge-bot](https://github.com/Harryqin99/build-knowledge-bot). Phase 1 (long-context infeasibility) and Phase 2 (basic RAG) were originally implemented there; this repo reimplements them in Python and continues into Phase 3+. Compare:

- **Stack & framework choices** — Spring AI's `QuestionAnswerAdvisor` vs LlamaIndex's `QueryEngine`
- **Per-query cost & latency** — same models, same prompts, different orchestration overhead
- **Failure modes** — RAG quality issues are language-agnostic; framework choices affect what's easy to mitigate

Findings from each phase live under `docs/observations/`.
