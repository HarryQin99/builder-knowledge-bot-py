# Phase 1: Long-Context Measurement (Python) — Design

**Status:** Approved 2026-04-27
**Phase:** 1 of 5
**Predecessor:** Phase 0 (`docs/plans/phase-0-scaffolding.md`) — complete in commit `2208dab`
**Sibling:** [Java repo Phase 1 findings](https://github.com/Harryqin99/build-knowledge-bot/blob/main/docs/observations/phase-1-findings.md)

## Goal

Confirm with Anthropic's authoritative tokenizer that the NCC 2022 Volume 2 corpus exceeds Claude's 200K context window by a margin no estimation error can close — the same conclusion the Java repo reached via the chars/4 heuristic, but with the real number. The deliverable is the writeup, not running infrastructure.

## Why this phase exists

The Java repo's Phase 1 outcome:

> The full NCC Vol 2 estimates at **607K tokens — 3.2× the ceiling**. There is no way to fit this corpus into a single Claude prompt. The chars/4 heuristic could be off by ±20%, but no estimation error closes a 3× gap.

That conclusion is settled. Python's job here is *not* to re-fight it. Python's job is to:

1. Replace the chars/4 heuristic with a Claude-authoritative count.
2. Produce a clean cross-stack writeup that contrasts the Java estimate with the Python exact count.
3. Demonstrate Python-side tooling for PDF + tokenizer interaction that the rest of the project will reuse.

The phase is intentionally a measurement spike. No `/ask` endpoint, no FastAPI startup checkpoint, no `CorpusLoader` module — the Java repo *also* skipped those after its measurement, for the same reason: the wall is the finding.

## Approach

### Scope: pure measurement spike

A standalone script + a findings doc. No changes to the FastAPI app, no new modules under `src/knowledge_bot/`, no automated tests. The script runs once locally, the number lands in the findings doc, the doc is the deliverable.

### Tokenizer: Anthropic `count_tokens` API only

Anthropic's `client.messages.count_tokens(model="claude-haiku-4-5", messages=[...])` is the authoritative source for "does this fit in Claude's 200K window?" It is free / unbilled.

`tiktoken` is OpenAI's tokenizer — it would only confirm the chars/4 heuristic against itself (chars/4 is derived from tiktoken). It is the wrong tool for a Claude-window question. Skipped here; Phase 2 may reintroduce it for OpenAI embedding cost calculations.

### PDF extraction: `pypdf`

Pure Python, no native dependencies, simplest for a one-shot script. Phase 2 may pick a different library (`pdfplumber`, `PyMuPDF`) for chunking-quality reasons, but that is a Phase 2 decision and does not block Phase 1.

### Model: `claude-haiku-4-5`

Matches the model locked in the Phase 0 plan and the Java repo's Phase 1 — the count_tokens API takes a `model` parameter, and we want the count for the model we will actually use.

### Corpus: full NCC 2022 Vol 2

Same source PDF as Java (`/Users/harry/Development/build-knowledge-bot/corpus/ncc-2022-vol2.pdf`). Copied into `corpus/ncc-2022-vol2.pdf` once at execution time. Stays gitignored. We measure the *full* corpus — the whole point is the headline number for the entire document.

### Stack lock

| Concern | Choice |
|---|---|
| PDF extraction | `pypdf` |
| Tokenizer | Anthropic `messages.count_tokens` |
| Model id (for token count) | `claude-haiku-4-5` |
| Env loading | `python-dotenv` (already a Phase 0 dep) |
| Script home | `scripts/measure_tokens.py` |

## Components

### `scripts/measure_tokens.py`

A standalone Python script invoked as `uv run python scripts/measure_tokens.py`. Not part of the FastAPI app, not imported by anything in `src/knowledge_bot/`.

**Behaviour:**

1. Load `.env` via `python-dotenv`; fail fast if `ANTHROPIC_API_KEY` is missing.
2. Open `corpus/ncc-2022-vol2.pdf` with `pypdf.PdfReader`. Fail fast with a clear message if the file does not exist (point user at the README setup step).
3. Iterate over `reader.pages`, call `.extract_text()` on each, concatenate into a single string.
4. Compute `len(text)` (characters) and `len(text) // 4` (chars/4 estimate, matches Java's heuristic).
5. Call `anthropic.Anthropic().messages.count_tokens(model="claude-haiku-4-5", messages=[{"role": "user", "content": text}])`. Read `.input_tokens` off the response.
6. Print a results table to stdout: page count, character count, chars/4 estimate, Anthropic exact count, ratio over 200K window, ratio over 190K ceiling (window minus 10K headroom for system prompt + question + answer).

The script is one file, ~50–80 lines, no abstractions. Three local helper functions at most: `load_pdf_text(path) -> str`, `print_results(...) -> None`, and a `main()` that wires them. No classes, no config objects. Numbers print to stdout; user pastes them into the findings doc.

### `docs/observations/phase-1-findings.md`

The deliverable. Structure mirrors the Java findings doc so the two read as a pair:

1. **Corpus measurement** — table with page count, characters, chars/4 estimate, Anthropic exact count, load time. Same row layout as Java's table, with one extra row for the Anthropic exact count.
2. **Finding: long-context approach is impossible for this corpus** — restates the conclusion using the new exact number. Notes whether the chars/4 heuristic was high or low and by what percentage.
3. **Why not use a subset** — same reasoning as Java (18K-token Part H4 subset doesn't produce a meaningful long-context baseline).
4. **What this means for Phase 2** — same forward-pointer to RAG.
5. **Cross-stack note** — one paragraph: "Java estimated 607K tokens via chars/4; Python's Anthropic count_tokens returned X tokens. The heuristic was off by Y%. The 3× gap holds regardless." Links to the Java findings doc.

The doc is self-contained. A reader landing on the Python repo gets the full conclusion without needing to read the Java side first.

### README update

Phase 1 row in `README.md`'s phases table changes from `—` to `done`. Status link points to `docs/observations/phase-1-findings.md`.

### Dependency additions

`uv add anthropic pypdf`. Both go to main deps (not dev-only) — `anthropic` is needed for Phase 2's `/ask` endpoint, `pypdf` will be the ingestion entry point for Phase 2 chunking unless we swap it then. Bringing them in now is not speculative; Phase 2 needs them.

## What is NOT in this phase

- No `/ask` endpoint, no FastAPI route changes, no `LongContextAnswerService` equivalent.
- No `CorpusLoader` module under `src/knowledge_bot/`. The script reads the PDF inline.
- No 190K-token startup checkpoint wired into the FastAPI app. The check exists only as a printed warning in the script.
- No automated tests. Pytest is unchanged from Phase 0.
- No `tiktoken`, no OpenAI integration, no embeddings, no pgvector.
- No model-pricing enum, no cost calculator. Phase 1 has no LLM inference call — `count_tokens` is unbilled — so there is nothing to cost.
- No Docker changes. Script runs locally via `uv run`; if you also want it to run in the container, that is a Phase 2 decision once we know what shape ingestion takes.

## Verification

1. `cp /Users/harry/Development/build-knowledge-bot/corpus/ncc-2022-vol2.pdf corpus/`
2. `uv run python scripts/measure_tokens.py` exits 0 and prints a results table including: page count ≈ 312, character count ≈ 2.4M, chars/4 estimate ≈ 607K, Anthropic exact count, percentage over 200K window, percentage over 190K ceiling.
3. `docs/observations/phase-1-findings.md` exists, contains the headline numbers, and links to the Java findings doc.
4. README's Phase 1 row reads `done` and links to the findings doc.
5. `git status` is clean after a single Phase 1 commit; the corpus PDF is not tracked.
6. `uv run pytest` still passes (Phase 0 tests unchanged).

## Risks / decisions

- **`pypdf.extract_text()` quality.** It is known to drop or reorder text in PDFs with complex layout. For a *measurement* purpose this matters less than for a *retrieval* purpose — even a 10% under-extraction would still leave the corpus 2.9× over the window. Acceptable for Phase 1; revisit in Phase 2.
- **`count_tokens` rate / size limits.** Anthropic's count_tokens accepts large message bodies, but a 2.4M-character payload may hit a request-size limit. If it does, the script falls back to counting in page-batched chunks and summing — caveats noted in the findings doc. (Resolve only if it actually trips; do not pre-engineer the fallback.)
- **PDF availability.** The corpus is gitignored. Anyone re-running the script needs the PDF locally first. The script's "file not found" error message points at the README setup step.

## Out of scope (Phase 2+)

LlamaIndex pipeline, ingestion, retrieval, `/ask` endpoint (Phase 2). Refusal layer, scope check, structured `can_answer` envelope (Phase 3). Eval harness with golden Q&A (Phase 4). Depth pick — hybrid retrieval / re-ranking / streaming / prompt caching (Phase 5).
