# Phase 1 Findings: Long-Context Measurement (Python)

## Corpus measurement

| Metric | Value |
|---|---|
| Document | NCC 2022 Volume 2 (Housing Provisions) |
| Pages | 312 |
| Characters extracted | 624,397 |
| chars / 4 estimate | 156,099 |
| Anthropic `count_tokens` (claude-haiku-4-5) | **190,412** |
| Load time (pymupdf) | 0.60s |
| % of Claude 200K window | 95.2% |
| % of 190K ceiling (window − 10K headroom) | 100.2% |

The headline number — **190,412 tokens** — is from Anthropic's `messages.count_tokens` endpoint against the model we will use in Phase 2 (`claude-haiku-4-5`). It is the authoritative count; it is not a heuristic.

## Finding: the corpus does not safely fit in long-context

At 190,412 tokens, the full NCC Vol 2 sits at **95.2% of Claude's 200K window** and **just over the 190K ceiling** once 10K of headroom is reserved for system prompt, question, and answer.

This is not "3× too big to fit" — it is "fits with zero margin," which is operationally the same problem:

- One follow-up question with conversation history pushes the prompt over the limit.
- Any future corpus growth (an NCC amendment, a second volume) breaks the build.
- The system prompt for an honest-answering RAG agent (Phase 3) is non-trivial — eating into the 10K headroom.
- Claude's recall on near-window prompts degrades; even when the call succeeds, the answer quality suffers.

A working long-context baseline would require shaving the corpus, the model, or the safety margin. Each of those choices would defeat the point of measuring against a real-world document.

## Why we trusted the Python number

The Python measurement was triangulated against three independent extractors and one reference CLI, all of which agreed within ~2%:

| Extractor | Characters extracted |
|---|---|
| `pypdf` | 623,527 |
| `pdfplumber` | 614,355 |
| `pymupdf` (chosen) | 624,397 |
| Apache PDFBox CLI (`pdfbox-app:export:text`) | 628,299 |

Convergence across four implementations — two pure Python, one C-bound, and one JVM reference — rules out an extractor-quality bug.

`pymupdf` was chosen for the script because it is the fastest of the three Python options, has the cleanest API for page iteration, and is the production-grade extractor used in most RAG pipelines (relevant for Phase 2 ingestion).

## Cross-stack note

This repo's [Java sibling](https://github.com/Harryqin99/build-knowledge-bot) reported "607K tokens (3.2× the 200K window)" in its Phase 1 findings, derived as `2,427,096 chars / 4`. That estimate does not replicate against any of the four extractors above — Apache PDFBox's own CLI extracts 628K characters, not 2.4M. The most likely explanation is that Spring AI's `PagePdfDocumentReader` wraps PDFBox's raw text with per-page metadata and formatting markers; the Java side appears to have summed `Document.getText().length()` over those wrapped page documents, conflating extracted text with framework-added overhead.

Same conclusion ("the long-context approach is impractical for this corpus") survives both measurements. The route is just sharper here:

- Java side: chars/4 heuristic on a wrapper-inflated char count, predicting 3× over the window.
- Python side: authoritative tokenizer on raw extracted text, measuring 95% of the window with no headroom.

Either reading motivates RAG; the Python reading does so without relying on a heuristic.

## What this means for Phase 2

1. **Chunking + embeddings** keep per-query token usage well below the window — typically <5K tokens of retrieved context per question, instead of the entire corpus.
2. **Retrieval** turns "stuff everything into the prompt" into "find the relevant 4–8 passages." The pgvector + LlamaIndex path planned for Phase 2 follows the same shape as the Java repo's Spring AI / pgvector implementation, holding the corpus, embedding model, and chunk-size parameters constant for a fair cross-stack comparison.
3. **The `/ask` endpoint** gets built in Phase 2 wired to RAG from the start — there is no useful long-context baseline to build first, since "barely fits" is not a representative measurement of long-context pain.

Phase 2 plan to follow.
