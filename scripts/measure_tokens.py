"""Phase 1 measurement: count tokens in the NCC 2022 Vol 2 corpus.

Confirms via Anthropic's authoritative tokenizer that the corpus exceeds
Claude's 200K context window. Output goes to stdout; results land in
docs/observations/phase-1-findings.md.

Run: uv run python scripts/measure_tokens.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import anthropic
import pymupdf
from dotenv import load_dotenv

CORPUS_PATH = Path("corpus/ncc-2022-vol2.pdf")
MODEL = "claude-haiku-4-5"
WINDOW_TOKENS = 200_000
HEADROOM_TOKENS = 10_000  # system prompt + question + answer
CEILING_TOKENS = WINDOW_TOKENS - HEADROOM_TOKENS


def load_pdf_text(path: Path) -> tuple[str, int, float]:
    """Extract concatenated text from every page of the PDF.

    Returns (text, page_count, load_seconds).
    """
    if not path.exists():
        sys.exit(
            f"Corpus not found at {path}. "
            f"See README setup section: copy NCC 2022 Vol 2 PDF into corpus/."
        )
    start = time.perf_counter()
    doc = pymupdf.open(str(path))
    try:
        page_count = len(doc)
        pages = [page.get_text() for page in doc]
    finally:
        doc.close()
    text = "\n".join(pages)
    elapsed = time.perf_counter() - start
    return text, page_count, elapsed


def count_claude_tokens(text: str) -> int:
    """Authoritative token count via Anthropic's count_tokens endpoint."""
    client = anthropic.Anthropic()
    response = client.messages.count_tokens(
        model=MODEL,
        messages=[{"role": "user", "content": text}],
    )
    return response.input_tokens


def print_results(
    *,
    page_count: int,
    char_count: int,
    chars_div_4: int,
    anthropic_tokens: int,
    load_seconds: float,
) -> None:
    rows = [
        ("Document", "NCC 2022 Volume 2 (Housing Provisions)"),
        ("Pages", f"{page_count}"),
        ("Characters", f"{char_count:,}"),
        ("chars / 4 estimate", f"{chars_div_4:,}"),
        (f"Anthropic count_tokens ({MODEL})", f"{anthropic_tokens:,}"),
        ("Load time (pymupdf)", f"{load_seconds:.2f}s"),
        ("% of 200K window", f"{anthropic_tokens / WINDOW_TOKENS:.1%}"),
        ("% of 190K ceiling", f"{anthropic_tokens / CEILING_TOKENS:.1%}"),
    ]
    width = max(len(k) for k, _ in rows)
    print()
    for k, v in rows:
        print(f"  {k.ljust(width)}  {v}")
    print()


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing. Set it in .env or the environment.")

    text, page_count, load_seconds = load_pdf_text(CORPUS_PATH)
    char_count = len(text)
    chars_div_4 = char_count // 4
    anthropic_tokens = count_claude_tokens(text)

    print_results(
        page_count=page_count,
        char_count=char_count,
        chars_div_4=chars_div_4,
        anthropic_tokens=anthropic_tokens,
        load_seconds=load_seconds,
    )


if __name__ == "__main__":
    main()
