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
