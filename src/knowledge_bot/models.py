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
