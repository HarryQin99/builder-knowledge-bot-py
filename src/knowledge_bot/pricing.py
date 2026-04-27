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
