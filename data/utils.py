# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Shared utilities for QueryRouter++."""

from __future__ import annotations

from queryrouter.data.loaders import ModelProfile


def estimate_query_cost(model: ModelProfile, total_tokens: int = 1000) -> float:
    """Estimate the cost of a query on a model.

    Assumes a 60/40 split between input and output tokens.

    Args:
        model: Model profile with pricing info.
        total_tokens: Estimated total token count.

    Returns:
        Estimated cost in USD.
    """
    input_tokens = int(total_tokens * 0.6)
    output_tokens = total_tokens - input_tokens
    return (
        model.cost_input_per_1m * input_tokens / 1_000_000
        + model.cost_output_per_1m * output_tokens / 1_000_000
    )
