# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Shared utilities for QueryRouter++."""

from __future__ import annotations

from queryrouter.data.loaders import ModelProfile


def estimate_query_cost(
    model: ModelProfile,
    total_tokens: int = 1000,
    expected_output_tokens: int | None = None,
) -> float:
    """Estimate the cost of a query on a model.

    Args:
        model: Model profile with pricing info.
        total_tokens: Estimated total token count (used if expected_output_tokens not provided).
        expected_output_tokens: Expected output token count. If provided, assumes remaining
            tokens are input (60/40 split is ignored).

    Returns:
        Estimated cost in USD.
    """
    if expected_output_tokens is not None:
        # Use explicit output token count
        output_tokens = expected_output_tokens
        input_tokens = max(1, total_tokens - output_tokens)
    else:
        # Fall back to 60/40 split
        input_tokens = int(total_tokens * 0.6)
        output_tokens = total_tokens - input_tokens
    
    return (
        model.cost_input_per_1m * input_tokens / 1_000_000
        + model.cost_output_per_1m * output_tokens / 1_000_000
    )
