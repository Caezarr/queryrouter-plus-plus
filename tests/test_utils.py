# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Unit tests for shared utilities."""

from __future__ import annotations

import pytest

from queryrouter.data.loaders import ModelProfile
from queryrouter.data.utils import estimate_query_cost


class TestEstimateQueryCost:
    def test_basic_cost(self) -> None:
        model = ModelProfile(
            model_id="m1",
            name="M1",
            provider="P",
            cost_input_per_1m=10.0,
            cost_output_per_1m=20.0,
        )
        # 1000 tokens: 600 input, 400 output
        # 10 * 600 / 1_000_000 + 20 * 400 / 1_000_000
        # = 0.006 + 0.008 = 0.014
        cost = estimate_query_cost(model, 1000)
        assert cost == pytest.approx(0.014)

    def test_zero_cost_model(self) -> None:
        model = ModelProfile(
            model_id="free",
            name="Free",
            provider="P",
            cost_input_per_1m=0.0,
            cost_output_per_1m=0.0,
        )
        assert estimate_query_cost(model) == pytest.approx(0.0)

    def test_custom_token_count(self) -> None:
        model = ModelProfile(
            model_id="m1",
            name="M1",
            provider="P",
            cost_input_per_1m=1.0,
            cost_output_per_1m=2.0,
        )
        cost = estimate_query_cost(model, 5000)
        # 3000 input, 2000 output
        # 1 * 3000 / 1_000_000 + 2 * 2000 / 1_000_000 = 0.003 + 0.004 = 0.007
        assert cost == pytest.approx(0.007)
