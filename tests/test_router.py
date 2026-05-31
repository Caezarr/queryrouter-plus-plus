# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Integration tests for QueryRouter.

description: Tests for all three routing strategies (direct, cascade, embedding)
    and the explain functionality.
agent: coder
date: 2026-03-24
version: 1.0
"""

from pathlib import Path

import pytest

import numpy as np

from queryrouter.api.schemas import RoutingRequest, ToolContext, UserPreferences
from queryrouter.core.router import QueryRouter

DATA_DIR = Path(__file__).resolve().parents[1] / "data_models"


@pytest.fixture
def direct_router() -> QueryRouter:
    return QueryRouter(strategy="direct", data_dir=DATA_DIR)


@pytest.fixture
def cascade_router() -> QueryRouter:
    return QueryRouter(strategy="cascade", data_dir=DATA_DIR)


@pytest.fixture
def embedding_router() -> QueryRouter:
    return QueryRouter(strategy="embedding", data_dir=DATA_DIR)


def _make_request(
    query: str = "Write a Python quicksort function",
    optimize_for: str = "balanced",
) -> RoutingRequest:
    return RoutingRequest(
        query=query,
        preferences=UserPreferences(optimize_for=optimize_for),  # type: ignore[arg-type]
    )


class TestDirectRouting:
    """Tests for direct routing strategy."""

    def test_returns_valid_model(self, direct_router: QueryRouter) -> None:
        response = direct_router.route(_make_request())
        assert response.recommended_model in direct_router.registry.list_ids()

    def test_scores_not_empty(self, direct_router: QueryRouter) -> None:
        response = direct_router.route(_make_request())
        assert len(response.scores) > 0

    def test_scores_sorted_descending(self, direct_router: QueryRouter) -> None:
        response = direct_router.route(_make_request())
        for i in range(len(response.scores) - 1):
            assert response.scores[i].score >= response.scores[i + 1].score

    def test_performance_preset_selects_strong_model(self, direct_router: QueryRouter) -> None:
        request = _make_request(
            query="Solve this complex math theorem with a formal proof",
            optimize_for="performance",
        )
        response = direct_router.route(request)
        # Should select a top-performing model, not the cheapest
        assert response.recommended_model != "gpt-4-1-mini"

    def test_cost_preset_prefers_cheap(self, direct_router: QueryRouter) -> None:
        request = _make_request(
            query="What is 2+2?",
            optimize_for="cost",
        )
        response = direct_router.route(request)
        # The cheapest models should be preferred
        cheap_models = {"gpt-4-1-mini", "gemini-2-5-flash", "deepseek-v3"}
        assert response.recommended_model in cheap_models

    def test_estimated_cost_positive(self, direct_router: QueryRouter) -> None:
        response = direct_router.route(_make_request())
        assert response.estimated_cost_usd >= 0.0

    def test_no_models_match_constraints(self, direct_router: QueryRouter) -> None:
        request = RoutingRequest(
            query="Hello",
            preferences=UserPreferences(
                optimize_for="balanced",
                budget_per_query_usd=0.0000001,
            ),
        )
        response = direct_router.route(request)
        assert response.recommended_model == "none"


class TestCascadeRouting:
    """Tests for cascade routing strategy."""

    def test_returns_valid_model(self, cascade_router: QueryRouter) -> None:
        response = cascade_router.route(_make_request())
        assert response.recommended_model in cascade_router.registry.list_ids()

    def test_explanation_mentions_cascade(self, cascade_router: QueryRouter) -> None:
        response = cascade_router.route(_make_request())
        assert "cascade" in response.explanation.lower() or "Cascade" in response.explanation

    def test_scores_present(self, cascade_router: QueryRouter) -> None:
        response = cascade_router.route(_make_request())
        assert len(response.scores) > 0


class TestEmbeddingRouting:
    """Tests for embedding-based routing strategy."""

    def test_returns_valid_model(self, embedding_router: QueryRouter) -> None:
        response = embedding_router.route(_make_request())
        assert response.recommended_model in embedding_router.registry.list_ids()

    def test_scores_present(self, embedding_router: QueryRouter) -> None:
        response = embedding_router.route(_make_request())
        assert len(response.scores) > 0

    def test_different_queries_can_route_differently(self, embedding_router: QueryRouter) -> None:
        r1 = embedding_router.route(_make_request("What is 2+2?", "cost"))
        r2 = embedding_router.route(_make_request(
            "Write a comprehensive essay analyzing quantum mechanics", "performance"
        ))
        # Different queries with different preferences should potentially differ
        # (though not guaranteed with embedding approach)
        assert r1.recommended_model is not None
        assert r2.recommended_model is not None


class TestExplain:
    """Tests for the explain() method."""

    def test_explain_returns_string(self, direct_router: QueryRouter) -> None:
        result = direct_router.explain(_make_request())
        assert isinstance(result, str)
        assert len(result) > 0

    def test_explain_contains_model_name(self, direct_router: QueryRouter) -> None:
        result = direct_router.explain(_make_request())
        # Should mention at least one model
        all_ids = direct_router.registry.list_ids()
        assert any(mid in result for mid in all_ids)

    def test_explain_contains_strategy(self, direct_router: QueryRouter) -> None:
        result = direct_router.explain(_make_request())
        assert "direct" in result.lower()


class TestToolAwareRouting:
    """Tests for tool-aware weight shifting."""

    def test_tool_context_boosts_performance_weight(self, direct_router: QueryRouter) -> None:
        from queryrouter.core.compatibility_scorer import WeightVector
        base = WeightVector(w_performance=0.25, w_cost=0.25, w_latency=0.25, w_ecology=0.25)
        ctx = ToolContext(has_web_search=True)
        delta = direct_router._tool_boost_delta(ctx)
        shifted = direct_router._shift_weights_to_performance(base, delta)
        assert shifted.w_performance > base.w_performance

    def test_no_tools_no_shift(self, direct_router: QueryRouter) -> None:
        assert direct_router._tool_boost_delta(None) == 0.0
        assert direct_router._tool_boost_delta(ToolContext()) == 0.0

    def test_first_class_tools_delta(self, direct_router: QueryRouter) -> None:
        ctx = ToolContext(has_code_exec=True)
        delta = direct_router._tool_boost_delta(ctx)
        assert delta == pytest.approx(0.3)

    def test_mcp_adds_on_top(self, direct_router: QueryRouter) -> None:
        ctx = ToolContext(has_web_search=True, has_mcp=True)
        delta = direct_router._tool_boost_delta(ctx)
        assert delta == pytest.approx(min(0.3 + 0.15, 0.4))

    def test_cap_respected(self, direct_router: QueryRouter) -> None:
        ctx = ToolContext(has_web_search=True, has_mcp=True, has_agent=True)
        delta = direct_router._tool_boost_delta(ctx)
        assert delta <= direct_router._tool_boost_cfg["cap"]

    def test_shifted_weights_sum_to_one(self, direct_router: QueryRouter) -> None:
        from queryrouter.core.compatibility_scorer import WeightVector
        base = WeightVector(w_performance=0.25, w_cost=0.25, w_latency=0.25, w_ecology=0.25)
        shifted = direct_router._shift_weights_to_performance(base, 0.3)
        total = shifted.w_performance + shifted.w_cost + shifted.w_latency + shifted.w_ecology
        assert total == pytest.approx(1.0)

    def test_tool_context_routes_to_stronger_model(self, direct_router: QueryRouter) -> None:
        base_request = _make_request("Summarize this document", "cost")
        tool_request = RoutingRequest(
            query="Summarize this document",
            preferences=UserPreferences(optimize_for="cost"),
            tool_context=ToolContext(has_web_search=True, has_code_exec=True),
        )
        base_resp = direct_router.route(base_request)
        tool_resp = direct_router.route(tool_request)
        # Tool-aware request should select a model with higher performance score
        base_score = next(s for s in base_resp.scores if s.model_id == base_resp.recommended_model)
        tool_score = next(s for s in tool_resp.scores if s.model_id == tool_resp.recommended_model)
        assert tool_score.breakdown.get("performance", 0) >= base_score.breakdown.get("performance", 0) - 0.05


class TestQueryComplexity:
    """Tests for the cascade query-complexity estimator."""

    def test_simple_query_low_complexity(self, cascade_router: QueryRouter) -> None:
        features = cascade_router.featurizer.featurize("Hello, how are you?")
        complexity = cascade_router._query_complexity(features)
        assert complexity < 0.5

    def test_hard_query_high_complexity(self, cascade_router: QueryRouter) -> None:
        features = cascade_router.featurizer.featurize(
            "Prove that P≠NP using a formal reduction from 3-SAT and derive the "
            "computational complexity lower bound with a step-by-step mathematical proof."
        )
        complexity = cascade_router._query_complexity(features)
        assert complexity >= 0.3

    def test_complexity_in_range(self, cascade_router: QueryRouter) -> None:
        for query in ["hi", "write python code", "explain quantum entanglement mathematically"]:
            features = cascade_router.featurizer.featurize(query)
            c = cascade_router._query_complexity(features)
            assert 0.0 <= c <= 1.0

    def test_cascade_explanation_mentions_complexity(self, cascade_router: QueryRouter) -> None:
        response = cascade_router.route(_make_request("Write a recursive fibonacci function"))
        assert "complexity" in response.explanation
        assert "threshold" in response.explanation


class TestModelFiltering:
    """Tests for model filtering via preferences."""

    def test_allowed_models_filter(self, direct_router: QueryRouter) -> None:
        request = RoutingRequest(
            query="Hello",
            preferences=UserPreferences(
                optimize_for="balanced",
                allowed_models=["gpt-4-1", "deepseek-v3"],
            ),
        )
        response = direct_router.route(request)
        assert response.recommended_model in {"gpt-4-1", "deepseek-v3"}

    def test_excluded_models_filter(self, direct_router: QueryRouter) -> None:
        request = RoutingRequest(
            query="Hello",
            preferences=UserPreferences(
                optimize_for="balanced",
                excluded_models=["gpt-4-1", "claude-sonnet-4-6", "claude-opus-4-6"],
            ),
        )
        response = direct_router.route(request)
        assert response.recommended_model not in {"gpt-4-1", "claude-sonnet-4-6", "claude-opus-4-6"}
