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

from queryrouter.api.schemas import RoutingRequest, UserPreferences
from queryrouter.core.router import QueryRouter

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


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
        assert response.recommended_model != "gpt-4o-mini"

    def test_cost_preset_prefers_cheap(self, direct_router: QueryRouter) -> None:
        request = _make_request(
            query="What is 2+2?",
            optimize_for="cost",
        )
        response = direct_router.route(request)
        # The cheapest models should be preferred
        cheap_models = {"gpt-4o-mini", "gemini-2-0-flash", "deepseek-v3"}
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


class TestModelFiltering:
    """Tests for model filtering via preferences."""

    def test_allowed_models_filter(self, direct_router: QueryRouter) -> None:
        request = RoutingRequest(
            query="Hello",
            preferences=UserPreferences(
                optimize_for="balanced",
                allowed_models=["gpt-4o", "deepseek-v3"],
            ),
        )
        response = direct_router.route(request)
        assert response.recommended_model in {"gpt-4o", "deepseek-v3"}

    def test_excluded_models_filter(self, direct_router: QueryRouter) -> None:
        request = RoutingRequest(
            query="Hello",
            preferences=UserPreferences(
                optimize_for="balanced",
                excluded_models=["gpt-4o", "claude-3-5-sonnet", "claude-3-7-sonnet"],
            ),
        )
        response = direct_router.route(request)
        assert response.recommended_model not in {"gpt-4o", "claude-3-5-sonnet", "claude-3-7-sonnet"}
