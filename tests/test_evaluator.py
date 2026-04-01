# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Unit tests for RouterEvaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from queryrouter.api.schemas import UserPreferences
from queryrouter.core.router import QueryRouter
from queryrouter.training.evaluator import ComparisonReport, EvaluationReport, RouterEvaluator

DATA_DIR = Path(__file__).resolve().parents[1] / "data_models"


@pytest.fixture
def evaluator() -> RouterEvaluator:
    return RouterEvaluator(preferences=UserPreferences(optimize_for="balanced"))


@pytest.fixture
def router() -> QueryRouter:
    return QueryRouter(strategy="direct", data_dir=DATA_DIR)


def _test_set(router: QueryRouter) -> list[tuple[str, str]]:
    """Build a small test set using the first model as oracle."""
    model_ids = router.registry.list_ids()
    return [
        ("Write a Python quicksort", model_ids[0]),
        ("What is 2+2?", model_ids[1] if len(model_ids) > 1 else model_ids[0]),
        ("Summarize this article", model_ids[0]),
    ]


class TestEvaluate:
    def test_returns_report(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        report = evaluator.evaluate(router, _test_set(router))
        assert isinstance(report, EvaluationReport)

    def test_n_queries(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        ts = _test_set(router)
        report = evaluator.evaluate(router, ts)
        assert report.n_queries == len(ts)

    def test_accuracy_in_range(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        report = evaluator.evaluate(router, _test_set(router))
        assert 0.0 <= report.accuracy_vs_oracle <= 1.0

    def test_avg_cost_non_negative(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        report = evaluator.evaluate(router, _test_set(router))
        assert report.avg_cost_per_query >= 0.0

    def test_model_distribution_sums(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        ts = _test_set(router)
        report = evaluator.evaluate(router, ts)
        assert sum(report.model_distribution.values()) == len(ts)

    def test_strategy_recorded(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        report = evaluator.evaluate(router, _test_set(router))
        assert report.strategy == "direct"


class TestCompareStrategies:
    def test_returns_comparison(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        ts = _test_set(router)
        result = evaluator.compare_strategies(
            ["direct", "cascade"],
            ts,
            data_dir=str(DATA_DIR),
        )
        assert isinstance(result, ComparisonReport)
        assert "direct" in result.reports
        assert "cascade" in result.reports

    def test_best_strategies_set(self, evaluator: RouterEvaluator, router: QueryRouter) -> None:
        ts = _test_set(router)
        result = evaluator.compare_strategies(
            ["direct", "cascade"],
            ts,
            data_dir=str(DATA_DIR),
        )
        assert result.best_accuracy_strategy in ("direct", "cascade")
        assert result.best_cost_strategy in ("direct", "cascade")
