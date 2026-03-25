# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Router evaluation module for QueryRouter++.

description: Evaluate routing quality with metrics like accuracy vs oracle,
    average cost, CO2 savings, and Pareto efficiency.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from queryrouter.api.schemas import RoutingRequest, UserPreferences
from queryrouter.core.router import QueryRouter


@dataclass
class EvaluationReport:
    """Evaluation metrics for a routing strategy.

    Attributes:
        strategy: Name of the routing strategy evaluated.
        accuracy_vs_oracle: Fraction of queries where the router picked
            the same model as the oracle (all-model scorer).
        avg_cost_per_query: Average estimated cost in USD per query.
        avg_latency_ms: Average estimated latency in milliseconds.
        co2_saved_vs_best_only: Percentage of CO2 saved compared to
            always routing to the highest-performance model.
        pareto_efficiency_score: Fraction of selected models that lie
            on the Pareto front.
        n_queries: Number of queries evaluated.
        model_distribution: Distribution of model selections.
    """

    strategy: str
    accuracy_vs_oracle: float = 0.0
    avg_cost_per_query: float = 0.0
    avg_latency_ms: float = 0.0
    co2_saved_vs_best_only: float = 0.0
    pareto_efficiency_score: float = 0.0
    n_queries: int = 0
    model_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class ComparisonReport:
    """Comparison of multiple routing strategies.

    Attributes:
        reports: Per-strategy evaluation reports.
        best_accuracy_strategy: Strategy with highest oracle accuracy.
        best_cost_strategy: Strategy with lowest average cost.
    """

    reports: dict[str, EvaluationReport] = field(default_factory=dict)
    best_accuracy_strategy: str = ""
    best_cost_strategy: str = ""


class RouterEvaluator:
    """Evaluate and compare routing strategies.

    Runs a test set through one or more QueryRouter instances and
    computes quality metrics including accuracy vs an oracle router,
    cost efficiency, and Pareto optimality.

    Args:
        preferences: Default user preferences for evaluation.

    Example:
        >>> evaluator = RouterEvaluator()
        >>> router = QueryRouter(strategy="direct", data_dir=data_path)
        >>> report = evaluator.evaluate(router, test_set)
        >>> report.accuracy_vs_oracle
        0.72
    """

    def __init__(
        self,
        preferences: UserPreferences | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            preferences: Default preferences for test queries.
                Defaults to balanced preset.
        """
        self.preferences = preferences or UserPreferences(optimize_for="balanced")

    def evaluate(
        self,
        router: QueryRouter,
        test_set: list[tuple[str, str]],
    ) -> EvaluationReport:
        """Evaluate a router on a test set.

        Args:
            router: The QueryRouter instance to evaluate.
            test_set: List of (query_string, oracle_model_id) tuples.

        Returns:
            EvaluationReport with computed metrics.
        """
        correct = 0
        total_cost = 0.0
        total_latency = 0.0
        model_counts: dict[str, int] = {}
        best_model_co2 = 0.0
        router_co2 = 0.0

        for query, oracle_model_id in test_set:
            request = RoutingRequest(
                query=query,
                preferences=self.preferences,
            )
            response = router.route(request)
            selected = response.recommended_model

            if selected == oracle_model_id:
                correct += 1

            total_cost += response.estimated_cost_usd
            total_latency += response.estimated_latency_ms

            model_counts[selected] = model_counts.get(selected, 0) + 1

            # CO2 comparison: estimate eco footprint
            try:
                selected_profile = router.registry.get_by_id(selected)
                selected_co2 = selected_profile.inference_co2_per_1m_grams or 50.0
            except KeyError:
                selected_co2 = 50.0

            try:
                oracle_profile = router.registry.get_by_id(oracle_model_id)
                oracle_co2 = oracle_profile.inference_co2_per_1m_grams or 50.0
            except KeyError:
                oracle_co2 = 50.0

            best_model_co2 += oracle_co2
            router_co2 += selected_co2

        n = max(len(test_set), 1)
        co2_saved = (
            (best_model_co2 - router_co2) / best_model_co2 * 100.0
            if best_model_co2 > 0
            else 0.0
        )

        # Pareto efficiency: check if selected models are Pareto-optimal
        pareto_count = self._count_pareto_selections(router, test_set)

        return EvaluationReport(
            strategy=router.strategy,
            accuracy_vs_oracle=correct / n,
            avg_cost_per_query=total_cost / n,
            avg_latency_ms=total_latency / n,
            co2_saved_vs_best_only=co2_saved,
            pareto_efficiency_score=pareto_count / n,
            n_queries=len(test_set),
            model_distribution=model_counts,
        )

    def compare_strategies(
        self,
        strategies: list[str],
        test_set: list[tuple[str, str]],
        data_dir: str | None = None,
    ) -> ComparisonReport:
        """Compare multiple routing strategies on the same test set.

        Args:
            strategies: List of strategy names ("direct", "cascade", "embedding").
            test_set: List of (query, oracle_model_id) tuples.
            data_dir: Path to data directory (passed to each router).

        Returns:
            ComparisonReport with per-strategy reports and rankings.
        """
        from pathlib import Path

        reports: dict[str, EvaluationReport] = {}

        for strategy in strategies:
            router = QueryRouter(
                strategy=strategy,  # type: ignore[arg-type]
                data_dir=Path(data_dir) if data_dir else None,
            )
            reports[strategy] = self.evaluate(router, test_set)

        best_accuracy = max(reports, key=lambda s: reports[s].accuracy_vs_oracle)
        best_cost = min(reports, key=lambda s: reports[s].avg_cost_per_query)

        return ComparisonReport(
            reports=reports,
            best_accuracy_strategy=best_accuracy,
            best_cost_strategy=best_cost,
        )

    def _count_pareto_selections(
        self,
        router: QueryRouter,
        test_set: list[tuple[str, str]],
    ) -> int:
        """Count how many selections are Pareto-optimal.

        Args:
            router: Router instance.
            test_set: Test queries.

        Returns:
            Number of Pareto-optimal selections.
        """
        from queryrouter.core.compatibility_scorer import WeightVector

        count = 0
        weights = WeightVector()  # balanced

        for query, _ in test_set:
            features = router.featurizer.featurize(query)
            models = router.registry.get_all()
            scored = router.scorer.score_all(features, models, weights)

            if not scored:
                continue

            # Get the selected model's axis scores
            request = RoutingRequest(query=query, preferences=self.preferences)
            response = router.route(request)
            selected_id = response.recommended_model

            selected_breakdown = None
            for ms in scored:
                if ms.model_id == selected_id:
                    selected_breakdown = ms.breakdown
                    break

            if selected_breakdown is None:
                continue

            # Check Pareto dominance
            is_dominated = False
            for ms in scored:
                if ms.model_id == selected_id:
                    continue
                dominates = all(
                    ms.breakdown.get(k, 0) >= selected_breakdown.get(k, 0)
                    for k in selected_breakdown
                ) and any(
                    ms.breakdown.get(k, 0) > selected_breakdown.get(k, 0)
                    for k in selected_breakdown
                )
                if dominates:
                    is_dominated = True
                    break

            if not is_dominated:
                count += 1

        return count
