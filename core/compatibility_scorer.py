# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Compatibility scoring for QueryRouter++.

description: Implements the composite scoring function
    C(q, m, w) = sum_i w_i * c_i(phi(q), psi(m)) from the formal framework.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from queryrouter.data.loaders import ModelProfile
from queryrouter.data.normalizers import FeatureNormalizer


@dataclass
class WeightVector:
    """User preference weight vector w = (w_P, w_K, w_L, w_E).

    Attributes:
        w_performance: Weight for the performance axis.
        w_cost: Weight for the cost axis (inverted: lower cost = higher).
        w_latency: Weight for the latency axis (inverted: lower = higher).
        w_ecology: Weight for the ecology axis (inverted: lower CO2 = higher).
    """

    w_performance: float = 0.25
    w_cost: float = 0.25
    w_latency: float = 0.25
    w_ecology: float = 0.25

    def as_array(self) -> np.ndarray:
        """Return weights as a numpy array [P, K, L, E].

        Returns:
            Array of shape (4,).
        """
        return np.array(
            [self.w_performance, self.w_cost, self.w_latency, self.w_ecology],
            dtype=np.float64,
        )

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> WeightVector:
        """Create a WeightVector from a preferences dict.

        Args:
            d: Dict with keys w_performance, w_cost, w_latency, w_ecology.

        Returns:
            WeightVector instance.
        """
        return cls(
            w_performance=d.get("w_performance", 0.25),
            w_cost=d.get("w_cost", 0.25),
            w_latency=d.get("w_latency", 0.25),
            w_ecology=d.get("w_ecology", 0.25),
        )


@dataclass
class ModelScore:
    """Score result for a single model.

    Attributes:
        model_id: Unique model identifier.
        score: Overall composite score S(q, m, w) in [0, 1].
        breakdown: Per-axis scores {performance, cost, latency, ecology}.
    """

    model_id: str
    score: float
    breakdown: dict[str, float] = field(default_factory=dict)


# -- Benchmark-to-task mapping --
# Maps task type indices to relevant benchmark indices in the normalizer
TASK_BENCHMARK_MAP: dict[str, list[str]] = {
    "coding": ["humaneval"],
    "math": ["gsm8k", "math"],
    "creative": ["mmlu"],  # proxy: general knowledge
    "factual": ["mmlu", "hellaswag", "arc"],
    "reasoning": ["mmlu", "arc", "hellaswag"],
    "summarization": ["mmlu"],
    "translation": ["mmlu"],
    "classification": ["mmlu", "hellaswag"],
    "conversation": ["mmlu"],
    "debugging": ["humaneval"],
}

BENCHMARK_NAMES = ["mmlu", "humaneval", "gsm8k", "math", "hellaswag", "arc"]


class CompatibilityScorer:
    """Compute compatibility score C(q, m, w) = sum_i w_i * c_i(phi(q), psi(m)).

    Implements the composite scoring function from Definition 7.1 of
    the formal framework. Each axis scorer maps query features and
    model profile to a score in [0, 1].

    Attributes:
        normalizer: Fitted FeatureNormalizer for model profiles.
    """

    def __init__(self, normalizer: FeatureNormalizer) -> None:
        """Initialize with a fitted feature normalizer.

        Args:
            normalizer: A FeatureNormalizer that has been fit() on all model profiles.
        """
        self.normalizer = normalizer

    def score(
        self,
        query_features: np.ndarray,
        model_profile: ModelProfile,
        weights: WeightVector,
    ) -> ModelScore:
        """Compute the composite compatibility score for a single model.

        Args:
            query_features: Feature vector phi(q) of shape (N_FEATURES,).
            model_profile: The model profile to score.
            weights: User preference weight vector.

        Returns:
            ModelScore with overall score and per-axis breakdown.
        """
        perf = self._performance_score(query_features, model_profile)
        cost = self._cost_score(query_features, model_profile)
        lat = self._latency_score(query_features, model_profile)
        eco = self._ecology_score(query_features, model_profile)

        w = weights.as_array()
        axes = np.array([perf, cost, lat, eco], dtype=np.float64)
        total = float(np.dot(w, axes))

        return ModelScore(
            model_id=model_profile.model_id,
            score=total,
            breakdown={
                "performance": perf,
                "cost": cost,
                "latency": lat,
                "ecology": eco,
            },
        )

    def score_all(
        self,
        query_features: np.ndarray,
        models: list[ModelProfile],
        weights: WeightVector,
    ) -> list[ModelScore]:
        """Score all models and return sorted results (highest first).

        Args:
            query_features: Feature vector phi(q) of shape (N_FEATURES,).
            models: List of model profiles to score.
            weights: User preference weight vector.

        Returns:
            List of ModelScore objects sorted by score descending.
        """
        results = [self.score(query_features, m, weights) for m in models]
        results.sort(key=lambda ms: ms.score, reverse=True)
        return results

    def _performance_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Compute the performance axis score P(q, m).

        Uses task-type signals from query features to weight relevant
        benchmarks, implementing the formal P(q,m) = phi_task(q)^T * psi_perf(m).

        Args:
            query_features: Feature vector phi(q).
            model_profile: Model profile.

        Returns:
            Performance score in [0, 1].
        """
        # Extract task type scores from query features (indices 11-20)
        task_scores = query_features[11:21]

        # Get normalized benchmark scores for the model
        bench_vec = self.normalizer.benchmark_normalizer.transform(model_profile)

        # Build relevance weights: for each benchmark, sum the task scores
        # of task types that use that benchmark
        bench_weights = np.zeros(len(BENCHMARK_NAMES), dtype=np.float64)
        for task_idx, task_name in enumerate(
            ["coding", "math", "creative", "factual", "reasoning",
             "summarization", "translation", "classification",
             "conversation", "debugging"]
        ):
            relevant_benchmarks = TASK_BENCHMARK_MAP.get(task_name, [])
            for bname in relevant_benchmarks:
                if bname in BENCHMARK_NAMES:
                    bidx = BENCHMARK_NAMES.index(bname)
                    bench_weights[bidx] += task_scores[task_idx]

        # Normalize weights
        weight_sum = bench_weights.sum()
        if weight_sum > 0:
            bench_weights /= weight_sum
        else:
            # Equal weight fallback
            bench_weights = np.ones(len(BENCHMARK_NAMES)) / len(BENCHMARK_NAMES)

        perf = float(np.dot(bench_weights, bench_vec))
        return float(np.clip(perf, 0.0, 1.0))

    def cost_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Public interface for the cost axis score. See ``_cost_score``."""
        return self._cost_score(query_features, model_profile)

    def latency_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Public interface for the latency axis score. See ``_latency_score``."""
        return self._latency_score(query_features, model_profile)

    def ecology_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Public interface for the ecology axis score. See ``_ecology_score``."""
        return self._ecology_score(query_features, model_profile)

    def _cost_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Compute the cost axis score K(q, m).

        Args:
            query_features: Feature vector phi(q).
            model_profile: Model profile.

        Returns:
            Cost score in [0, 1] where 1.0 = cheapest.
        """
        return self.normalizer.cost_normalizer.transform(model_profile)

    def _latency_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Compute the latency axis score L(q, m).

        Args:
            query_features: Feature vector phi(q).
            model_profile: Model profile.

        Returns:
            Latency score in [0, 1] where 1.0 = fastest.
        """
        if model_profile.latency_ms is not None and self.normalizer.max_latency > 0:
            normalized = model_profile.latency_ms / self.normalizer.max_latency
            return float(np.clip(1.0 - normalized, 0.0, 1.0))
        return 0.5  # Default for unknown latency

    def _ecology_score(
        self, query_features: np.ndarray, model_profile: ModelProfile
    ) -> float:
        """Compute the ecology axis score E(q, m).

        Args:
            query_features: Feature vector phi(q).
            model_profile: Model profile.

        Returns:
            Ecology score in [0, 1] where 1.0 = most eco-friendly.
        """
        return self.normalizer.eco_normalizer.transform(model_profile)
