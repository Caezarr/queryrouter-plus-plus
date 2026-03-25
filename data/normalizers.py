# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Feature normalization for QueryRouter++ model profiles.

description: Normalize benchmark, cost, and ecological scores to [0,1] scale
    for use in the compatibility scoring function S(q, m, w).
agent: coder
date: 2026-03-23
version: 1.0
"""

from __future__ import annotations

import numpy as np

from queryrouter.data.loaders import ModelProfile


class BenchmarkNormalizer:
    """Min-max normalize benchmark scores across the model registry.

    Each benchmark dimension is independently scaled to [0, 1] based on
    the observed min and max across all models. Missing values are imputed
    with the column mean before normalization.

    Attributes:
        benchmark_names: Ordered list of benchmark column names.
        mins: Per-benchmark minimum values (set after fit).
        maxs: Per-benchmark maximum values (set after fit).
        means: Per-benchmark mean values for imputation (set after fit).
    """

    BENCHMARK_NAMES: list[str] = [
        "mmlu",
        "humaneval",
        "gsm8k",
        "math",
        "hellaswag",
        "arc",
    ]

    def __init__(self) -> None:
        """Initialize the normalizer (unfitted)."""
        self.mins: np.ndarray = np.array([])
        self.maxs: np.ndarray = np.array([])
        self.means: np.ndarray = np.array([])
        self._fitted: bool = False

    def fit(self, profiles: list[ModelProfile]) -> None:
        """Compute min, max, and mean for each benchmark across all profiles.

        Args:
            profiles: List of ModelProfile objects to derive statistics from.
        """
        n_benchmarks = len(self.BENCHMARK_NAMES)
        values: list[list[float]] = [[] for _ in range(n_benchmarks)]

        for profile in profiles:
            for i, bname in enumerate(self.BENCHMARK_NAMES):
                val = profile.benchmarks.get(bname)
                if val is not None:
                    values[i].append(val)

        self.means = np.array(
            [np.mean(v) if v else 0.5 for v in values], dtype=np.float64
        )
        self.mins = np.array(
            [np.min(v) if v else 0.0 for v in values], dtype=np.float64
        )
        self.maxs = np.array(
            [np.max(v) if v else 1.0 for v in values], dtype=np.float64
        )
        self._fitted = True

    def transform(self, profile: ModelProfile) -> np.ndarray:
        """Normalize a single profile's benchmark scores to [0, 1].

        Missing benchmarks are imputed with the column mean before scaling.

        Args:
            profile: The model profile to normalize.

        Returns:
            Array of shape (n_benchmarks,) with values in [0, 1].

        Raises:
            RuntimeError: If the normalizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("BenchmarkNormalizer must be fit() before transform()")

        raw = np.array(
            [
                profile.benchmarks.get(bname, None)
                for bname in self.BENCHMARK_NAMES
            ],
            dtype=object,
        )

        filled = np.array(
            [
                float(raw[i]) if raw[i] is not None else self.means[i]
                for i in range(len(self.BENCHMARK_NAMES))
            ],
            dtype=np.float64,
        )

        denom = self.maxs - self.mins
        denom = np.where(denom == 0, 1.0, denom)
        normalized = (filled - self.mins) / denom
        return np.clip(normalized, 0.0, 1.0)


class CostNormalizer:
    """Normalize cost values to [0, 1] with inversion (lower cost = higher score).

    Uses min-max scaling on the total cost (input + output) across models,
    then inverts so that the cheapest model scores 1.0 and the most expensive
    scores 0.0.

    Attributes:
        min_cost: Minimum observed total cost per 1M tokens.
        max_cost: Maximum observed total cost per 1M tokens.
    """

    def __init__(self) -> None:
        """Initialize the normalizer (unfitted)."""
        self.min_cost: float = 0.0
        self.max_cost: float = 1.0
        self._fitted: bool = False

    def fit(self, profiles: list[ModelProfile]) -> None:
        """Compute min and max total cost across all profiles.

        Args:
            profiles: List of ModelProfile objects.
        """
        costs = [p.cost_input_per_1m + p.cost_output_per_1m for p in profiles]
        if costs:
            self.min_cost = min(costs)
            self.max_cost = max(costs)
        self._fitted = True

    def transform(self, profile: ModelProfile) -> float:
        """Return inverted normalized cost score for a model.

        Args:
            profile: The model profile to score.

        Returns:
            Score in [0, 1] where 1.0 = cheapest and 0.0 = most expensive.

        Raises:
            RuntimeError: If the normalizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("CostNormalizer must be fit() before transform()")

        total_cost = profile.cost_input_per_1m + profile.cost_output_per_1m
        denom = self.max_cost - self.min_cost
        if denom == 0:
            return 1.0
        normalized = (total_cost - self.min_cost) / denom
        return float(np.clip(1.0 - normalized, 0.0, 1.0))


class EcoNormalizer:
    """Normalize ecological impact scores to [0, 1] (lower impact = higher score).

    Models with known inference CO2 are scored inversely. Models without
    CO2 data are assigned a penalty score based on their cost as a proxy
    (higher cost generally implies higher energy consumption).

    Attributes:
        min_co2: Minimum observed inference CO2 (grams per 1M tokens).
        max_co2: Maximum observed inference CO2 (grams per 1M tokens).
        has_co2_data: Whether any model in the registry has CO2 data.
    """

    def __init__(self) -> None:
        """Initialize the normalizer (unfitted)."""
        self.min_co2: float = 0.0
        self.max_co2: float = 1.0
        self.has_co2_data: bool = False
        self._cost_normalizer: CostNormalizer = CostNormalizer()
        self._fitted: bool = False

    def fit(self, profiles: list[ModelProfile]) -> None:
        """Compute statistics from ecological data.

        Falls back to cost-based proxy when CO2 data is unavailable for
        some or all models.

        Args:
            profiles: List of ModelProfile objects.
        """
        co2_values = [
            p.inference_co2_per_1m_grams
            for p in profiles
            if p.inference_co2_per_1m_grams is not None
        ]

        if co2_values:
            self.has_co2_data = True
            self.min_co2 = min(co2_values)
            self.max_co2 = max(co2_values)

        self._cost_normalizer.fit(profiles)
        self._fitted = True

    def transform(self, profile: ModelProfile) -> float:
        """Return normalized ecological score for a model.

        Args:
            profile: The model profile to score.

        Returns:
            Score in [0, 1] where 1.0 = most eco-friendly.

        Raises:
            RuntimeError: If the normalizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("EcoNormalizer must be fit() before transform()")

        if profile.inference_co2_per_1m_grams is not None and self.has_co2_data:
            denom = self.max_co2 - self.min_co2
            if denom == 0:
                return 1.0
            normalized = (profile.inference_co2_per_1m_grams - self.min_co2) / denom
            return float(np.clip(1.0 - normalized, 0.0, 1.0))

        # Fallback: use cost as proxy (cheaper models tend to be more efficient)
        return self._cost_normalizer.transform(profile)


class FeatureNormalizer:
    """Orchestrate all normalizers to produce the full feature vector psi(m).

    Combines normalized benchmark scores, cost score, latency score, and
    ecological score into a single vector representing a model's characteristics.

    The output vector has the following layout:
        [benchmark_0, ..., benchmark_n, cost_score, latency_score, eco_score]

    Attributes:
        benchmark_normalizer: Normalizer for benchmark dimensions.
        cost_normalizer: Normalizer for cost dimension.
        eco_normalizer: Normalizer for ecological dimension.
        max_latency: Maximum latency observed (for normalization).
    """

    def __init__(self) -> None:
        """Initialize all sub-normalizers."""
        self.benchmark_normalizer = BenchmarkNormalizer()
        self.cost_normalizer = CostNormalizer()
        self.eco_normalizer = EcoNormalizer()
        self.max_latency: float = 1000.0
        self._fitted: bool = False

    def fit(self, profiles: list[ModelProfile]) -> None:
        """Fit all sub-normalizers on the model registry.

        Args:
            profiles: List of all ModelProfile objects.
        """
        self.benchmark_normalizer.fit(profiles)
        self.cost_normalizer.fit(profiles)
        self.eco_normalizer.fit(profiles)

        latencies = [
            p.latency_ms for p in profiles if p.latency_ms is not None
        ]
        if latencies:
            self.max_latency = float(max(latencies))

        self._fitted = True

    def transform(self, profile: ModelProfile) -> np.ndarray:
        """Produce the full normalized feature vector psi(m) for a model.

        Args:
            profile: The model profile to vectorize.

        Returns:
            Array of shape (n_benchmarks + 3,) containing:
                - Normalized benchmark scores (one per benchmark)
                - Normalized cost score (inverted)
                - Normalized latency score (inverted)
                - Normalized ecological score (inverted)

        Raises:
            RuntimeError: If the normalizer has not been fitted.
        """
        if not self._fitted:
            raise RuntimeError("FeatureNormalizer must be fit() before transform()")

        bench_vec = self.benchmark_normalizer.transform(profile)
        cost_score = self.cost_normalizer.transform(profile)

        # Latency: invert so lower latency = higher score
        if profile.latency_ms is not None and self.max_latency > 0:
            latency_score = 1.0 - (profile.latency_ms / self.max_latency)
            latency_score = float(np.clip(latency_score, 0.0, 1.0))
        else:
            latency_score = 0.5  # Default for unknown latency

        eco_score = self.eco_normalizer.transform(profile)

        return np.concatenate(
            [bench_vec, np.array([cost_score, latency_score, eco_score])]
        )

    @property
    def feature_names(self) -> list[str]:
        """Return ordered names for each dimension of the feature vector.

        Returns:
            List of feature name strings.
        """
        return [
            *BenchmarkNormalizer.BENCHMARK_NAMES,
            "cost",
            "latency",
            "ecology",
        ]

    @property
    def n_features(self) -> int:
        """Return the total number of features in psi(m).

        Returns:
            Integer count of feature dimensions.
        """
        return len(self.feature_names)
