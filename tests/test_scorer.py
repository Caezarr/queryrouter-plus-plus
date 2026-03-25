# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Unit tests for CompatibilityScorer.

description: Tests for the compatibility scoring function
    S(q, m, w) = sum_i w_i * f_i(phi(q), psi(m)).
agent: coder
date: 2026-03-24
version: 1.0
"""

import numpy as np
import pytest

from queryrouter.core.compatibility_scorer import (
    CompatibilityScorer,
    ModelScore,
    WeightVector,
)
from queryrouter.core.query_featurizer import QueryFeaturizer
from queryrouter.data.loaders import ModelProfile
from queryrouter.data.normalizers import FeatureNormalizer


def _make_profiles() -> list[ModelProfile]:
    """Create test model profiles."""
    return [
        ModelProfile(
            model_id="cheap-model",
            name="Cheap Model",
            provider="TestCo",
            benchmarks={"mmlu": 0.7, "humaneval": 0.5, "gsm8k": 0.6, "math": 0.4, "hellaswag": 0.7, "arc": 0.6},
            cost_input_per_1m=0.1,
            cost_output_per_1m=0.3,
            latency_ms=200,
            inference_co2_per_1m_grams=20.0,
        ),
        ModelProfile(
            model_id="expensive-model",
            name="Expensive Model",
            provider="TestCo",
            benchmarks={"mmlu": 0.95, "humaneval": 0.95, "gsm8k": 0.98, "math": 0.9, "hellaswag": 0.96, "arc": 0.95},
            cost_input_per_1m=10.0,
            cost_output_per_1m=30.0,
            latency_ms=800,
            inference_co2_per_1m_grams=100.0,
        ),
        ModelProfile(
            model_id="balanced-model",
            name="Balanced Model",
            provider="TestCo",
            benchmarks={"mmlu": 0.85, "humaneval": 0.8, "gsm8k": 0.85, "math": 0.7, "hellaswag": 0.85, "arc": 0.8},
            cost_input_per_1m=2.0,
            cost_output_per_1m=6.0,
            latency_ms=400,
            inference_co2_per_1m_grams=50.0,
        ),
    ]


@pytest.fixture
def profiles() -> list[ModelProfile]:
    return _make_profiles()


@pytest.fixture
def scorer(profiles: list[ModelProfile]) -> CompatibilityScorer:
    normalizer = FeatureNormalizer()
    normalizer.fit(profiles)
    return CompatibilityScorer(normalizer)


@pytest.fixture
def featurizer() -> QueryFeaturizer:
    return QueryFeaturizer()


class TestScore:
    """Tests for score() method."""

    def test_score_in_range(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Write a Python function")
        weights = WeightVector(0.25, 0.25, 0.25, 0.25)
        for profile in profiles:
            result = scorer.score(features, profile, weights)
            assert 0.0 <= result.score <= 1.0

    def test_performance_weights_favor_expensive(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Solve a complex math problem step by step")
        perf_weights = WeightVector(1.0, 0.0, 0.0, 0.0)
        scores = {p.model_id: scorer.score(features, p, perf_weights).score for p in profiles}
        assert scores["expensive-model"] >= scores["cheap-model"]

    def test_cost_weights_favor_cheap(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("What is 2+2?")
        cost_weights = WeightVector(0.0, 1.0, 0.0, 0.0)
        scores = {p.model_id: scorer.score(features, p, cost_weights).score for p in profiles}
        assert scores["cheap-model"] >= scores["expensive-model"]

    def test_breakdown_present(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Hello")
        weights = WeightVector(0.25, 0.25, 0.25, 0.25)
        result = scorer.score(features, profiles[0], weights)
        assert "performance" in result.breakdown
        assert "cost" in result.breakdown
        assert "latency" in result.breakdown
        assert "ecology" in result.breakdown

    def test_breakdown_values_in_range(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Explain quantum physics")
        weights = WeightVector(0.25, 0.25, 0.25, 0.25)
        for profile in profiles:
            result = scorer.score(features, profile, weights)
            for axis, value in result.breakdown.items():
                assert 0.0 <= value <= 1.0, f"{axis} out of range for {profile.model_id}"

    def test_weighted_sum_consistency(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Test query")
        weights = WeightVector(0.4, 0.3, 0.2, 0.1)
        result = scorer.score(features, profiles[0], weights)
        bd = result.breakdown
        expected = (
            0.4 * bd["performance"]
            + 0.3 * bd["cost"]
            + 0.2 * bd["latency"]
            + 0.1 * bd["ecology"]
        )
        assert abs(result.score - expected) < 1e-6


class TestScoreAll:
    """Tests for score_all() method."""

    def test_returns_sorted(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Write code")
        weights = WeightVector(0.25, 0.25, 0.25, 0.25)
        results = scorer.score_all(features, profiles, weights)
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score

    def test_all_models_scored(self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Anything")
        weights = WeightVector(0.25, 0.25, 0.25, 0.25)
        results = scorer.score_all(features, profiles, weights)
        assert len(results) == len(profiles)

    def test_empty_models(self, scorer: CompatibilityScorer, featurizer: QueryFeaturizer) -> None:
        features = featurizer.featurize("Hello")
        weights = WeightVector(0.25, 0.25, 0.25, 0.25)
        results = scorer.score_all(features, [], weights)
        assert len(results) == 0


class TestWeightVector:
    """Tests for WeightVector."""

    def test_as_array(self) -> None:
        w = WeightVector(0.4, 0.3, 0.2, 0.1)
        arr = w.as_array()
        np.testing.assert_array_almost_equal(arr, [0.4, 0.3, 0.2, 0.1])

    def test_from_dict(self) -> None:
        d = {"w_performance": 0.5, "w_cost": 0.2, "w_latency": 0.2, "w_ecology": 0.1}
        w = WeightVector.from_dict(d)
        assert w.w_performance == 0.5
        assert w.w_cost == 0.2

    def test_from_dict_defaults(self) -> None:
        w = WeightVector.from_dict({})
        assert w.w_performance == 0.25
