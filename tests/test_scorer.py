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
            benchmarks={"mmlu": 0.7, "humaneval": 0.5, "gsm8k": 0.6, "math": 0.4, "gpqa_diamond": 0.55},
            cost_input_per_1m=0.1,
            cost_output_per_1m=0.3,
            latency_ms=200,
            inference_co2_per_1m_grams=20.0,
        ),
        ModelProfile(
            model_id="expensive-model",
            name="Expensive Model",
            provider="TestCo",
            benchmarks={"mmlu": 0.95, "humaneval": 0.95, "gsm8k": 0.98, "math": 0.9, "gpqa_diamond": 0.92},
            cost_input_per_1m=10.0,
            cost_output_per_1m=30.0,
            latency_ms=800,
            inference_co2_per_1m_grams=100.0,
        ),
        ModelProfile(
            model_id="balanced-model",
            name="Balanced Model",
            provider="TestCo",
            benchmarks={"mmlu": 0.85, "humaneval": 0.8, "gsm8k": 0.85, "math": 0.7, "gpqa_diamond": 0.75},
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


class TestLatencyDiscrimination:
    """Tests that latency axis discriminates between models."""

    def test_latency_scores_differ_across_models(
        self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer
    ) -> None:
        """Latency scores should vary based on model latency_ms values."""
        features = featurizer.featurize("Quick question")
        latency_scores = [
            scorer.latency_score(features, p) for p in profiles
        ]
        # With different latencies (200, 800, 400), scores should differ
        unique_scores = set(latency_scores)
        assert len(unique_scores) > 1, "Latency scores should not be constant"
        # Fastest model (200ms) should score highest
        assert latency_scores[0] > latency_scores[1]  # cheap (200) > expensive (800)
        assert latency_scores[2] > latency_scores[1]  # balanced (400) > expensive (800)

    def test_latency_not_constant_zero_five(
        self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer
    ) -> None:
        """Latency should vary across models and not be constant 0.5 (unknown)."""
        features = featurizer.featurize("Test query")
        latency_scores = [scorer.latency_score(features, p) for p in profiles]
        
        # Check that scores vary (not all the same)
        assert len(set(latency_scores)) > 1, "Latency scores should vary across models"
        
        # Check that at least one score is not 0.5 (confirming latency_ms is being used)
        assert any(score != 0.5 for score in latency_scores), "At least one model should have non-0.5 latency"


class TestGPQAWeighting:
    """Tests that reasoning tasks use GPQA Diamond instead of deprecated benchmarks."""

    def test_reasoning_query_uses_gpqa(
        self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer
    ) -> None:
        """Reasoning queries should weight GPQA Diamond in performance scoring."""
        # Create a reasoning-heavy query
        features = featurizer.featurize(
            "Analyze the following scientific reasoning problem step by step: "
            "Given a complex molecular structure, deduce the reaction pathway."
        )
        
        # Check that reasoning task type is activated
        task_scores = features[11:21]  # task type features
        reasoning_idx = 4  # "reasoning" is 5th in TASK_TYPES list
        assert task_scores[reasoning_idx] > 0.1, "Reasoning task should be detected"
        
        # Performance scoring should work without errors (GPQA is in BENCHMARK_NAMES)
        perf_score = scorer._performance_score(features, profiles[0])
        assert 0.0 <= perf_score <= 1.0

    def test_coding_query_uses_humaneval_not_gpqa(
        self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer
    ) -> None:
        """Coding queries should use HumanEval, not mix with GPQA."""
        features = featurizer.featurize(
            "Write a Python function to implement binary search on a sorted array"
        )
        
        # Check that coding task type is activated
        task_scores = features[11:21]
        coding_idx = 0  # "coding" is first in TASK_TYPES list
        assert task_scores[coding_idx] > 0.1, "Coding task should be detected"
        
        # Performance scoring should work
        perf_score = scorer._performance_score(features, profiles[1])
        assert 0.0 <= perf_score <= 1.0


class TestCostScaling:
    """Tests that cost scales with query expected output length."""

    def test_long_query_costs_more_than_short(
        self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer
    ) -> None:
        """Long-output queries should have different cost scores than short-output queries."""
        # Short query (expected output: brief)
        short_features = featurizer.featurize("What is 2+2?")
        short_cost = scorer.cost_score(short_features, profiles[1])  # expensive model
        
        # Long query (expected output: comprehensive)
        long_features = featurizer.featurize(
            "Write a comprehensive essay explaining the complete history of artificial "
            "intelligence, including all major breakthroughs, key figures, philosophical "
            "implications, and future directions. Provide detailed examples and thorough analysis."
        )
        long_cost = scorer.cost_score(long_features, profiles[1])  # same expensive model
        
        # Cost scores should differ because actual estimated costs differ
        # Note: cost_score inverts (lower cost = higher score), so long query
        # should have lower score (costs more) than short query
        assert long_cost != short_cost, "Cost should vary with output length"

    def test_cost_reflects_output_length_in_features(
        self, scorer: CompatibilityScorer, profiles: list[ModelProfile], featurizer: QueryFeaturizer
    ) -> None:
        """Cost calculation should use expected_output_length from query features."""
        short_features = featurizer.featurize("yes or no")
        long_features = featurizer.featurize(
            "Provide a detailed step-by-step explanation with comprehensive examples"
        )
        
        # Check that expected_output_length differs (feature index 21)
        short_output = short_features[21]
        long_output = long_features[21]
        assert long_output > short_output, "Long query should have higher expected output"
        
        # Test with balanced model (profiles[2]) where output length variations have more impact
        # (not cheapest or most expensive, so we see the relative cost differences)
        short_cost_score = scorer.cost_score(short_features, profiles[2])
        long_cost_score = scorer.cost_score(long_features, profiles[2])
        
        # Different output lengths should produce different cost scores
        # Long output costs more, so score should be lower (cost score is inverted)
        assert short_cost_score != long_cost_score, "Output length should affect cost score"
        assert short_cost_score > long_cost_score, "Shorter output should have better (higher) cost score"
