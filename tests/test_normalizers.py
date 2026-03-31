# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Unit tests for feature normalizers."""

from __future__ import annotations

import numpy as np
import pytest

from queryrouter.data.loaders import ModelProfile
from queryrouter.data.normalizers import (
    BenchmarkNormalizer,
    CostNormalizer,
    EcoNormalizer,
    FeatureNormalizer,
)


def _profiles() -> list[ModelProfile]:
    return [
        ModelProfile(
            model_id="cheap",
            name="Cheap",
            provider="T",
            benchmarks={"mmlu": 0.7, "humaneval": 0.5, "gsm8k": 0.6, "math": 0.4, "hellaswag": 0.7, "arc": 0.6},
            cost_input_per_1m=0.1,
            cost_output_per_1m=0.3,
            latency_ms=200,
            inference_co2_per_1m_grams=10.0,
        ),
        ModelProfile(
            model_id="expensive",
            name="Expensive",
            provider="T",
            benchmarks={"mmlu": 0.95, "humaneval": 0.95, "gsm8k": 0.98, "math": 0.9, "hellaswag": 0.96, "arc": 0.95},
            cost_input_per_1m=10.0,
            cost_output_per_1m=30.0,
            latency_ms=800,
            inference_co2_per_1m_grams=100.0,
        ),
    ]


class TestBenchmarkNormalizer:
    def test_transform_shape(self) -> None:
        bn = BenchmarkNormalizer()
        bn.fit(_profiles())
        result = bn.transform(_profiles()[0])
        assert result.shape == (6,)

    def test_transform_range(self) -> None:
        bn = BenchmarkNormalizer()
        bn.fit(_profiles())
        for p in _profiles():
            result = bn.transform(p)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_min_maps_to_zero(self) -> None:
        bn = BenchmarkNormalizer()
        bn.fit(_profiles())
        cheap = bn.transform(_profiles()[0])
        assert np.all(cheap <= 1.0)
        # Cheapest benchmarks should map to 0.0
        assert cheap[0] == pytest.approx(0.0)  # mmlu: 0.7 is min

    def test_max_maps_to_one(self) -> None:
        bn = BenchmarkNormalizer()
        bn.fit(_profiles())
        expensive = bn.transform(_profiles()[1])
        assert expensive[0] == pytest.approx(1.0)  # mmlu: 0.95 is max

    def test_missing_benchmark_imputed(self) -> None:
        profiles = _profiles()
        sparse = ModelProfile(
            model_id="sparse",
            name="Sparse",
            provider="T",
            benchmarks={"mmlu": 0.8},
        )
        profiles.append(sparse)
        bn = BenchmarkNormalizer()
        bn.fit(profiles)
        result = bn.transform(sparse)
        assert result.shape == (6,)
        assert np.all(np.isfinite(result))

    def test_unfitted_raises(self) -> None:
        bn = BenchmarkNormalizer()
        with pytest.raises(RuntimeError):
            bn.transform(_profiles()[0])


class TestCostNormalizer:
    def test_cheapest_scores_one(self) -> None:
        cn = CostNormalizer()
        cn.fit(_profiles())
        assert cn.transform(_profiles()[0]) == pytest.approx(1.0)

    def test_most_expensive_scores_zero(self) -> None:
        cn = CostNormalizer()
        cn.fit(_profiles())
        assert cn.transform(_profiles()[1]) == pytest.approx(0.0)

    def test_single_model_scores_one(self) -> None:
        cn = CostNormalizer()
        cn.fit([_profiles()[0]])
        assert cn.transform(_profiles()[0]) == pytest.approx(1.0)

    def test_unfitted_raises(self) -> None:
        cn = CostNormalizer()
        with pytest.raises(RuntimeError):
            cn.transform(_profiles()[0])


class TestEcoNormalizer:
    def test_lowest_co2_scores_one(self) -> None:
        en = EcoNormalizer()
        en.fit(_profiles())
        assert en.transform(_profiles()[0]) == pytest.approx(1.0)

    def test_highest_co2_scores_zero(self) -> None:
        en = EcoNormalizer()
        en.fit(_profiles())
        assert en.transform(_profiles()[1]) == pytest.approx(0.0)

    def test_missing_co2_falls_back_to_cost(self) -> None:
        profiles = _profiles()
        no_co2 = ModelProfile(
            model_id="no-co2",
            name="No CO2",
            provider="T",
            cost_input_per_1m=5.0,
            cost_output_per_1m=15.0,
        )
        profiles.append(no_co2)
        en = EcoNormalizer()
        en.fit(profiles)
        score = en.transform(no_co2)
        assert 0.0 <= score <= 1.0

    def test_unfitted_raises(self) -> None:
        en = EcoNormalizer()
        with pytest.raises(RuntimeError):
            en.transform(_profiles()[0])


class TestFeatureNormalizer:
    def test_transform_shape(self) -> None:
        fn = FeatureNormalizer()
        fn.fit(_profiles())
        result = fn.transform(_profiles()[0])
        assert result.shape == (fn.n_features,)

    def test_transform_range(self) -> None:
        fn = FeatureNormalizer()
        fn.fit(_profiles())
        for p in _profiles():
            result = fn.transform(p)
            assert np.all(result >= 0.0)
            assert np.all(result <= 1.0)

    def test_feature_names_length(self) -> None:
        fn = FeatureNormalizer()
        fn.fit(_profiles())
        assert len(fn.feature_names) == fn.n_features

    def test_max_latency_computed(self) -> None:
        fn = FeatureNormalizer()
        fn.fit(_profiles())
        assert fn.max_latency == 800.0

    def test_unfitted_raises(self) -> None:
        fn = FeatureNormalizer()
        with pytest.raises(RuntimeError):
            fn.transform(_profiles()[0])
