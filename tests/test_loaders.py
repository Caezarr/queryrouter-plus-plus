# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Unit tests for data loaders."""

from __future__ import annotations

from pathlib import Path

import pytest

from queryrouter.data.loaders import (
    ModelRegistry,
    _parse_float,
    _parse_int,
)

# -- Helpers --


def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    with open(path, "w") as f:
        f.write(header + "\n")
        for row in rows:
            f.write(row + "\n")


def _make_data_dir(tmp_path: Path) -> Path:
    """Create a minimal three-CSV data directory."""
    _write_csv(
        tmp_path / "models_benchmark_matrix.csv",
        "model_id,model_name,provider,mmlu_score,humaneval_score,gsm8k_score,math_score,hellaswag_score,arc_score,chatbot_arena_elo",
        [
            "model-a,Model A,ProvA,0.9,0.8,0.85,0.7,0.88,0.82,1400",
            "model-b,Model B,ProvB,0.7,0.6,0.65,0.5,0.68,0.62,1200",
        ],
    )
    _write_csv(
        tmp_path / "models_cost_matrix.csv",
        "model_id,model_name,provider,input_price_per_1m_tokens_usd,output_price_per_1m_tokens_usd,context_window_k,avg_latency_ms",
        [
            "model-a,Model A,ProvA,10.0,30.0,128,500",
            "model-b,Model B,ProvB,0.5,1.5,32,200",
        ],
    )
    _write_csv(
        tmp_path / "models_eco_matrix.csv",
        "model_id,model_name,provider,estimated_params_billions,training_co2_tons,inference_co2_per_1m_tokens_grams,hardware_type,confidence_level",
        [
            "model-a,Model A,ProvA,175,500,100.0,H100,HIGH",
            "model-b,Model B,ProvB,7,20,10.0,A100,MEDIUM",
        ],
    )
    return tmp_path


# -- Tests for parse helpers --


class TestParseFloat:
    def test_valid(self) -> None:
        assert _parse_float("3.14") == pytest.approx(3.14)

    def test_integer_string(self) -> None:
        assert _parse_float("42") == pytest.approx(42.0)

    def test_empty(self) -> None:
        assert _parse_float("") is None

    def test_na(self) -> None:
        assert _parse_float("N/A") is None

    def test_whitespace(self) -> None:
        assert _parse_float("  2.5  ") == pytest.approx(2.5)

    def test_nan(self) -> None:
        assert _parse_float("nan") is None

    def test_invalid(self) -> None:
        assert _parse_float("not_a_number") is None


class TestParseInt:
    def test_valid(self) -> None:
        assert _parse_int("42") == 42

    def test_float_string(self) -> None:
        assert _parse_int("3.9") == 3

    def test_empty(self) -> None:
        assert _parse_int("") is None


# -- Tests for ModelRegistry --


class TestModelRegistry:
    def test_loads_all_models(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        assert len(registry.models) == 2

    def test_benchmark_data_merged(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        profile = registry.get("model-a")
        assert profile.benchmarks["mmlu"] == pytest.approx(0.9)
        assert profile.benchmarks["humaneval"] == pytest.approx(0.8)

    def test_cost_data_merged(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        profile = registry.get("model-a")
        assert profile.cost_input_per_1m == pytest.approx(10.0)
        assert profile.cost_output_per_1m == pytest.approx(30.0)
        assert profile.context_window_k == 128
        assert profile.latency_ms == 500

    def test_eco_data_merged(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        profile = registry.get("model-a")
        assert profile.params_billions == pytest.approx(175.0)
        assert profile.inference_co2_per_1m_grams == pytest.approx(100.0)
        assert profile.eco_confidence == "HIGH"

    def test_get_missing_raises(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        with pytest.raises(KeyError, match="not-real"):
            registry.get("not-real")

    def test_list_model_ids(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        ids = registry.list_model_ids()
        assert ids == ["model-a", "model-b"]

    def test_list_providers(self, tmp_path: Path) -> None:
        data_dir = _make_data_dir(tmp_path)
        registry = ModelRegistry(data_dir)
        providers = registry.list_providers()
        assert "ProvA" in providers
        assert "ProvB" in providers

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            ModelRegistry(tmp_path)

    def test_comment_lines_skipped(self, tmp_path: Path) -> None:
        with open(tmp_path / "models_benchmark_matrix.csv", "w") as f:
            f.write("# this is a comment\n")
            f.write("# another comment\n")
            f.write(
                "model_id,model_name,provider,mmlu_score,humaneval_score,gsm8k_score,math_score,hellaswag_score,arc_score,chatbot_arena_elo\n"
            )
            f.write("m1,M1,P1,0.8,0.7,0.6,0.5,0.4,0.3,1300\n")
        _write_csv(
            tmp_path / "models_cost_matrix.csv",
            "model_id,model_name,provider,input_price_per_1m_tokens_usd,output_price_per_1m_tokens_usd,context_window_k,avg_latency_ms",
            ["m1,M1,P1,1.0,2.0,128,100"],
        )
        _write_csv(
            tmp_path / "models_eco_matrix.csv",
            "model_id,model_name,provider,estimated_params_billions,training_co2_tons,inference_co2_per_1m_tokens_grams,hardware_type,confidence_level",
            ["m1,M1,P1,10,50,20,H100,HIGH"],
        )
        registry = ModelRegistry(tmp_path)
        assert "m1" in registry.models

    def test_na_benchmark_treated_as_missing(self, tmp_path: Path) -> None:
        _write_csv(
            tmp_path / "models_benchmark_matrix.csv",
            "model_id,model_name,provider,mmlu_score,humaneval_score,gsm8k_score,math_score,hellaswag_score,arc_score,chatbot_arena_elo",
            ["m1,M1,P1,0.8,N/A,N/A,N/A,N/A,N/A,"],
        )
        _write_csv(
            tmp_path / "models_cost_matrix.csv",
            "model_id,model_name,provider,input_price_per_1m_tokens_usd,output_price_per_1m_tokens_usd,context_window_k,avg_latency_ms",
            ["m1,M1,P1,1.0,2.0,128,100"],
        )
        _write_csv(
            tmp_path / "models_eco_matrix.csv",
            "model_id,model_name,provider,estimated_params_billions,training_co2_tons,inference_co2_per_1m_tokens_grams,hardware_type,confidence_level",
            ["m1,M1,P1,,,,,LOW"],
        )
        registry = ModelRegistry(tmp_path)
        profile = registry.get("m1")
        assert "mmlu" in profile.benchmarks
        assert "humaneval" not in profile.benchmarks


class TestProductionEcoData:
    """Tests that verify production ecological data meets requirements for issue #13."""

    def test_all_models_have_co2_data(self) -> None:
        """Verify all 12 models have inference CO2 estimates (issue #13)."""
        data_dir = Path(__file__).parent.parent / "data_models"
        registry = ModelRegistry(data_dir)
        
        # All models should have CO2 data
        models_without_co2 = []
        for model_id in registry.list_model_ids():
            profile = registry.get(model_id)
            if profile.inference_co2_per_1m_grams is None:
                models_without_co2.append(model_id)
        
        assert len(models_without_co2) == 0, (
            f"Models without CO2 data: {models_without_co2}. "
            "All models must have inference_co2_per_1m_grams for eco axis to work."
        )

    def test_eco_axis_differentiates_models(self) -> None:
        """Verify eco scores differentiate models across the pool (issue #13)."""
        from queryrouter.data.normalizers import EcoNormalizer
        
        data_dir = Path(__file__).parent.parent / "data_models"
        registry = ModelRegistry(data_dir)
        profiles = [registry.get(mid) for mid in registry.list_model_ids()]
        
        # Fit the eco normalizer
        en = EcoNormalizer()
        en.fit(profiles)
        
        # Get all eco scores
        eco_scores = {p.model_id: en.transform(p) for p in profiles}
        
        # Should have at least 12 models
        assert len(eco_scores) >= 12
        
        # Scores should span a meaningful range (not all the same)
        unique_scores = len(set(eco_scores.values()))
        assert unique_scores >= 5, (
            f"Eco axis only produces {unique_scores} unique scores across "
            f"{len(eco_scores)} models. Need more differentiation."
        )
        
        # Min and max should be well separated (at least 0.5 range)
        score_range = max(eco_scores.values()) - min(eco_scores.values())
        assert score_range >= 0.5, (
            f"Eco score range is only {score_range:.2f}. "
            f"Need at least 0.5 for meaningful differentiation."
        )
        
        # Verify some expected relationships based on CO2 values
        # Gemini Flash (1.1g) should score highest
        # o3 (65.0g) should score lowest or near-lowest
        gemini_flash = eco_scores.get("gemini-2-5-flash")
        o3_score = eco_scores.get("o3")
        
        if gemini_flash is not None and o3_score is not None:
            assert gemini_flash > o3_score, (
                "Gemini Flash (1.1g CO2) should score higher than o3 (65.0g CO2)"
            )
