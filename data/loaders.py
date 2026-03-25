# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Data loaders for QueryRouter++ model registry and query taxonomy.

description: Load benchmark, cost, and ecological CSV matrices plus query taxonomy JSON
    into unified ModelProfile dataclasses and structured taxonomy objects.
agent: coder
date: 2026-03-23
version: 1.0
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelProfile:
    """Unified profile for a single LLM, combining benchmark, cost, and eco data.

    Represents psi(m) in the formal framework -- the complete characterization
    of a model's capabilities, pricing, and environmental footprint.

    Attributes:
        model_id: Unique identifier (e.g. "gpt-4o", "claude-3-5-sonnet").
        name: Human-readable model name with version info.
        provider: Organization providing the model (e.g. "OpenAI", "Anthropic").
        benchmarks: Mapping of benchmark name to score in [0, 1].
        cost_input_per_1m: Price in USD per 1M input tokens.
        cost_output_per_1m: Price in USD per 1M output tokens.
        context_window_k: Maximum context window in thousands of tokens.
        latency_ms: Average inference latency in milliseconds (None if unknown).
        params_billions: Number of parameters in billions (None if undisclosed).
        training_co2_tons: Estimated training CO2 emissions in metric tons (None if unknown).
        inference_co2_per_1m_grams: CO2 grams per 1M tokens at inference (None if unknown).
        hardware_type: Training/inference hardware (e.g. "NVIDIA H100").
        eco_confidence: Confidence level for ecological data ("HIGH", "MEDIUM", "LOW").
        chatbot_arena_elo: Chatbot Arena Elo rating (None if not available).
    """

    model_id: str
    name: str
    provider: str
    benchmarks: dict[str, float] = field(default_factory=dict)
    cost_input_per_1m: float = 0.0
    cost_output_per_1m: float = 0.0
    context_window_k: int = 0
    latency_ms: Optional[int] = None
    params_billions: Optional[float] = None
    training_co2_tons: Optional[float] = None
    inference_co2_per_1m_grams: Optional[float] = None
    hardware_type: str = ""
    eco_confidence: str = "LOW"
    chatbot_arena_elo: Optional[int] = None


def _parse_float(value: str) -> Optional[float]:
    """Parse a CSV cell as float, returning None for empty or N/A values.

    Args:
        value: Raw string from CSV cell.

    Returns:
        Parsed float or None if the value is empty or non-numeric.
    """
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        result = float(value)
        return result if not math.isnan(result) else None
    except ValueError:
        return None


def _parse_int(value: str) -> Optional[int]:
    """Parse a CSV cell as integer, returning None for empty or N/A values.

    Args:
        value: Raw string from CSV cell.

    Returns:
        Parsed int or None if the value is empty or non-numeric.
    """
    f = _parse_float(value)
    return int(f) if f is not None else None


class ModelRegistry:
    """Load and merge benchmark, cost, and ecological CSVs into ModelProfile objects.

    The registry reads three CSV files produced by the data-collector agent and
    merges them on ``model_id`` into a unified collection of :class:`ModelProfile`
    instances.

    Attributes:
        models: Mapping of model_id to its ModelProfile.

    Example:
        >>> registry = ModelRegistry(data_dir=Path("workspace/data"))
        >>> profile = registry.get("gpt-4o")
        >>> profile.cost_input_per_1m
        2.5
    """

    BENCHMARK_COLUMNS: list[str] = [
        "mmlu_score",
        "humaneval_score",
        "gsm8k_score",
        "math_score",
        "hellaswag_score",
        "arc_score",
    ]

    def __init__(self, data_dir: Path) -> None:
        """Initialize the registry by loading and merging all data files.

        Args:
            data_dir: Path to the directory containing the CSV data files.

        Raises:
            FileNotFoundError: If any required CSV file is missing.
        """
        self.data_dir = data_dir
        self.models: dict[str, ModelProfile] = {}
        self._load_benchmarks()
        self._load_costs()
        self._load_eco()

    def _read_csv(self, filename: str) -> list[dict[str, str]]:
        """Read a CSV file, skipping comment lines starting with '#'.

        Args:
            filename: Name of the CSV file within data_dir.

        Returns:
            List of row dictionaries keyed by column header.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        filepath = self.data_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Required data file not found: {filepath}")

        rows: list[dict[str, str]] = []
        with open(filepath, encoding="utf-8") as f:
            # Skip comment lines at the top
            lines = [line for line in f if not line.startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            rows.append(row)
        return rows

    def _load_benchmarks(self) -> None:
        """Load models_benchmark_matrix.csv and populate model profiles."""
        rows = self._read_csv("models_benchmark_matrix.csv")
        for row in rows:
            model_id = row["model_id"].strip()
            benchmarks: dict[str, float] = {}
            for col in self.BENCHMARK_COLUMNS:
                val = _parse_float(row.get(col, ""))
                if val is not None:
                    benchmarks[col.replace("_score", "")] = val

            elo = _parse_int(row.get("chatbot_arena_elo", ""))

            self.models[model_id] = ModelProfile(
                model_id=model_id,
                name=row.get("model_name", "").strip(),
                provider=row.get("provider", "").strip(),
                benchmarks=benchmarks,
                chatbot_arena_elo=elo,
            )

    def _load_costs(self) -> None:
        """Load models_cost_matrix.csv and merge cost data into existing profiles."""
        rows = self._read_csv("models_cost_matrix.csv")
        for row in rows:
            model_id = row["model_id"].strip()
            profile = self.models.get(model_id)
            if profile is None:
                continue
            profile.cost_input_per_1m = _parse_float(
                row.get("input_price_per_1m_tokens_usd", "")
            ) or 0.0
            profile.cost_output_per_1m = _parse_float(
                row.get("output_price_per_1m_tokens_usd", "")
            ) or 0.0
            profile.context_window_k = _parse_int(row.get("context_window_k", "")) or 0
            profile.latency_ms = _parse_int(row.get("avg_latency_ms", ""))

    def _load_eco(self) -> None:
        """Load models_eco_matrix.csv and merge ecological data into existing profiles."""
        rows = self._read_csv("models_eco_matrix.csv")
        for row in rows:
            model_id = row["model_id"].strip()
            profile = self.models.get(model_id)
            if profile is None:
                continue
            profile.params_billions = _parse_float(row.get("estimated_params_billions", ""))
            profile.training_co2_tons = _parse_float(row.get("training_co2_tons", ""))
            profile.inference_co2_per_1m_grams = _parse_float(
                row.get("inference_co2_per_1m_tokens_grams", "")
            )
            profile.hardware_type = row.get("hardware_type", "").strip()
            profile.eco_confidence = row.get("confidence_level", "LOW").strip()

    def get(self, model_id: str) -> ModelProfile:
        """Retrieve a model profile by its identifier.

        Args:
            model_id: The unique model identifier.

        Returns:
            The corresponding ModelProfile.

        Raises:
            KeyError: If the model_id is not in the registry.
        """
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self.models[model_id]

    def list_model_ids(self) -> list[str]:
        """Return all registered model identifiers.

        Returns:
            Sorted list of model_id strings.
        """
        return sorted(self.models.keys())

    def list_providers(self) -> list[str]:
        """Return distinct provider names.

        Returns:
            Sorted list of unique provider strings.
        """
        return sorted({p.provider for p in self.models.values()})


@dataclass
class QueryCategory:
    """A single category from the query taxonomy.

    Attributes:
        name: Category identifier (e.g. "coding", "math_reasoning").
        description: Human-readable description of the category.
        examples: Example queries belonging to this category.
        complexity_level: Expected complexity ("simple", "medium", "complex").
        model_requirements: Minimum model capabilities for this category.
        benchmark_coverage: Relevant benchmarks for evaluating this category.
    """

    name: str
    description: str
    examples: list[str] = field(default_factory=list)
    complexity_level: str = "medium"
    model_requirements: dict[str, str] = field(default_factory=dict)
    benchmark_coverage: list[str] = field(default_factory=list)


@dataclass
class TaxonomyDimension:
    """A classification dimension from the query taxonomy.

    Attributes:
        name: Dimension identifier (e.g. "task_type", "complexity").
        description: What this dimension captures.
        values: Allowed values for this dimension.
    """

    name: str
    description: str
    values: list[str] = field(default_factory=list)


class QueryTaxonomyLoader:
    """Load the query taxonomy JSON into structured category and dimension objects.

    The taxonomy defines the space of query types that QueryRouter++ can
    classify and route.

    Attributes:
        categories: List of query categories.
        dimensions: List of taxonomy dimensions.

    Example:
        >>> taxonomy = QueryTaxonomyLoader(Path("workspace/data/query_taxonomy.json"))
        >>> taxonomy.get_category("coding").complexity_level
        'medium-complex'
    """

    def __init__(self, taxonomy_path: Path) -> None:
        """Initialize by loading and parsing the taxonomy JSON.

        Args:
            taxonomy_path: Path to query_taxonomy.json.

        Raises:
            FileNotFoundError: If the file does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        if not taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

        with open(taxonomy_path, encoding="utf-8") as f:
            data = json.load(f)

        self.categories: list[QueryCategory] = []
        for cat in data.get("categories", []):
            self.categories.append(
                QueryCategory(
                    name=cat["name"],
                    description=cat.get("description", ""),
                    examples=cat.get("examples", []),
                    complexity_level=cat.get("complexity_level", "medium"),
                    model_requirements=cat.get("typical_model_requirements", {}),
                    benchmark_coverage=cat.get("benchmark_coverage", []),
                )
            )

        self.dimensions: list[TaxonomyDimension] = []
        for dim_name, dim_data in data.get("dimensions", {}).items():
            self.dimensions.append(
                TaxonomyDimension(
                    name=dim_name,
                    description=dim_data.get("description", ""),
                    values=dim_data.get("values", []),
                )
            )

    def get_category(self, name: str) -> QueryCategory:
        """Retrieve a category by name.

        Args:
            name: The category name (e.g. "coding").

        Returns:
            The matching QueryCategory.

        Raises:
            KeyError: If no category with that name exists.
        """
        for cat in self.categories:
            if cat.name == name:
                return cat
        raise KeyError(f"Category '{name}' not found in taxonomy")

    def list_category_names(self) -> list[str]:
        """Return all category names.

        Returns:
            List of category name strings.
        """
        return [cat.name for cat in self.categories]

    def list_dimension_names(self) -> list[str]:
        """Return all dimension names.

        Returns:
            List of dimension name strings.
        """
        return [dim.name for dim in self.dimensions]
