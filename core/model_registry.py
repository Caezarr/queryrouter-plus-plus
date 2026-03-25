# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Model registry for QueryRouter++.

description: High-level ModelRegistry that wraps data loaders and provides
    filtering, lookup, and listing operations on ModelProfile objects.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

from pathlib import Path

from queryrouter.data.loaders import ModelProfile, ModelRegistry as _BaseRegistry


class ModelRegistry:
    """Registry of LLM model profiles with filtering and lookup.

    Wraps the data-layer ModelRegistry to provide a clean API for the
    routing engine. Loads all three CSV matrices on initialization and
    merges them into unified ModelProfile objects.

    Attributes:
        data_dir: Path to the data directory containing CSV files.

    Example:
        >>> registry = ModelRegistry(Path("workspace/data"))
        >>> all_models = registry.get_all()
        >>> gpt = registry.get_by_id("gpt-4o")
    """

    def __init__(self, data_dir: Path) -> None:
        """Initialize the registry by loading data from CSV files.

        Args:
            data_dir: Path to the directory containing
                models_benchmark_matrix.csv, models_cost_matrix.csv,
                and models_eco_matrix.csv.
        """
        self._base = _BaseRegistry(data_dir)

    def get_all(self) -> list[ModelProfile]:
        """Return all registered model profiles.

        Returns:
            List of all ModelProfile objects, sorted by model_id.
        """
        return [self._base.models[mid] for mid in sorted(self._base.models)]

    def get_by_id(self, model_id: str) -> ModelProfile:
        """Retrieve a single model profile by its identifier.

        Args:
            model_id: The unique model identifier (e.g. "gpt-4o").

        Returns:
            The corresponding ModelProfile.

        Raises:
            KeyError: If the model_id is not found.
        """
        return self._base.get(model_id)

    def get_allowed(
        self,
        allowed: list[str] | None = None,
        excluded: list[str] | None = None,
    ) -> list[ModelProfile]:
        """Return models filtered by whitelist and blacklist.

        Args:
            allowed: If set, only return models whose model_id is in this list.
                None means all models are eligible.
            excluded: If set, remove models whose model_id is in this list.
                None means no exclusions.

        Returns:
            Filtered list of ModelProfile objects.
        """
        models = self.get_all()

        if allowed is not None:
            allowed_set = set(allowed)
            models = [m for m in models if m.model_id in allowed_set]

        if excluded is not None:
            excluded_set = set(excluded)
            models = [m for m in models if m.model_id not in excluded_set]

        return models

    def list_ids(self) -> list[str]:
        """Return all registered model identifiers.

        Returns:
            Sorted list of model_id strings.
        """
        return self._base.list_model_ids()

    def list_providers(self) -> list[str]:
        """Return distinct provider names.

        Returns:
            Sorted list of unique provider strings.
        """
        return self._base.list_providers()
