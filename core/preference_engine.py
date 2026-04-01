# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Preference resolution engine for QueryRouter++.

description: Resolves UserPreferences into concrete WeightVector and applies
    hard constraints (budget, latency) to filter eligible models.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

from queryrouter.api.schemas import UserPreferences
from queryrouter.core.compatibility_scorer import WeightVector
from queryrouter.data.loaders import ModelProfile
from queryrouter.data.utils import estimate_query_cost


class PreferenceEngine:
    """Resolve user preferences into weight vectors and apply hard constraints.

    Handles both preset optimization modes (performance, cost, etc.) and
    custom weight vectors. Also applies budget and latency hard constraints
    to filter the model pool before scoring.

    Example:
        >>> engine = PreferenceEngine()
        >>> prefs = UserPreferences(optimize_for="cost", budget_per_query_usd=0.01)
        >>> weights = engine.resolve(prefs)
        >>> weights.w_cost
        0.7
    """

    def resolve(self, preferences: UserPreferences) -> WeightVector:
        """Resolve preferences into a concrete weight vector.

        For preset modes, returns the corresponding predefined weights.
        For custom mode, validates and returns the user-provided weights.

        Args:
            preferences: UserPreferences with resolved weights dict.

        Returns:
            WeightVector with w_performance, w_cost, w_latency, w_ecology.

        Raises:
            ValueError: If weights are not available (should not happen
                if UserPreferences validation passed).
        """
        if preferences.weights is None:
            raise ValueError("Preferences must have resolved weights")

        return WeightVector.from_dict(preferences.weights)

    def filter_models(
        self,
        models: list[ModelProfile],
        preferences: UserPreferences,
        estimated_tokens: int = 1000,
    ) -> list[ModelProfile]:
        """Apply hard constraints to filter the model pool.

        Removes models that violate budget or latency constraints.

        Args:
            models: List of candidate model profiles.
            preferences: UserPreferences with optional hard constraints.
            estimated_tokens: Estimated total tokens (input + output) for
                cost calculation. Defaults to 1000.

        Returns:
            Filtered list of ModelProfile objects that satisfy all constraints.
        """
        filtered = models

        # Budget constraint
        if preferences.budget_per_query_usd is not None:
            budget = preferences.budget_per_query_usd
            filtered = [
                m for m in filtered
                if estimate_query_cost(m, estimated_tokens) <= budget
            ]

        # Latency constraint
        if preferences.max_latency_ms is not None:
            max_lat = preferences.max_latency_ms
            filtered = [
                m for m in filtered
                if m.latency_ms is None or m.latency_ms <= max_lat
            ]

        return filtered

