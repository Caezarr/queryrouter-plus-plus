# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Pydantic v2 request/response schemas for the QueryRouter++ API.

description: Defines UserPreferences, RoutingRequest, ModelScore, and
    RoutingResponse models with full validation.
agent: coder
date: 2026-03-23
version: 1.0
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


# -- Preset weight vectors for optimize_for modes --

PRESET_WEIGHTS: dict[str, dict[str, float]] = {
    "performance": {
        "w_performance": 0.7,
        "w_cost": 0.1,
        "w_latency": 0.1,
        "w_ecology": 0.1,
    },
    "cost": {
        "w_performance": 0.1,
        "w_cost": 0.7,
        "w_latency": 0.1,
        "w_ecology": 0.1,
    },
    "cost_performance": {
        "w_performance": 0.4,
        "w_cost": 0.4,
        "w_latency": 0.1,
        "w_ecology": 0.1,
    },
    "ecology": {
        "w_performance": 0.1,
        "w_cost": 0.1,
        "w_latency": 0.1,
        "w_ecology": 0.7,
    },
    "balanced": {
        "w_performance": 0.25,
        "w_cost": 0.25,
        "w_latency": 0.25,
        "w_ecology": 0.25,
    },
}

WEIGHT_KEYS: list[str] = ["w_performance", "w_cost", "w_latency", "w_ecology"]


class UserPreferences(BaseModel):
    """User preferences controlling multi-criteria LLM routing.

    Defines the optimization objective (preset or custom weights), hard
    constraints (budget, latency), and model filtering rules.

    When ``optimize_for`` is set to a preset value (e.g. ``"performance"``),
    the corresponding weight vector is auto-assigned. Use ``"custom"`` and
    provide explicit ``weights`` for fine-grained control.

    Attributes:
        optimize_for: Optimization preset or "custom" for manual weights.
        weights: Weight vector {w_performance, w_cost, w_latency, w_ecology}.
            Auto-populated for presets; required for "custom".
        budget_per_query_usd: Max cost in USD per query (hard filter). None = no limit.
        max_latency_ms: Max latency in ms (hard filter). None = no limit.
        eco_mode: Bias toward ecologically efficient models.
        allowed_models: Whitelist of model_ids. None = all eligible.
        excluded_models: Blacklist of model_ids. None = no exclusions.

    Example:
        >>> prefs = UserPreferences(optimize_for="cost", budget_per_query_usd=0.01)
        >>> prefs.weights
        {'w_performance': 0.1, 'w_cost': 0.7, 'w_latency': 0.1, 'w_ecology': 0.1}
    """

    optimize_for: Literal[
        "performance", "cost", "cost_performance", "ecology", "balanced", "custom"
    ]
    weights: Optional[dict[str, float]] = Field(
        default=None,
        description="Weight vector {w_performance, w_cost, w_latency, w_ecology}. Sum must equal 1.",
    )
    budget_per_query_usd: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum USD cost per query. Models exceeding this are excluded.",
    )
    max_latency_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description="Maximum acceptable inference latency in milliseconds.",
    )
    eco_mode: bool = Field(
        default=False,
        description="When true, applies a bonus to eco-friendly models.",
    )
    allowed_models: Optional[list[str]] = Field(
        default=None,
        description="If set, only these model_ids are considered.",
    )
    excluded_models: Optional[list[str]] = Field(
        default=None,
        description="Model_ids to exclude from routing.",
    )

    @model_validator(mode="after")
    def resolve_weights(self) -> "UserPreferences":
        """Resolve weights from preset or validate custom weights.

        For preset modes, auto-assigns the corresponding weight vector.
        For custom mode, validates that weights are provided, contain the
        correct keys, and sum to 1.0.

        Returns:
            The validated UserPreferences instance.

        Raises:
            ValueError: If custom weights are missing, have wrong keys, or
                don't sum to 1.
        """
        if self.optimize_for == "custom":
            if self.weights is None:
                raise ValueError("weights are required when optimize_for='custom'")
            missing = set(WEIGHT_KEYS) - set(self.weights.keys())
            if missing:
                raise ValueError(f"Missing weight keys: {missing}")
            weight_sum = sum(self.weights[k] for k in WEIGHT_KEYS)
            if abs(weight_sum - 1.0) > 1e-6:
                raise ValueError(
                    f"Weights must sum to 1.0, got {weight_sum:.6f}"
                )
        else:
            self.weights = PRESET_WEIGHTS[self.optimize_for].copy()
        return self

    def get_weight(self, axis: str) -> float:
        """Get the weight for a specific optimization axis.

        Args:
            axis: One of "performance", "cost", "latency", "ecology".

        Returns:
            The weight value for that axis.

        Raises:
            KeyError: If the axis is not recognized.
        """
        key = f"w_{axis}"
        if self.weights is None or key not in self.weights:
            raise KeyError(f"Unknown weight axis: {axis}")
        return self.weights[key]


class RoutingRequest(BaseModel):
    """Request body for the /route endpoint.

    Attributes:
        query: The user's raw query string to route.
        preferences: User preferences controlling model selection.
        context: Optional metadata (e.g. conversation history, domain hints).
    """

    query: str = Field(
        ...,
        min_length=1,
        description="The query to route to the optimal LLM.",
    )
    preferences: UserPreferences = Field(
        ...,
        description="User preferences for routing optimization.",
    )
    context: Optional[dict[str, str]] = Field(
        default=None,
        description="Optional context metadata (conversation_id, domain, etc.).",
    )


class ModelScore(BaseModel):
    """Score breakdown for a single model in a routing decision.

    Attributes:
        model_id: Unique model identifier.
        score: Overall composite score S(q, m, w) in [0, 1].
        breakdown: Per-axis scores {performance, cost, latency, ecology}.
    """

    model_id: str = Field(..., description="Unique model identifier.")
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Composite score in [0, 1].",
    )
    breakdown: dict[str, float] = Field(
        default_factory=dict,
        description="Per-axis score breakdown.",
    )


class RoutingResponse(BaseModel):
    """Response body for the /route endpoint.

    Attributes:
        recommended_model: The model_id selected by the router.
        scores: Scored list of all eligible models, sorted descending.
        explanation: Human-readable explanation of the routing decision.
        estimated_cost_usd: Estimated cost in USD for this query on the selected model.
        estimated_latency_ms: Estimated latency in ms for the selected model.
    """

    recommended_model: str = Field(
        ...,
        description="The model_id of the recommended LLM.",
    )
    scores: list[ModelScore] = Field(
        default_factory=list,
        description="All evaluated models with scores, sorted by score descending.",
    )
    explanation: str = Field(
        default="",
        description="Human-readable explanation of the routing decision.",
    )
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD for this query.",
    )
    estimated_latency_ms: int = Field(
        default=0,
        ge=0,
        description="Estimated inference latency in milliseconds.",
    )
