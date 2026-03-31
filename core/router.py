# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Main routing engine for QueryRouter++.

description: QueryRouter class implementing direct, cascade, and embedding-based
    routing strategies for multi-criteria LLM selection.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from queryrouter.api.schemas import (
    ModelScore as APIModelScore,
    RoutingRequest,
    RoutingResponse,
    UserPreferences,
)
from queryrouter.core.compatibility_scorer import (
    CompatibilityScorer,
    ModelScore,
    WeightVector,
)
from queryrouter.core.model_registry import ModelRegistry
from queryrouter.core.preference_engine import PreferenceEngine
from queryrouter.core.query_featurizer import QueryFeaturizer
from queryrouter.data.loaders import ModelProfile
from queryrouter.data.normalizers import FeatureNormalizer
from queryrouter.data.utils import estimate_query_cost


class QueryRouter:
    """Main entry point for multi-criteria LLM routing.

    Supports three routing strategies:
        - direct: Score all models, pick the best.
        - cascade: Try cheaper models first, escalate if confidence is low.
        - embedding: Use cosine similarity in a shared latent space.

    Args:
        strategy: Routing strategy to use.
        data_dir: Path to the data directory with CSV matrices.
        cascade_threshold: Confidence threshold for cascade strategy.

    Example:
        >>> router = QueryRouter(strategy="direct", data_dir=Path("workspace/data"))
        >>> request = RoutingRequest(
        ...     query="Write a Python quicksort",
        ...     preferences=UserPreferences(optimize_for="cost_performance"),
        ... )
        >>> response = router.route(request)
        >>> response.recommended_model
        'deepseek-v3'
    """

    def __init__(
        self,
        strategy: Literal["direct", "cascade", "embedding"] = "direct",
        data_dir: Path | None = None,
        cascade_threshold: float = 0.6,
    ) -> None:
        """Initialize the router with all components.

        Args:
            strategy: One of "direct", "cascade", "embedding".
            data_dir: Path to data directory. If None, uses default workspace path.
            cascade_threshold: Minimum score threshold for cascade strategy
                before escalating to the next model tier.
        """
        self.strategy = strategy
        self.cascade_threshold = cascade_threshold

        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[2] / "data_models"

        self.registry = ModelRegistry(data_dir)
        self.featurizer = QueryFeaturizer()
        self.preference_engine = PreferenceEngine()

        # Fit normalizer on all models
        all_models = self.registry.get_all()
        self.normalizer = FeatureNormalizer()
        self.normalizer.fit(all_models)

        self.scorer = CompatibilityScorer(self.normalizer)

        # Pre-compute model embeddings for embedding strategy
        self._model_embeddings: dict[str, np.ndarray] | None = None
        if strategy == "embedding":
            self._init_model_embeddings(all_models)

    def route(self, request: RoutingRequest) -> RoutingResponse:
        """Route a query to the optimal model based on strategy and preferences.

        Args:
            request: RoutingRequest with query string and user preferences.

        Returns:
            RoutingResponse with recommended model and score breakdown.
        """
        preferences = request.preferences
        weights = self.preference_engine.resolve(preferences)
        query_features = self.featurizer.featurize(request.query)

        # Get eligible models after filtering
        models = self.registry.get_allowed(
            allowed=preferences.allowed_models,
            excluded=preferences.excluded_models,
        )
        models = self.preference_engine.filter_models(models, preferences)

        if not models:
            return RoutingResponse(
                recommended_model="none",
                scores=[],
                explanation="No models satisfy the given constraints.",
                estimated_cost_usd=0.0,
                estimated_latency_ms=0,
            )

        if self.strategy == "cascade":
            return self._route_cascade(query_features, models, weights)
        if self.strategy == "embedding":
            return self._route_embedding(query_features, models, weights)
        return self._route_direct(query_features, models, weights)

    def explain(self, request: RoutingRequest) -> str:
        """Generate a human-readable explanation of the routing decision.

        Args:
            request: RoutingRequest to explain.

        Returns:
            Multi-line explanation string.
        """
        response = self.route(request)
        lines = [
            f"Routing Strategy: {self.strategy}",
            f"Recommended Model: {response.recommended_model}",
            f"Overall Score: {response.scores[0].score:.3f}" if response.scores else "",
            "",
            "Score Breakdown:",
        ]

        if response.scores:
            top = response.scores[0]
            for axis, value in top.breakdown.items():
                weight_key = f"w_{axis}"
                w = request.preferences.weights.get(weight_key, 0.0) if request.preferences.weights else 0.0
                lines.append(f"  {axis}: {value:.3f} (weight={w:.2f})")

        lines.extend([
            "",
            f"Estimated Cost: ${response.estimated_cost_usd:.6f}",
            f"Estimated Latency: {response.estimated_latency_ms}ms",
            "",
            "All Models Ranked:",
        ])
        for i, ms in enumerate(response.scores[:5], 1):
            lines.append(f"  {i}. {ms.model_id}: {ms.score:.3f}")

        return "\n".join(lines)

    def _route_direct(
        self,
        query_features: np.ndarray,
        models: list[ModelProfile],
        weights: WeightVector,
    ) -> RoutingResponse:
        """Direct routing: score all models, return the best.

        Args:
            query_features: Feature vector phi(q).
            models: Eligible model profiles.
            weights: Resolved weight vector.

        Returns:
            RoutingResponse.
        """
        scored = self.scorer.score_all(query_features, models, weights)
        return self._build_response(scored, models)

    def _route_cascade(
        self,
        query_features: np.ndarray,
        models: list[ModelProfile],
        weights: WeightVector,
    ) -> RoutingResponse:
        """Cascade routing: try cheaper models first, escalate if score is low.

        Models are sorted by cost (cheapest first). If the compatibility
        score exceeds the cascade threshold, that model is selected.
        Otherwise, escalate to the next more expensive model.

        Args:
            query_features: Feature vector phi(q).
            models: Eligible model profiles.
            weights: Resolved weight vector.

        Returns:
            RoutingResponse.
        """
        # Sort by total cost ascending
        sorted_models = sorted(
            models,
            key=lambda m: m.cost_input_per_1m + m.cost_output_per_1m,
        )

        all_scored = []
        selected_score = None

        for model in sorted_models:
            ms = self.scorer.score(query_features, model, weights)
            all_scored.append(ms)

            if ms.score >= self.cascade_threshold and selected_score is None:
                selected_score = ms

        # If no model passed threshold, use the last (most expensive)
        if selected_score is None:
            selected_score = all_scored[-1]

        # Sort all scores descending for the response
        all_scored.sort(key=lambda x: x.score, reverse=True)

        # Put the selected model first
        api_scores = self._convert_scores(all_scored)
        best_model = next(
            (m for m in sorted_models if m.model_id == selected_score.model_id),
            sorted_models[-1],
        )

        return RoutingResponse(
            recommended_model=selected_score.model_id,
            scores=api_scores,
            explanation=f"Cascade routing selected {selected_score.model_id} "
            f"(score={selected_score.score:.3f}, threshold={self.cascade_threshold})",
            estimated_cost_usd=estimate_query_cost(best_model),
            estimated_latency_ms=best_model.latency_ms or 0,
        )

    def _route_embedding(
        self,
        query_features: np.ndarray,
        models: list[ModelProfile],
        weights: WeightVector,
    ) -> RoutingResponse:
        """Embedding-based routing using cosine similarity.

        Uses the query feature vector as a proxy embedding and computes
        cosine similarity against model profile vectors. The performance
        axis is replaced by cosine similarity while cost, latency, and
        ecology axes remain as in direct routing.

        Args:
            query_features: Feature vector phi(q).
            models: Eligible model profiles.
            weights: Resolved weight vector.

        Returns:
            RoutingResponse.
        """
        # Use query features as embedding proxy
        query_emb = query_features / (np.linalg.norm(query_features) + 1e-10)

        scored = []
        w = weights.as_array()
        for model in models:
            # Use precomputed embeddings when available
            if self._model_embeddings and model.model_id in self._model_embeddings:
                model_emb = self._model_embeddings[model.model_id]
            else:
                model_vec = self.normalizer.transform(model)
                model_emb = model_vec / (np.linalg.norm(model_vec) + 1e-10)

            # Cosine similarity for performance axis
            cos_sim = float(np.dot(query_emb[:len(model_emb)], model_emb))
            cos_sim = max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))  # map [-1,1] to [0,1]

            # Other axes from standard scorer
            cost_s = self.scorer.cost_score(query_features, model)
            lat_s = self.scorer.latency_score(query_features, model)
            eco_s = self.scorer.ecology_score(query_features, model)

            total = float(
                w[0] * cos_sim + w[1] * cost_s + w[2] * lat_s + w[3] * eco_s
            )

            scored.append(ModelScore(
                model_id=model.model_id,
                score=total,
                breakdown={
                    "performance": cos_sim,
                    "cost": cost_s,
                    "latency": lat_s,
                    "ecology": eco_s,
                },
            ))

        scored.sort(key=lambda x: x.score, reverse=True)
        return self._build_response(scored, models)

    def _init_model_embeddings(self, models: list[ModelProfile]) -> None:
        """Pre-compute normalized model profile vectors.

        Args:
            models: List of all model profiles.
        """
        self._model_embeddings = {}
        for m in models:
            vec = self.normalizer.transform(m)
            norm = np.linalg.norm(vec)
            self._model_embeddings[m.model_id] = vec / (norm + 1e-10)

    def _build_response(
        self,
        scored: list[ModelScore],
        models: list[ModelProfile],
    ) -> RoutingResponse:
        """Build a RoutingResponse from scored results.

        Args:
            scored: List of ModelScore objects, sorted by score descending.
            models: List of eligible model profiles.

        Returns:
            RoutingResponse.
        """
        if not scored:
            return RoutingResponse(
                recommended_model="none",
                scores=[],
                explanation="No models could be scored.",
            )

        best_id = scored[0].model_id
        best_model = next((m for m in models if m.model_id == best_id), None)

        api_scores = self._convert_scores(scored)

        return RoutingResponse(
            recommended_model=best_id,
            scores=api_scores,
            explanation=f"{self.strategy.capitalize()} routing selected {best_id} "
            f"with score {scored[0].score:.3f}",
            estimated_cost_usd=estimate_query_cost(best_model) if best_model else 0.0,
            estimated_latency_ms=best_model.latency_ms or 0 if best_model else 0,
        )

    def _convert_scores(self, scored: list) -> list[APIModelScore]:
        """Convert internal ModelScore objects to API schema.

        Args:
            scored: List of internal ModelScore dataclass instances.

        Returns:
            List of API ModelScore Pydantic models.
        """
        return [
            APIModelScore(
                model_id=ms.model_id,
                score=round(ms.score, 4),
                breakdown=ms.breakdown,
            )
            for ms in scored
        ]

