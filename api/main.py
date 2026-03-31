# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""FastAPI application for QueryRouter++.

description: REST API exposing routing, model listing, health check, and
    explanation endpoints for multi-criteria LLM routing.
agent: coder
date: 2026-03-24
version: 1.0
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from queryrouter.api.openai_compat import router as openai_router
from queryrouter.api.schemas import RoutingRequest, RoutingResponse
from queryrouter.core.router import QueryRouter

app = FastAPI(
    title="QueryRouter++",
    description="Multi-criteria LLM routing API based on formalized query-model compatibility.",
    version="0.2.0",
    license_info={"name": "MIT"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount OpenAI-compatible proxy endpoints (/v1/chat/completions, /v1/models)
app.include_router(openai_router)

# Default data directory relative to workspace
_DATA_DIR = Path(__file__).resolve().parents[1] / "data_models"

# Lazy-initialized router instances
_routers: dict[str, QueryRouter] = {}


def _get_router(strategy: str = "direct") -> QueryRouter:
    """Get or create a cached router instance.

    Args:
        strategy: Routing strategy name.

    Returns:
        QueryRouter instance.
    """
    if strategy not in _routers:
        data_dir = _DATA_DIR if _DATA_DIR.exists() else None
        _routers[strategy] = QueryRouter(strategy=strategy, data_dir=data_dir)  # type: ignore[arg-type]
    return _routers[strategy]


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with status "ok" and version info.
    """
    return {"status": "ok", "version": "0.2.0"}


@app.get("/models")
def list_models() -> dict[str, list[dict]]:
    """List all available models with their profiles.

    Returns:
        Dictionary with "models" key containing list of model profile dicts.
    """
    router = _get_router()
    models = router.registry.get_all()
    return {
        "models": [
            {
                "model_id": m.model_id,
                "name": m.name,
                "provider": m.provider,
                "cost_input_per_1m": m.cost_input_per_1m,
                "cost_output_per_1m": m.cost_output_per_1m,
                "context_window_k": m.context_window_k,
                "latency_ms": m.latency_ms,
                "benchmarks": m.benchmarks,
                "eco_confidence": m.eco_confidence,
            }
            for m in models
        ]
    }


@app.post("/route", response_model=RoutingResponse)
def route_query(request: RoutingRequest) -> RoutingResponse:
    """Route a query to the optimal LLM.

    Args:
        request: RoutingRequest with query and user preferences.

    Returns:
        RoutingResponse with recommended model and score breakdown.

    Raises:
        HTTPException: If routing fails.
    """
    try:
        strategy = request.context.get("strategy", "direct") if request.context else "direct"
        router = _get_router(strategy)
        return router.route(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/explain")
def explain_routing(request: RoutingRequest) -> dict[str, str]:
    """Generate a human-readable explanation of the routing decision.

    Args:
        request: RoutingRequest to explain.

    Returns:
        Dictionary with "explanation" key.

    Raises:
        HTTPException: If explanation generation fails.
    """
    try:
        strategy = request.context.get("strategy", "direct") if request.context else "direct"
        router = _get_router(strategy)
        explanation = router.explain(request)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
