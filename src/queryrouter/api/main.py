# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""FastAPI application for QueryRouter++.

REST API exposing routing, model listing, health check, and
explanation endpoints for multi-criteria LLM routing.
"""

from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from queryrouter import __version__
from queryrouter.api.dependencies import get_router
from queryrouter.api.openai_compat import router as openai_router
from queryrouter.api.schemas import RoutingRequest, RoutingResponse
from queryrouter.core.router import STRATEGIES, Strategy

app = FastAPI(
    title="QueryRouter++",
    description="Multi-criteria LLM routing API based on formalized query-model compatibility.",
    version=__version__,
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

# Router factory is shared via api.dependencies to avoid duplication.


def _requested_strategy(request: RoutingRequest) -> Strategy:
    """Read ``context["strategy"]``; unknown values fall back to ``direct``.

    Unknown names already routed like ``direct``. Normalizing them here also
    keeps user input from growing the per-strategy router cache without bound.
    """
    requested = (request.context or {}).get("strategy", "direct")
    for strategy in STRATEGIES:
        if requested == strategy:
            return strategy
    return "direct"


@app.get("/health")
def health_check() -> dict[str, str]:
    """Health check endpoint.

    Returns:
        Dictionary with status "ok" and version info.
    """
    return {"status": "ok", "version": __version__}


@app.get("/models")
def list_models() -> dict[str, list[dict[str, Any]]]:
    """List all available models with their profiles.

    Returns:
        Dictionary with "models" key containing list of model profile dicts.
    """
    router = get_router()
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
        router = get_router(_requested_strategy(request))
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
        router = get_router(_requested_strategy(request))
        explanation = router.explain(request)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


def main() -> None:
    """Run the API with uvicorn (entry point of the ``queryrouter`` script).

    Host and port are read from ``QUERYROUTER_HOST`` and ``QUERYROUTER_PORT``
    (see ``.env.example``), defaulting to ``127.0.0.1:8000``.
    """
    host = os.getenv("QUERYROUTER_HOST", "127.0.0.1")
    port = int(os.getenv("QUERYROUTER_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
