# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""OpenAI-compatible proxy for QueryRouter++.

Exposes /v1/chat/completions and /v1/models so that QueryRouter++ can be
used as a drop-in replacement for any OpenAI-compatible client (LibreChat,
Open WebUI, etc.).

The proxy:
  1. Reads the user's routing preference from the request.
  2. Routes the query to the optimal model via QueryRouter.
  3. Forwards the full chat request to the chosen provider's API.
  4. Streams the response back to the caller.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from queryrouter.api.dependencies import get_router
from queryrouter.api.schemas import RoutingRequest, UserPreferences
from queryrouter.data.loaders import ModelProfile

router = APIRouter(prefix="/v1")

# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------

# Map provider name → (base_url, api_key_env_var)
PROVIDER_CONFIG: dict[str, tuple[str, str]] = {
    "OpenAI": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    # Anthropic does not offer an OpenAI-compatible endpoint natively.
    # Users must set QUERYROUTER_ANTHROPIC_BASE_URL to an OpenAI-compatible
    # proxy (e.g. LiteLLM) or leave empty to disable direct Anthropic routing.
    "Anthropic": ("", "ANTHROPIC_API_KEY"),
    "Google": ("https://generativelanguage.googleapis.com/v1beta/openai", "GOOGLE_API_KEY"),
    "DeepSeek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "Meta": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "Mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "Alibaba": ("https://dashscope-intl.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"),
}

# Override base URLs via env vars (e.g. QUERYROUTER_OPENAI_BASE_URL)
def _get_provider_url(provider: str) -> str:
    env_key = f"QUERYROUTER_{provider.upper()}_BASE_URL"
    default_url, _ = PROVIDER_CONFIG.get(provider, ("", ""))
    return os.getenv(env_key, default_url)


def _get_provider_key(provider: str) -> str:
    _, env_key = PROVIDER_CONFIG.get(provider, ("", ""))
    return os.getenv(env_key, "")


# ---------------------------------------------------------------------------
# Request / response schemas (OpenAI-compatible subset)
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = "queryrouter-auto"
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    # QueryRouter++ custom fields
    routing_preference: str | None = Field(
        default=None,
        description="Routing preference: performance, cost, cost_performance, ecology, balanced",
    )

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_query(messages: list[ChatMessage]) -> str:
    """Extract the routing query from the last user message."""
    for msg in reversed(messages):
        if msg.role == "user" and msg.content:
            if isinstance(msg.content, str):
                return msg.content
            # Vision / multipart: take first text block
            if isinstance(msg.content, list):
                for part in msg.content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        return part["text"]
    return ""


_VALID_PREFERENCES = {"performance", "cost", "cost_performance", "ecology", "balanced"}


def _resolve_preference(request: ChatCompletionRequest) -> str:
    """Resolve routing preference from request fields or model name."""
    # 1. Explicit field
    if request.routing_preference:
        pref = request.routing_preference
        return pref if pref in _VALID_PREFERENCES else "balanced"

    # 2. From extra body / modelKwargs (LibreChat sends custom params here)
    extra = request.model_extra or {}
    if "routing_preference" in extra:
        pref = str(extra["routing_preference"])
        return pref if pref in _VALID_PREFERENCES else "balanced"
    model_kwargs = extra.get("modelKwargs", {})
    if isinstance(model_kwargs, dict) and "routing_preference" in model_kwargs:
        pref = str(model_kwargs["routing_preference"])
        return pref if pref in _VALID_PREFERENCES else "balanced"

    # 3. Encoded in model name: "queryrouter-cost", "queryrouter-ecology", etc.
    model = request.model.lower()
    # Check cost_performance before cost/performance to avoid partial matches
    for pref in ("cost_performance", "performance", "ecology", "cost", "balanced"):
        if pref in model:
            return pref

    return "balanced"


def _build_upstream_body(
    request: ChatCompletionRequest,
    model_id: str,
) -> dict[str, Any]:
    """Build the request body for the upstream provider."""
    body: dict[str, Any] = {
        "model": model_id,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        "stream": request.stream,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    return body


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/models")
def list_models() -> dict:
    """OpenAI-compatible model listing."""
    qr = get_router()
    models = qr.registry.get_all()
    now = int(time.time())
    return {
        "object": "list",
        "data": [
            {
                "id": f"queryrouter-{m.model_id}",
                "object": "model",
                "created": now,
                "owned_by": "queryrouter",
            }
            for m in models
        ] + [
            {"id": "queryrouter-auto", "object": "model", "created": now, "owned_by": "queryrouter"},
            {"id": "queryrouter-performance", "object": "model", "created": now, "owned_by": "queryrouter"},
            {"id": "queryrouter-cost", "object": "model", "created": now, "owned_by": "queryrouter"},
            {"id": "queryrouter-cost_performance", "object": "model", "created": now, "owned_by": "queryrouter"},
            {"id": "queryrouter-ecology", "object": "model", "created": now, "owned_by": "queryrouter"},
            {"id": "queryrouter-balanced", "object": "model", "created": now, "owned_by": "queryrouter"},
        ],
    }


@router.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """OpenAI-compatible chat completions with intelligent routing.

    Accepts standard OpenAI chat format. Routes to the best model based
    on the user's preference, then proxies the request to the provider.
    """
    qr = get_router()

    # --- Route ---
    preference = _resolve_preference(request)
    query_text = _extract_query(request.messages)

    if not query_text:
        raise HTTPException(status_code=400, detail="No user message found")

    routing_req = RoutingRequest(
        query=query_text,
        preferences=UserPreferences(optimize_for=preference),  # type: ignore[arg-type]
    )
    routing_resp = qr.route(routing_req)
    chosen_id = routing_resp.recommended_model

    if chosen_id == "none":
        raise HTTPException(status_code=503, detail="No model satisfies the given constraints")

    # --- Resolve provider ---
    model_profile: ModelProfile = qr.registry.get_by_id(chosen_id)
    provider = model_profile.provider
    base_url = _get_provider_url(provider)
    api_key = _get_provider_key(provider)

    if not base_url or not api_key:
        raise HTTPException(
            status_code=503,
            detail=f"Provider '{provider}' not configured. Set the API key and base URL.",
        )

    # --- Proxy ---
    upstream_body = _build_upstream_body(request, chosen_id)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    url = f"{base_url.rstrip('/')}/chat/completions"

    if request.stream:
        return StreamingResponse(
            _stream_proxy(url, headers, upstream_body, chosen_id, routing_resp.explanation),
            media_type="text/event-stream",
        )

    # Non-streaming
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(url, json=upstream_body, headers=headers)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text)
        data = resp.json()
        # Inject routing metadata
        data["queryrouter"] = {
            "routed_model": chosen_id,
            "preference": preference,
            "explanation": routing_resp.explanation,
            "scores": [
                {"model_id": s.model_id, "score": s.score}
                for s in routing_resp.scores[:3]
            ],
        }
        return data


async def _stream_proxy(
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    chosen_id: str,
    explanation: str,
) -> Any:
    """Stream SSE events from upstream, injecting routing info in the first chunk."""
    first = True
    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, json=body, headers=headers) as resp:
            if resp.status_code != 200:
                error_body = await resp.aread()
                yield f"data: {json.dumps({'error': {'message': error_body.decode(), 'code': resp.status_code}})}\n\n"
                return

            async for line in resp.aiter_lines():
                if not line:
                    yield "\n"
                    continue

                if first and line.startswith("data: ") and line != "data: [DONE]":
                    first = False
                    try:
                        chunk = json.loads(line[6:])
                        chunk["queryrouter"] = {
                            "routed_model": chosen_id,
                            "explanation": explanation,
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        continue
                    except json.JSONDecodeError:
                        pass

                if line.startswith("data:"):
                    yield f"{line}\n\n"
                else:
                    yield f"{line}\n"
