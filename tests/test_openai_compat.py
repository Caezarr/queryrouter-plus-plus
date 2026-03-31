# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Tests for the OpenAI-compatible proxy endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from queryrouter.api.main import app
from queryrouter.api.openai_compat import (
    ChatCompletionRequest,
    ChatMessage,
    _extract_query,
    _resolve_preference,
)

client = TestClient(app)


class TestListModels:
    def test_returns_models(self) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) > 0

    def test_includes_preset_models(self) -> None:
        response = client.get("/v1/models")
        ids = [m["id"] for m in response.json()["data"]]
        assert "queryrouter-auto" in ids
        assert "queryrouter-performance" in ids
        assert "queryrouter-cost" in ids
        assert "queryrouter-ecology" in ids

    def test_includes_real_models(self) -> None:
        response = client.get("/v1/models")
        ids = [m["id"] for m in response.json()["data"]]
        # Should include at least one real model wrapped as queryrouter-{id}
        assert any(mid.startswith("queryrouter-") and mid != "queryrouter-auto" for mid in ids)


class TestExtractQuery:
    def test_last_user_message(self) -> None:
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="First question"),
            ChatMessage(role="assistant", content="Answer"),
            ChatMessage(role="user", content="Second question"),
        ]
        assert _extract_query(messages) == "Second question"

    def test_no_user_message(self) -> None:
        messages = [ChatMessage(role="system", content="System prompt")]
        assert _extract_query(messages) == ""

    def test_multipart_content(self) -> None:
        messages = [
            ChatMessage(
                role="user",
                content=[
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                    {"type": "text", "text": "What is this?"},
                ],
            ),
        ]
        assert _extract_query(messages) == "What is this?"


class TestResolvePreference:
    def test_explicit_field(self) -> None:
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            routing_preference="cost",
        )
        assert _resolve_preference(req) == "cost"

    def test_from_model_name(self) -> None:
        req = ChatCompletionRequest(
            model="queryrouter-ecology",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert _resolve_preference(req) == "ecology"

    def test_cost_performance_from_model(self) -> None:
        req = ChatCompletionRequest(
            model="queryrouter-cost_performance",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert _resolve_preference(req) == "cost_performance"

    def test_defaults_to_balanced(self) -> None:
        req = ChatCompletionRequest(
            model="queryrouter-auto",
            messages=[ChatMessage(role="user", content="hi")],
        )
        assert _resolve_preference(req) == "balanced"


class TestChatCompletionsRouting:
    """Test that the endpoint routes correctly (without hitting upstream)."""

    def test_no_provider_returns_503(self) -> None:
        """Without API keys configured, proxy should fail with 503."""
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "queryrouter-auto",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
        assert response.status_code == 503

    def test_empty_messages_returns_400(self) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "queryrouter-auto",
                "messages": [{"role": "system", "content": "System only"}],
            },
        )
        assert response.status_code == 400
