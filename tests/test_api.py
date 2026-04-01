# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""API endpoint tests for QueryRouter++.

description: Tests for FastAPI endpoints using TestClient.
agent: coder
date: 2026-03-24
version: 1.0
"""

import pytest
from fastapi.testclient import TestClient

from queryrouter.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_ok(self) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_returns_version(self) -> None:
        response = client.get("/health")
        assert response.json()["version"] == "0.2.0"


class TestModelsEndpoint:
    """Tests for GET /models."""

    def test_list_models(self) -> None:
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert len(data["models"]) > 0

    def test_model_fields(self) -> None:
        response = client.get("/models")
        model = response.json()["models"][0]
        assert "model_id" in model
        assert "name" in model
        assert "provider" in model
        assert "cost_input_per_1m" in model

    def test_known_model_present(self) -> None:
        response = client.get("/models")
        model_ids = [m["model_id"] for m in response.json()["models"]]
        assert "gpt-4-1" in model_ids


class TestRouteEndpoint:
    """Tests for POST /route."""

    def test_route_basic(self) -> None:
        response = client.post("/route", json={
            "query": "Write a Python function to reverse a string",
            "preferences": {"optimize_for": "balanced"},
        })
        assert response.status_code == 200
        data = response.json()
        assert "recommended_model" in data
        assert "scores" in data
        assert len(data["scores"]) > 0

    def test_route_with_cost_preference(self) -> None:
        response = client.post("/route", json={
            "query": "What is the capital of France?",
            "preferences": {"optimize_for": "cost"},
        })
        assert response.status_code == 200
        data = response.json()
        assert data["recommended_model"] != ""

    def test_route_with_budget_constraint(self) -> None:
        response = client.post("/route", json={
            "query": "Hello",
            "preferences": {
                "optimize_for": "balanced",
                "budget_per_query_usd": 0.01,
            },
        })
        assert response.status_code == 200

    def test_route_invalid_preference(self) -> None:
        response = client.post("/route", json={
            "query": "Hello",
            "preferences": {"optimize_for": "invalid_mode"},
        })
        assert response.status_code == 422

    def test_route_empty_query(self) -> None:
        response = client.post("/route", json={
            "query": "",
            "preferences": {"optimize_for": "balanced"},
        })
        assert response.status_code == 422

    def test_route_scores_sorted(self) -> None:
        response = client.post("/route", json={
            "query": "Explain machine learning",
            "preferences": {"optimize_for": "balanced"},
        })
        scores = response.json()["scores"]
        for i in range(len(scores) - 1):
            assert scores[i]["score"] >= scores[i + 1]["score"]


class TestExplainEndpoint:
    """Tests for POST /explain."""

    def test_explain_basic(self) -> None:
        response = client.post("/explain", json={
            "query": "Write a quicksort in Python",
            "preferences": {"optimize_for": "performance"},
        })
        assert response.status_code == 200
        data = response.json()
        assert "explanation" in data
        assert len(data["explanation"]) > 0

    def test_explain_mentions_model(self) -> None:
        response = client.post("/explain", json={
            "query": "What is 2+2?",
            "preferences": {"optimize_for": "cost"},
        })
        explanation = response.json()["explanation"]
        # Should mention at least one model name
        assert any(
            mid in explanation
            for mid in ["gpt", "claude", "gemini", "deepseek", "llama", "qwen", "mistral"]
        )
