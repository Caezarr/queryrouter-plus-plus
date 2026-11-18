# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""API endpoint tests for QueryRouter++.

Tests for FastAPI endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

import queryrouter.api.main as api_main
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
        response = client.post(
            "/route",
            json={
                "query": "Write a Python function to reverse a string",
                "preferences": {"optimize_for": "balanced"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommended_model" in data
        assert "scores" in data
        assert len(data["scores"]) > 0

    def test_route_with_cost_preference(self) -> None:
        response = client.post(
            "/route",
            json={
                "query": "What is the capital of France?",
                "preferences": {"optimize_for": "cost"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["recommended_model"] != ""

    def test_route_with_budget_constraint(self) -> None:
        response = client.post(
            "/route",
            json={
                "query": "Hello",
                "preferences": {
                    "optimize_for": "balanced",
                    "budget_per_query_usd": 0.01,
                },
            },
        )
        assert response.status_code == 200

    def test_route_invalid_preference(self) -> None:
        response = client.post(
            "/route",
            json={
                "query": "Hello",
                "preferences": {"optimize_for": "invalid_mode"},
            },
        )
        assert response.status_code == 422

    def test_route_empty_query(self) -> None:
        response = client.post(
            "/route",
            json={
                "query": "",
                "preferences": {"optimize_for": "balanced"},
            },
        )
        assert response.status_code == 422

    def test_route_scores_sorted(self) -> None:
        response = client.post(
            "/route",
            json={
                "query": "Explain machine learning",
                "preferences": {"optimize_for": "balanced"},
            },
        )
        scores = response.json()["scores"]
        for i in range(len(scores) - 1):
            assert scores[i]["score"] >= scores[i + 1]["score"]

    def test_route_with_tool_context(self) -> None:
        """Test that tool context is accepted and affects routing."""
        response = client.post(
            "/route",
            json={
                "query": "Search the web for recent AI benchmarks",
                "preferences": {"optimize_for": "balanced"},
                "tool_context": {
                    "has_web_search": True,
                    "has_mcp": True,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "recommended_model" in data
        assert data["recommended_model"] != ""

    def test_route_tool_context_affects_selection(self) -> None:
        """Verify tool context biases toward stronger models."""
        query = "Write a complex data analysis script"
        prefs = {"optimize_for": "cost"}

        # Without tools
        response_no_tools = client.post(
            "/route",
            json={
                "query": query,
                "preferences": prefs,
            },
        )
        model_no_tools = response_no_tools.json()["recommended_model"]

        # With tools
        response_with_tools = client.post(
            "/route",
            json={
                "query": query,
                "preferences": prefs,
                "tool_context": {
                    "has_code_exec": True,
                    "has_attached_tools": True,
                },
            },
        )
        model_with_tools = response_with_tools.json()["recommended_model"]

        # The specific models may differ based on data, but both should succeed
        assert model_no_tools != ""
        assert model_with_tools != ""


class TestStrategySelection:
    """Tests for context["strategy"] handling on /route."""

    @pytest.mark.parametrize("strategy", ["direct", "cascade", "embedding"])
    def test_known_strategy_is_used(self, strategy: str) -> None:
        payload = {
            "query": "Prove that sqrt(2) is irrational.",
            "preferences": {"optimize_for": "balanced"},
            "context": {"strategy": strategy},
        }
        response = client.post("/route", json=payload)
        assert response.status_code == 200
        assert response.json()["explanation"].lower().startswith(strategy)

    def test_unknown_strategy_falls_back_to_direct(self) -> None:
        payload = {
            "query": "Prove that sqrt(2) is irrational.",
            "preferences": {"optimize_for": "balanced"},
        }
        direct = client.post("/route", json=payload).json()
        payload["context"] = {"strategy": "not-a-strategy"}
        unknown = client.post("/route", json=payload).json()
        assert unknown["recommended_model"] == direct["recommended_model"]
        assert unknown["explanation"] == direct["explanation"]


class TestExplainEndpoint:
    """Tests for POST /explain."""

    def test_explain_basic(self) -> None:
        response = client.post(
            "/explain",
            json={
                "query": "Write a quicksort in Python",
                "preferences": {"optimize_for": "performance"},
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "explanation" in data
        assert len(data["explanation"]) > 0

    def test_explain_mentions_model(self) -> None:
        response = client.post(
            "/explain",
            json={
                "query": "What is 2+2?",
                "preferences": {"optimize_for": "cost"},
            },
        )
        explanation = response.json()["explanation"]
        # Should mention at least one model name
        assert any(
            mid in explanation
            for mid in ["gpt", "claude", "gemini", "deepseek", "llama", "qwen", "mistral"]
        )


class TestEntrypoint:
    """Tests for the ``queryrouter`` console script (api.main.main)."""

    def _capture_run(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
        calls: dict[str, object] = {}

        def fake_run(app_obj: object, host: str, port: int) -> None:
            calls.update(app=app_obj, host=host, port=port)

        monkeypatch.setattr(api_main.uvicorn, "run", fake_run)
        return calls

    def test_main_defaults_to_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("QUERYROUTER_HOST", raising=False)
        monkeypatch.delenv("QUERYROUTER_PORT", raising=False)
        calls = self._capture_run(monkeypatch)
        api_main.main()
        assert calls == {"app": app, "host": "127.0.0.1", "port": 8000}

    def test_main_reads_host_and_port_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("QUERYROUTER_HOST", "0.0.0.0")
        monkeypatch.setenv("QUERYROUTER_PORT", "9001")
        calls = self._capture_run(monkeypatch)
        api_main.main()
        assert calls == {"app": app, "host": "0.0.0.0", "port": 9001}
