# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Tests for OpenAPI schema generation and validation.

description: Verifies that the OpenAPI schema is valid, complete, and documents
    all required fields for the /route endpoint and score breakdown.
agent: coder
date: 2026-10-05
version: 1.0
"""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from queryrouter.api.main import app

client = TestClient(app)


class TestOpenAPISchema:
    """Tests for OpenAPI 3 schema export and validation."""

    def test_openapi_json_endpoint_exists(self) -> None:
        """Test that /openapi.json endpoint is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

    def test_openapi_json_is_valid(self) -> None:
        """Test that the OpenAPI JSON is valid and has required fields."""
        response = client.get("/openapi.json")
        schema = response.json()

        # Check top-level OpenAPI 3.x structure
        assert schema["openapi"].startswith("3.")
        assert "info" in schema
        assert "paths" in schema
        assert "components" in schema

    def test_openapi_schema_has_info(self) -> None:
        """Test that OpenAPI schema has proper info section."""
        response = client.get("/openapi.json")
        schema = response.json()

        info = schema["info"]
        assert "title" in info
        assert info["title"] == "QueryRouter++"
        assert "description" in info
        assert "version" in info
        assert "license" in info
        assert info["license"]["name"] == "MIT"

    def test_openapi_schema_documents_route_endpoint(self) -> None:
        """Test that /route endpoint is documented in OpenAPI schema."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "/route" in schema["paths"]
        route_spec = schema["paths"]["/route"]
        assert "post" in route_spec

        # Check operation details
        post_spec = route_spec["post"]
        assert "summary" in post_spec
        assert "requestBody" in post_spec
        assert "responses" in post_spec

    def test_openapi_route_request_schema(self) -> None:
        """Test that /route request schema documents all required fields."""
        response = client.get("/openapi.json")
        schema = response.json()

        route_post = schema["paths"]["/route"]["post"]
        request_ref = route_post["requestBody"]["content"]["application/json"]["schema"]["$ref"]

        # Extract schema name from $ref
        schema_name = request_ref.split("/")[-1]
        assert schema_name == "RoutingRequest"

        # Check the schema definition
        routing_request = schema["components"]["schemas"]["RoutingRequest"]
        assert "properties" in routing_request

        # Required fields
        assert "query" in routing_request["properties"]
        assert "preferences" in routing_request["properties"]

        # Optional fields
        assert "tool_context" in routing_request["properties"]
        assert "context" in routing_request["properties"]

    def test_openapi_route_response_schema(self) -> None:
        """Test that /route response schema documents all fields including score breakdown."""
        response = client.get("/openapi.json")
        schema = response.json()

        route_post = schema["paths"]["/route"]["post"]
        response_ref = route_post["responses"]["200"]["content"]["application/json"]["schema"][
            "$ref"
        ]

        # Extract schema name from $ref
        schema_name = response_ref.split("/")[-1]
        assert schema_name == "RoutingResponse"

        # Check the schema definition
        routing_response = schema["components"]["schemas"]["RoutingResponse"]
        assert "properties" in routing_response

        # Check required fields
        assert "recommended_model" in routing_response["properties"]
        assert "scores" in routing_response["properties"]
        assert "explanation" in routing_response["properties"]
        assert "estimated_cost_usd" in routing_response["properties"]
        assert "estimated_latency_ms" in routing_response["properties"]

    def test_openapi_model_score_breakdown(self) -> None:
        """Test that ModelScore schema documents the breakdown structure."""
        response = client.get("/openapi.json")
        schema = response.json()

        # ModelScore should be referenced in components/schemas
        assert "ModelScore" in schema["components"]["schemas"]
        model_score = schema["components"]["schemas"]["ModelScore"]

        assert "properties" in model_score
        assert "model_id" in model_score["properties"]
        assert "score" in model_score["properties"]
        assert "breakdown" in model_score["properties"]

    def test_openapi_user_preferences_schema(self) -> None:
        """Test that UserPreferences schema is fully documented."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "UserPreferences" in schema["components"]["schemas"]
        prefs = schema["components"]["schemas"]["UserPreferences"]

        assert "properties" in prefs
        assert "optimize_for" in prefs["properties"]

        # Check that optimize_for has enum values
        optimize_for = prefs["properties"]["optimize_for"]
        assert "enum" in optimize_for or "anyOf" in optimize_for

    def test_openapi_tool_context_schema(self) -> None:
        """Test that ToolContext schema documents all tool flags."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "ToolContext" in schema["components"]["schemas"]
        tool_context = schema["components"]["schemas"]["ToolContext"]

        assert "properties" in tool_context
        # Check for key tool flags
        assert "has_web_search" in tool_context["properties"]
        assert "has_code_exec" in tool_context["properties"]
        assert "has_mcp" in tool_context["properties"]

    def test_static_openapi_json_exists(self) -> None:
        """Test that static OpenAPI JSON file exists in docs/."""
        repo_root = Path(__file__).parent.parent
        openapi_json = repo_root / "docs" / "openapi.json"

        assert openapi_json.exists(), "docs/openapi.json should exist"

        # Verify it's valid JSON
        with open(openapi_json) as f:
            schema = json.load(f)

        assert schema["openapi"].startswith("3.")

    def test_static_openapi_yaml_exists(self) -> None:
        """Test that static OpenAPI YAML file exists in docs/."""
        repo_root = Path(__file__).parent.parent
        openapi_yaml = repo_root / "docs" / "openapi.yaml"

        assert openapi_yaml.exists(), "docs/openapi.yaml should exist"

    def test_static_schema_matches_live_schema(self) -> None:
        """Test that static OpenAPI JSON matches the live schema from the app."""
        repo_root = Path(__file__).parent.parent
        openapi_json = repo_root / "docs" / "openapi.json"

        with open(openapi_json) as f:
            static_schema = json.load(f)

        live_schema = app.openapi()

        # Compare key structures (allowing for minor differences like ordering)
        assert static_schema["openapi"] == live_schema["openapi"]
        assert static_schema["info"] == live_schema["info"]
        assert set(static_schema["paths"].keys()) == set(live_schema["paths"].keys())

        # Check that /route endpoint spec is identical
        assert static_schema["paths"]["/route"] == live_schema["paths"]["/route"]

    def test_openapi_health_endpoint_documented(self) -> None:
        """Test that /health endpoint is documented."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "/health" in schema["paths"]
        assert "get" in schema["paths"]["/health"]

    def test_openapi_models_endpoint_documented(self) -> None:
        """Test that /models endpoint is documented."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "/models" in schema["paths"]
        assert "get" in schema["paths"]["/models"]

    def test_openapi_explain_endpoint_documented(self) -> None:
        """Test that /explain endpoint is documented."""
        response = client.get("/openapi.json")
        schema = response.json()

        assert "/explain" in schema["paths"]
        assert "post" in schema["paths"]["/explain"]
