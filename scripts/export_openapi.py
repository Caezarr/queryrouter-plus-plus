#!/usr/bin/env python3
# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Export OpenAPI 3 schema from the FastAPI application.

description: Generates openapi.json and openapi.yaml from the FastAPI app.
agent: coder
date: 2026-10-05
version: 1.0
"""

import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# Add parent directory to path to import queryrouter
sys.path.insert(0, str(Path(__file__).parent.parent))

from queryrouter.api.main import app


def export_openapi(output_dir: Path, formats: list[str] = ["json", "yaml"]) -> None:
    """Export OpenAPI schema to specified formats.

    Args:
        output_dir: Directory to write schema files to.
        formats: List of formats to export ("json", "yaml", or both).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get the OpenAPI schema from FastAPI
    openapi_schema = app.openapi()

    # Export JSON
    if "json" in formats:
        json_path = output_dir / "openapi.json"
        with open(json_path, "w") as f:
            json.dump(openapi_schema, f, indent=2)
        print(f"✓ Exported OpenAPI schema to {json_path}")

    # Export YAML
    if "yaml" in formats and yaml is not None:
        yaml_path = output_dir / "openapi.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(openapi_schema, f, sort_keys=False, default_flow_style=False)
        print(f"✓ Exported OpenAPI schema to {yaml_path}")
    elif "yaml" in formats:
        print("⚠ PyYAML not installed, skipping YAML export", file=sys.stderr)


def main() -> None:
    """CLI entry point."""
    repo_root = Path(__file__).parent.parent
    docs_dir = repo_root / "docs"

    export_openapi(docs_dir, formats=["json", "yaml"])


if __name__ == "__main__":
    main()
