# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Shared router factory for QueryRouter++ API modules.

Provides a single cached router instance factory used by both the native
API endpoints (main.py) and the OpenAI-compatible proxy (openai_compat.py).
"""

from __future__ import annotations

from pathlib import Path

from queryrouter.core.router import QueryRouter

_DATA_DIR = Path(__file__).resolve().parents[1] / "data_models"
_routers: dict[tuple[str, float], QueryRouter] = {}


def get_router(strategy: str = "direct", cascade_threshold: float = 0.6) -> QueryRouter:
    """Get or create a cached router instance.

    Args:
        strategy: Routing strategy name.
        cascade_threshold: Threshold for cascade strategy (default 0.6).

    Returns:
        QueryRouter instance.
    """
    cache_key = (strategy, cascade_threshold)
    if cache_key not in _routers:
        data_dir = _DATA_DIR if _DATA_DIR.exists() else None
        _routers[cache_key] = QueryRouter(
            strategy=strategy,
            data_dir=data_dir,
            cascade_threshold=cascade_threshold,
        )
    return _routers[cache_key]
