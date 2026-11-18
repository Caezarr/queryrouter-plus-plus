# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Filesystem locations of the data the router loads at runtime.

The model matrices (``data_models/``) and the LibreChat presets
(``config/presets.yaml``) live at the repository root, next to ``src/``, so
operators can edit or volume-mount them without rebuilding the package. Every
module resolves them through this file instead of walking ``__file__`` itself.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data_models"
DEFAULT_PRESETS_PATH = REPO_ROOT / "config" / "presets.yaml"
