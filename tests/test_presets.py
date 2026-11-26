# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Non-regression tests: preset model ids must match the data matrices."""

from __future__ import annotations

from pathlib import Path

from queryrouter.api.presets import PresetManager
from queryrouter.data.loaders import ModelRegistry

_ROOT = Path(__file__).resolve().parent.parent
_PRESETS_PATH = _ROOT / "config" / "presets.yaml"
_DATA_DIR = _ROOT / "data_models"


def test_presets_yaml_is_loaded() -> None:
    """The real presets.yaml parses (no silent fallback to defaults)."""
    manager = PresetManager(config_path=_PRESETS_PATH)
    # The built-in fallback presets have no allowed_models: a list proves the file was read.
    assert manager.presets["eco"].allowed_models is not None


def test_preset_allowed_models_exist_in_matrices() -> None:
    """Unknown ids in allowed_models are silently dropped by the router."""
    known = set(ModelRegistry(data_dir=_DATA_DIR).models)
    manager = PresetManager(config_path=_PRESETS_PATH)
    unknown = {
        preset_id: sorted(set(preset.allowed_models) - known)
        for preset_id, preset in manager.presets.items()
        if preset.allowed_models is not None and set(preset.allowed_models) - known
    }
    assert not unknown, f"allowed_models ids missing from data_models/: {unknown}"
