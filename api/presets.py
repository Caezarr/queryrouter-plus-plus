# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""Preset configuration loader for QueryRouter++.

Loads presets from YAML config and provides preset resolution for LibreChat integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml


@dataclass
class PresetConfig:
    """Configuration for a single preset/mode.
    
    Attributes:
        name: Display name for the preset.
        description: Human-readable description.
        icon: Emoji/icon for display.
        allowed_models: List of model IDs allowed, or None for all.
        weights: Weight vector (w_performance, w_cost, w_latency, w_ecology).
        strategy: Routing strategy.
        cascade_threshold: Threshold for cascade strategy.
    """
    name: str
    description: str
    icon: str = ""
    allowed_models: list[str] | None = None
    weights: dict[str, float] = field(default_factory=dict)
    strategy: Literal["direct", "cascade", "embedding"] = "direct"
    cascade_threshold: float = 0.6

    def __post_init__(self) -> None:
        """Set default weights if not provided."""
        if not self.weights:
            self.weights = {
                "w_performance": 0.25,
                "w_cost": 0.25,
                "w_latency": 0.25,
                "w_ecology": 0.25,
            }


class PresetManager:
    """Manages preset configurations from YAML file.
    
    Loads presets from config/presets.yaml and provides lookup by preset ID.
    Maps LibreChat mode names to preset configurations.
    
    Attributes:
        presets: Mapping of preset_id to PresetConfig.
        config_path: Path to the presets YAML file.
    """

    # Mapping from LibreChat model names to preset IDs
    MODE_TO_PRESET: dict[str, str] = {
        "mode-ecologique": "eco",
        "mode-eco": "eco",
        "eco": "eco",
        "mode-performance": "performance",
        "performance": "performance",
        "perf": "performance",
        "mode-economique": "economique",
        "economique": "economique",
        "cheap": "economique",
        "cost": "economique",
        "mode-equilibre": "equilibre",
        "equilibre": "equilibre",
        "balanced": "equilibre",
        "default": "equilibre",
    }

    def __init__(self, config_path: Path | None = None) -> None:
        """Initialize the preset manager.
        
        Args:
            config_path: Path to presets.yaml. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).resolve().parents[1] / "config" / "presets.yaml"
        
        self.config_path = config_path
        self.presets: dict[str, PresetConfig] = {}
        self._load_presets()

    def _load_presets(self) -> None:
        """Load presets from YAML file."""
        if not self.config_path.exists():
            # Use default presets if file doesn't exist
            self._load_defaults()
            return

        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            presets_data = data.get("presets", {})
            for preset_id, config in presets_data.items():
                self.presets[preset_id] = PresetConfig(
                    name=config.get("name", preset_id),
                    description=config.get("description", ""),
                    icon=config.get("icon", ""),
                    allowed_models=config.get("allowed_models"),
                    weights=config.get("weights", {}),
                    strategy=config.get("strategy", "direct"),
                    cascade_threshold=config.get("cascade_threshold", 0.6),
                )
        except Exception as e:
            # Fallback to defaults on error
            print(f"Warning: Failed to load presets from {self.config_path}: {e}")
            self._load_defaults()

    def _load_defaults(self) -> None:
        """Load default presets if YAML file is missing or invalid."""
        self.presets = {
            "eco": PresetConfig(
                name="Mode Écologique",
                description="Privilégie l'empreinte carbone",
                icon="🌱",
                allowed_models=None,
                weights={
                    "w_performance": 0.15,
                    "w_cost": 0.10,
                    "w_latency": 0.10,
                    "w_ecology": 0.65,
                },
                strategy="direct",
            ),
            "performance": PresetConfig(
                name="Mode Performance",
                description="La meilleure qualité possible",
                icon="⚡",
                allowed_models=None,
                weights={
                    "w_performance": 0.85,
                    "w_cost": 0.05,
                    "w_latency": 0.05,
                    "w_ecology": 0.05,
                },
                strategy="direct",
            ),
            "economique": PresetConfig(
                name="Mode Économique",
                description="Le moins cher possible",
                icon="💰",
                allowed_models=None,
                weights={
                    "w_performance": 0.10,
                    "w_cost": 0.80,
                    "w_latency": 0.05,
                    "w_ecology": 0.05,
                },
                strategy="cascade",
                cascade_threshold=0.6,
            ),
            "equilibre": PresetConfig(
                name="Mode Équilibré",
                description="Bon rapport qualité/prix",
                icon="⚖️",
                allowed_models=None,
                weights={
                    "w_performance": 0.30,
                    "w_cost": 0.40,
                    "w_latency": 0.15,
                    "w_ecology": 0.15,
                },
                strategy="direct",
            ),
        }

    def get_preset(self, preset_id: str) -> PresetConfig | None:
        """Get a preset by its ID.
        
        Args:
            preset_id: The preset identifier (e.g., "eco", "performance").
            
        Returns:
            PresetConfig or None if not found.
        """
        return self.presets.get(preset_id)

    def resolve_mode(self, mode_name: str) -> PresetConfig:
        """Resolve a LibreChat mode name to a preset config.
        
        Args:
            mode_name: The mode/model name from LibreChat (e.g., "mode-ecologique").
            
        Returns:
            PresetConfig for the resolved preset, or default (equilibre) if unknown.
        """
        mode_lower = mode_name.lower()
        preset_id = self.MODE_TO_PRESET.get(mode_lower, "equilibre")
        return self.presets.get(preset_id, self.presets.get("equilibre", list(self.presets.values())[0]))

    def list_presets(self) -> list[dict]:
        """List all available presets with their metadata.
        
        Returns:
            List of preset metadata dicts.
        """
        return [
            {
                "id": preset_id,
                "name": config.name,
                "description": config.description,
                "icon": config.icon,
                "strategy": config.strategy,
                "models_count": len(config.allowed_models) if config.allowed_models else None,
            }
            for preset_id, config in self.presets.items()
        ]


# Global preset manager instance
_preset_manager: PresetManager | None = None


def get_preset_manager() -> PresetManager:
    """Get or create the global preset manager instance."""
    global _preset_manager
    if _preset_manager is None:
        _preset_manager = PresetManager()
    return _preset_manager


def resolve_mode(mode_name: str) -> PresetConfig:
    """Convenience function to resolve a mode name to preset config."""
    return get_preset_manager().resolve_mode(mode_name)


def get_preset(preset_id: str) -> PresetConfig | None:
    """Convenience function to get a preset by ID."""
    return get_preset_manager().get_preset(preset_id)


def list_presets() -> list[dict]:
    """Convenience function to list all presets."""
    return get_preset_manager().list_presets()
