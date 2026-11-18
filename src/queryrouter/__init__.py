# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""QueryRouter++: multi-criteria LLM routing.

Public entry points:
    - QueryFeaturizer: extract the feature vector phi(q) from a raw query
    - ModelRegistry: model profiles psi(m) loaded from the data_models/ CSVs
    - CompatibilityScorer: C(q, m, w) = sum_i w_i * A_i(q, m)
    - PreferenceEngine: resolve user preferences into weights and hard filters
    - QueryRouter: pick a model with the direct, cascade or embedding strategy
"""

__version__ = "0.2.0"

from queryrouter.core.compatibility_scorer import CompatibilityScorer
from queryrouter.core.model_registry import ModelRegistry
from queryrouter.core.preference_engine import PreferenceEngine
from queryrouter.core.query_featurizer import QueryFeaturizer
from queryrouter.core.router import QueryRouter

__all__ = [
    "__version__",
    "QueryFeaturizer",
    "ModelRegistry",
    "CompatibilityScorer",
    "PreferenceEngine",
    "QueryRouter",
]
