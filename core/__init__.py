# MIT License
# Copyright (c) 2026 QueryRouter++ Team

"""QueryRouter++ core package.

description: Package init with version and main class exports.
agent: coder
date: 2026-03-23
version: 1.0

This package implements the core routing logic for multi-criteria
LLM query routing based on formalized query-model compatibility.

Key components:
    - QueryFeaturizer: Extract feature vectors phi(q) from raw queries
    - ModelRegistry: Database of model profiles with characteristics psi(m)
    - CompatibilityScorer: Compute S(q, m, w) = sum_i w_i * f_i(q, m)
    - PreferenceEngine: Manage and resolve user preference weights
    - Router: Decision engine selecting argmax_m S(q, m, w)
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "QueryFeaturizer",
    "ModelRegistry",
    "CompatibilityScorer",
    "PreferenceEngine",
    "QueryRouter",
]
