"""
intelligence/registry/__init__.py
────────────────────────────────────
Model registry package — SQLite-based tracking of training runs.
"""
from intelligence.registry.model_registry import ModelRegistry

__all__ = ["ModelRegistry"]
