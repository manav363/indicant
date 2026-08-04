"""
intelligence/models/base.py
─────────────────────────────
Abstract base class for all models in the pipeline.

Every model (LogisticRegression, RandomForest, GradientBoost) must
implement this interface. This enforces consistency and lets the
pipeline swap models without changing any other code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelConfig:
    """Base config. Each model subclasses this with its own hyperparams."""
    prediction_horizon_months: int = 6
    confidence_threshold: float = 0.55
    random_state: int = 42


@dataclass
class PredictionResult:
    """
    Structured output from any model's predict_df() call.

    signal          : "BUY" | "HOLD" | "SELL"
    confidence      : probability of the predicted class (0.5–1.0)
    probability_up  : raw P(price higher in N months)
    model_name      : which model produced this
    ticker          : stock ticker
    feature_importances : optional dict of feature → importance score
    """
    ticker: str
    signal: str
    confidence: float
    probability_up: float
    model_name: str
    feature_importances: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "signal": self.signal,
            "confidence": round(self.confidence, 4),
            "probability_up": round(self.probability_up, 4),
            "model": self.model_name,
        }


class BaseModel(ABC):
    """Abstract interface all models must implement."""

    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        registry: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaseModel:
        """Train the model on features X and labels y.

        `registry` and `metadata` are part of the interface because
        `backtest.engine.run_backtest` passes them on every fold. They were
        omitted here while both concrete models accepted them, so the ABC
        described an interface no caller actually used — and a new subclass
        written against this signature would have raised TypeError the first
        time it reached a backtest, not at definition time.
        """
        ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return shape (m, 2): [P(Y=0), P(Y=1)] for each sample."""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions."""
        ...

    @abstractmethod
    def predict_df(self, X_df: pd.DataFrame, threshold: float = 0.5) -> PredictionResult:
        """Predict from DataFrame and return a PredictionResult."""
        ...

    @abstractmethod
    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Return DataFrame with feature importances."""
        ...
