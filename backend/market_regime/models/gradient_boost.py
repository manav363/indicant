"""
market_regime/models/gradient_boost.py
────────────────────────────────────────
Gradient Boosting model using XGBoost.

This is the "enhance with libraries" step — after understanding the
math via logistic regression, we use XGBoost for production because:
- Handles non-linear relationships automatically
- Built-in feature importance (gain, cover, frequency)
- Regularisation (L1 + L2) built in
- Handles missing values natively
- Orders of magnitude faster than our scratch implementations

The math behind gradient boosting:
───────────────────────────────────
Gradient boosting builds an ensemble of weak learners (decision trees)
sequentially. Each tree corrects the errors of the previous ones.

Mathematically:
    F_0(x) = initial prediction (e.g. log-odds of base rate)

    For m = 1 to M:
        1. Compute pseudo-residuals (negative gradient of loss):
           r_i = -∂L(y_i, F_{m-1}(x_i)) / ∂F_{m-1}(x_i)

           For log-loss: r_i = y_i - σ(F_{m-1}(x_i))
           = actual - predicted probability
           (This is the same gradient as in logistic regression!)

        2. Fit a decision tree h_m(x) to the residuals r

        3. Update: F_m(x) = F_{m-1}(x) + η * h_m(x)
           where η (learning_rate) shrinks each tree's contribution.
           Lower η = more trees needed but better generalisation.

    Final prediction: F_M(x) → sigmoid → P(Y=1)

Why this works:
    Each tree focuses on the samples the previous model got wrong.
    It's gradient descent in function space rather than parameter space.
    The result is a powerful non-linear model that can capture
    complex interactions between features (e.g. high RSI + low volume
    → weak signal, but high RSI + high OBV → strong signal).

XGBoost improvements over vanilla GBM:
    - Second-order gradients (Newton's method, not just gradient)
    - Column subsampling (like Random Forest — reduces correlation)
    - L1 (alpha) + L2 (lambda) regularisation on tree weights
    - Sparsity-aware split finding (handles NaN natively)
    - Parallel tree construction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from market_regime.models.base import BaseModel, ModelConfig, PredictionResult

logger = logging.getLogger(__name__)


@dataclass
class GradientBoostConfig(ModelConfig):
    """
    XGBoost hyperparameters.

    n_estimators : number of trees (M in the math above)
        More trees = more capacity but slower + potential overfit.
        Use early stopping to find the optimal number.

    learning_rate (η) : shrinkage applied to each tree
        Lower = need more trees, but generalises better.
        Typical: 0.01-0.1

    max_depth : maximum depth of each tree
        Controls complexity. Deeper = more interactions captured
        but higher risk of overfitting.
        For financial data: 3-6 is usually optimal.

    subsample : fraction of training samples used per tree
        Introduces randomness → reduces variance (like bagging).

    colsample_bytree : fraction of features used per tree
        Further reduces correlation between trees.

    min_child_weight : minimum sum of instance weights in a leaf
        Higher = more conservative, reduces overfitting.

    gamma : minimum loss reduction to make a split
        Regularisation — only split if it reduces loss by at least γ.

    reg_alpha (L1) : sparsity-inducing regularisation
    reg_lambda (L2) : weight decay regularisation

    calibrate : whether to calibrate probabilities with Platt scaling
        XGBoost probabilities can be poorly calibrated (too extreme).
        Platt scaling (sigmoid) or isotonic regression fixes this.
        Important for honest confidence scores.
    """
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 4
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 10    # high = more conservative (good for finance)
    gamma: float = 0.1
    reg_alpha: float = 0.1        # L1
    reg_lambda: float = 1.0       # L2
    early_stopping_rounds: int = 50
    calibrate: bool = True        # calibrate probabilities


class GradientBoostModel(BaseModel):
    """
    XGBoost classifier with optional probability calibration.

    Wraps xgb.XGBClassifier with our BaseModel interface.
    """

    def __init__(self, config: Optional[GradientBoostConfig] = None) -> None:
        self.config = config or GradientBoostConfig()
        self.model: Optional[xgb.XGBClassifier] = None
        self.calibrated_model = None
        self._calibrator = None
        self._calibrated = False
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "GradientBoostModel":
        """
        Train XGBoost model.

        Parameters
        ----------
        X, y : training data
        feature_names : list of feature names for importance plots
        X_val, y_val : optional validation set for early stopping.
            If not provided, uses 20% of training data.
        """
        cfg = self.config
        self.feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Split validation set for early stopping if not provided
        if X_val is None:
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
        else:
            X_train, y_train = X, y

        # Class weight for imbalanced labels
        # If 40% buy / 60% hold, scale_pos_weight = 60/40 = 1.5
        # This prevents the model from always predicting the majority class
        pos_count = y_train.sum()
        neg_count = len(y_train) - pos_count
        scale_pos_weight = neg_count / max(pos_count, 1)

        self.model = xgb.XGBClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            min_child_weight=cfg.min_child_weight,
            gamma=cfg.gamma,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=cfg.random_state,
            n_jobs=-1,
            early_stopping_rounds=cfg.early_stopping_rounds,
        )

        logger.info(
            "Training XGBoost: %d train samples, %d val samples, "
            "%d features. Class balance: %.1f%% positive.",
            len(X_train), len(X_val), X.shape[1], y_train.mean() * 100
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        actual_trees = self.model.best_iteration + 1 if hasattr(self.model, 'best_iteration') else cfg.n_estimators
        logger.info("XGBoost trained: %d trees used (early stopping).", actual_trees)

        # ── Probability calibration ────────────────────────────────────────
        # XGBoost is often overconfident or underconfident in its probabilities.
        # Platt scaling fits a sigmoid on top of the raw scores using
        # cross-validation, mapping them to better-calibrated probabilities.
        #
        # After calibration:
        # When model says P=0.7, it should be right ~70% of the time.
        if cfg.calibrate:
            logger.info("Calibrating probabilities with Platt scaling...")
            from sklearn.linear_model import LogisticRegression as _LR
            raw_scores = self.model.predict_proba(X_val)[:, 1].reshape(-1, 1)
            self._calibrator = _LR()
            self._calibrator.fit(raw_scores, y_val)
            self._calibrated = True
            logger.info("Calibration complete.")

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities if available, else raw."""
        self._check_fitted()
        raw = self.model.predict_proba(X)
        if self._calibrated and self._calibrator is not None:
            p1 = self._calibrator.predict_proba(raw[:, 1].reshape(-1, 1))[:, 1]
            return np.column_stack([1 - p1, p1])
        return raw

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def predict_df(
        self,
        X_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> PredictionResult:
        proba = self.predict_proba(X_df.values)
        p_up = float(proba[-1, 1])
        signal = "BUY" if p_up >= threshold else "HOLD"

        importances = self.feature_importance()
        imp_dict = dict(zip(importances["feature"], importances["importance"]))

        return PredictionResult(
            ticker="",
            signal=signal,
            confidence=p_up if p_up >= 0.5 else 1 - p_up,
            probability_up=p_up,
            model_name="GradientBoost_XGBoost",
            feature_importances=imp_dict,
        )

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        XGBoost feature importance by 'gain'.

        Gain = average improvement in loss for splits using this feature.
        More meaningful than 'frequency' (how often feature is used).
        """
        self._check_fitted()
        scores = self.model.get_booster().get_score(importance_type="gain")

        df = pd.DataFrame([
            {"feature": k, "importance": v}
            for k, v in scores.items()
        ]).sort_values("importance", ascending=False)

        # Map back feature names if available
        if self.feature_names:
            df["feature"] = df["feature"].apply(
                lambda f: self.feature_names[int(f[1:])]
                if f.startswith("f") and f[1:].isdigit()
                else f
            )

        return df.head(top_n).reset_index(drop=True)

    def _check_fitted(self) -> None:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
