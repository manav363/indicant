"""
market_regime/models/logistic.py
──────────────────────────────────
Logistic Regression implemented from scratch using numpy.

Why start here?
- It's the foundation of all classification ML.
- Every concept (loss function, gradient, regularisation, probability
  calibration) appears in more complex models too.
- If you can't explain logistic regression, you can't defend any model.

What we're predicting:
    Given features at time t, will the stock price be higher in
    PREDICTION_HORIZON months? (binary: 1=up, 0=not up)

    This is framed as: P(Y=1 | X) where Y=1 means "price higher in 6 months"

Math covered in this file:
    1. Sigmoid function
    2. Log-likelihood loss (binary cross-entropy)
    3. L2 regularisation (weight decay)
    4. Gradient descent (batch and mini-batch)
    5. Gradient computation (backprop for 1-layer network)
    6. Probability calibration check
    7. Feature importance via coefficient magnitude
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from market_regime.models.base import BaseModel, ModelConfig, PredictionResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LogisticConfig(ModelConfig):
    """
    Hyperparameters for logistic regression.

    learning_rate : α in the gradient update rule
        w ← w - α * ∇L(w)
        Too high → diverges. Too low → slow convergence.
        Typical range: 0.001 to 0.1

    n_epochs : number of full passes over the training data
        Each epoch = one complete gradient descent step (batch GD)
        or multiple steps (mini-batch SGD)

    lambda_l2 : L2 regularisation strength (λ)
        Adds λ * ||w||² to the loss, penalising large weights.
        Prevents overfitting. Higher λ = more regularisation.
        λ=0 means no regularisation.

    batch_size : None = full batch GD, int = mini-batch SGD
        Full batch: stable but slow on large datasets
        Mini-batch: noisier but faster, often finds better minima

    tolerance : early stopping threshold
        Stop training if loss improvement < tolerance for patience epochs.

    patience : epochs to wait before early stopping
    """
    learning_rate: float = 0.01
    n_epochs: int = 1000
    lambda_l2: float = 0.01
    batch_size: Optional[int] = None    # None = full batch
    tolerance: float = 1e-6
    patience: int = 20
    fit_intercept: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class LogisticRegressionScratch(BaseModel):
    """
    Binary logistic regression trained with gradient descent.

    The model learns weights w and bias b such that:

        z = w · x + b              (linear combination)
        ŷ = σ(z) = 1/(1+e^{-z})   (sigmoid → probability)

    Training minimises the binary cross-entropy loss:

        L(w, b) = -(1/m) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
                + (λ/2m) ||w||²    (L2 regularisation term)

    Gradients (derived by chain rule):

        ∂L/∂w = (1/m) Xᵀ(ŷ - y) + (λ/m) w
        ∂L/∂b = (1/m) Σ(ŷ - y)

    Update rule (gradient descent):

        w ← w - α * ∂L/∂w
        b ← b - α * ∂L/∂b
    """

    def __init__(self, config: Optional[LogisticConfig] = None) -> None:
        self.config = config or LogisticConfig()
        self.weights: Optional[np.ndarray] = None   # shape: (n_features,)
        self.bias: float = 0.0
        self.feature_names: list[str] = []
        self.loss_history: list[float] = []
        self.is_fitted: bool = False
        self._scaler_mean: Optional[np.ndarray] = None
        self._scaler_std: Optional[np.ndarray] = None

    # ── Core math ─────────────────────────────────────────────────────────

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """
        Sigmoid (logistic) function.

        σ(z) = 1 / (1 + e^{-z})

        Maps any real number to (0, 1) — interpretable as probability.

        Properties:
        - σ(0) = 0.5
        - σ(z) → 1 as z → +∞
        - σ(z) → 0 as z → -∞
        - σ'(z) = σ(z)(1 - σ(z))   ← used in backprop

        Numerically stable implementation:
        For large negative z, e^{-z} overflows.
        We use: σ(z) = e^z / (1 + e^z) for z < 0
        """
        return np.where(
            z >= 0,
            1.0 / (1.0 + np.exp(-z)),
            np.exp(z) / (1.0 + np.exp(z))
        )

    @staticmethod
    def binary_cross_entropy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        weights: np.ndarray,
        lambda_l2: float,
    ) -> float:
        """
        Binary cross-entropy loss with L2 regularisation.

        L = -(1/m) Σ [y log(ŷ) + (1-y) log(1-ŷ)] + (λ/2m) ||w||²

        The log terms:
        - When y=1: only log(ŷ) matters → penalises low confidence on true positives
        - When y=0: only log(1-ŷ) matters → penalises high confidence on negatives

        L2 term:
        - (λ/2m) ||w||² = (λ/2m) Σ w_j²
        - Penalises large weights → encourages simpler models
        - Does NOT regularise the bias term (standard practice)

        Clip predictions to avoid log(0) = -∞
        """
        m = len(y_true)
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)

        # Cross-entropy term
        ce = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )

        # L2 regularisation term (note: 2m denominator, not m, is convention)
        l2 = (lambda_l2 / (2 * m)) * np.sum(weights ** 2)

        return float(ce + l2)

    def _compute_gradients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Compute gradients of loss w.r.t. weights and bias.

        Derivation:
            Error = ŷ - y   (prediction minus truth)

            ∂L/∂w = (1/m) Xᵀ · error + (λ/m) · w
                    └── CE gradient ──┘   └── L2 ──┘

            ∂L/∂b = (1/m) Σ error
                    (bias is not regularised)

        This is the vectorised form — computes gradients for ALL
        weights simultaneously using matrix multiplication.
        Much faster than looping over features.
        """
        m = len(y)
        cfg = self.config
        assert isinstance(cfg, LogisticConfig)

        error = y_pred - y                                      # (m,)
        grad_w = (X.T @ error) / m + (cfg.lambda_l2 / m) * self.weights  # (n,)
        grad_b = float(np.mean(error))

        return grad_w, grad_b

    # ── Training ───────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        registry: Optional[Any] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> "LogisticRegressionScratch":
        """
        Train the model using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (m, n)
            Feature matrix. m = samples, n = features.
        y : np.ndarray, shape (m,)
            Binary labels. Must be 0 or 1.
        feature_names : list[str], optional
            Names for each feature column. Used for reporting.
        registry : ModelRegistry, optional
            If provided, logs this training run after fit completes.
        metadata : dict, optional
            Run metadata for the registry (ticker, data_start, etc.).

        Returns
        -------
        self (for method chaining)
        """
        cfg = self.config
        assert isinstance(cfg, LogisticConfig)

        # Validate inputs
        X, y = self._validate_inputs(X, y)
        m, n = X.shape

        # Standardise features: z = (x - μ) / σ
        # Critical for gradient descent convergence.
        # Without this, features on different scales have wildly different
        # gradient magnitudes, causing slow/unstable training.
        X, self._scaler_mean, self._scaler_std = self._standardise(X)

        # Initialise weights
        # Xavier initialisation: w ~ N(0, 1/n)
        # Keeps gradients in a reasonable range at the start
        rng = np.random.default_rng(42)
        self.weights = rng.normal(0, 1.0 / np.sqrt(n), size=n)
        self.bias = 0.0
        self.feature_names = feature_names or [f"f{i}" for i in range(n)]
        self.loss_history = []

        best_loss = float("inf")
        patience_counter = 0

        logger.info(
            "Training LogisticRegression: m=%d samples, n=%d features, "
            "%d epochs, lr=%.4f, λ=%.4f",
            m, n, cfg.n_epochs, cfg.learning_rate, cfg.lambda_l2,
        )

        for epoch in range(cfg.n_epochs):
            # ── Mini-batch or full batch ────────────────────────────────
            if cfg.batch_size is not None:
                X_batch, y_batch = self._get_batch(X, y, cfg.batch_size, rng)
            else:
                X_batch, y_batch = X, y

            # ── Forward pass ────────────────────────────────────────────
            # z = Xw + b
            z = X_batch @ self.weights + self.bias

            # ŷ = σ(z)
            y_pred = self.sigmoid(z)

            # ── Loss ────────────────────────────────────────────────────
            loss = self.binary_cross_entropy(y_batch, y_pred, self.weights, cfg.lambda_l2)
            self.loss_history.append(loss)

            # ── Backward pass (gradients) ────────────────────────────────
            grad_w, grad_b = self._compute_gradients(X_batch, y_batch, y_pred)

            # ── Weight update ────────────────────────────────────────────
            # w ← w - α * ∂L/∂w
            # b ← b - α * ∂L/∂b
            self.weights -= cfg.learning_rate * grad_w
            self.bias -= cfg.learning_rate * grad_b

            # ── Early stopping ───────────────────────────────────────────
            if loss < best_loss - cfg.tolerance:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d (loss=%.6f)", epoch, loss)
                    break

            if (epoch + 1) % 100 == 0:
                logger.debug("Epoch %d/%d — loss: %.6f", epoch + 1, cfg.n_epochs, loss)

        self.is_fitted = True
        logger.info(
            "Training complete. Final loss: %.6f (%.0f%% of class 1)",
            self.loss_history[-1], y.mean() * 100
        )

        # ── Model registry logging ─────────────────────────────────────────
        if registry is not None:
            from market_regime.registry.model_registry import ModelRegistry as _Reg
            if not isinstance(registry, _Reg):
                logger.warning("fit() called with invalid registry instance — skipping log.")
            else:
                run_meta = {
                    "ticker": (metadata or {}).get("ticker", ""),
                    "model_type": "logistic",
                    "model_config": cfg,
                    "data_start": (metadata or {}).get("data_start", ""),
                    "data_end": (metadata or {}).get("data_end", ""),
                    "n_samples": len(X),
                    "n_features": X.shape[1],
                    "horizon_days": (metadata or {}).get("horizon_days", 126),
                    "label_threshold": (metadata or {}).get("label_threshold", 0.0),
                    "feature_list": self.feature_names,
                }
                registry.log_run(run_meta)
                logger.info("Logged run to model registry.")

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Returns
        -------
        np.ndarray, shape (m, 2)
            [:, 0] = P(Y=0), [:, 1] = P(Y=1)
        """
        self._check_fitted()
        X = self._apply_scaler(X)
        z = X @ self.weights + self.bias
        p1 = self.sigmoid(z)
        return np.column_stack([1 - p1, p1])

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels.

        Threshold: predict 1 if P(Y=1) > threshold, else 0.
        Default threshold = 0.5, but for trading we may prefer
        a higher threshold (e.g. 0.6) to only act on high-confidence signals.
        """
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    # ── BaseModel interface ────────────────────────────────────────────────

    def fit_df(
        self,
        X_df: pd.DataFrame,
        y: pd.Series,
    ) -> "LogisticRegressionScratch":
        """Fit from DataFrame (convenience wrapper)."""
        return self.fit(
            X_df.values,
            y.values,
            feature_names=list(X_df.columns),
        )

    def predict_df(
        self,
        X_df: pd.DataFrame,
        threshold: float = 0.5,
    ) -> PredictionResult:
        """Predict from DataFrame and return structured result."""
        proba = self.predict_proba(X_df.values)
        p_up = proba[:, 1]
        labels = (p_up >= threshold).astype(int)

        signal = np.where(labels == 1, "BUY", "HOLD")

        return PredictionResult(
            ticker="",
            signal=signal[-1],
            confidence=float(p_up[-1]),
            probability_up=float(p_up[-1]),
            model_name="LogisticRegressionScratch",
        )

    # ── Interpretability ───────────────────────────────────────────────────

    def feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Feature importance = absolute value of learned weights.

        In logistic regression, larger |w_j| means feature j has
        more influence on the prediction.

        Note: this is only valid because we standardised features.
        Without standardisation, weight magnitude depends on feature scale
        and is NOT a reliable importance measure.
        """
        self._check_fitted()
        assert self.weights is not None

        importance = pd.DataFrame({
            "feature": self.feature_names,
            "weight": self.weights,
            "abs_weight": np.abs(self.weights),
        }).sort_values("abs_weight", ascending=False)

        return importance.head(top_n).reset_index(drop=True)

    def get_loss_curve(self) -> pd.Series:
        """Return training loss history as a Series."""
        return pd.Series(self.loss_history, name="loss")

    # ── Internal helpers ───────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if len(X) != len(y):
            raise ValueError(f"X ({len(X)}) and y ({len(y)}) length mismatch")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y must contain only 0 and 1")
        if np.isnan(X).any():
            raise ValueError("X contains NaN values — run preprocessor first")

        return X, y

    @staticmethod
    def _standardise(
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Standardise features: z = (x - μ) / σ

        Returns (X_scaled, mean, std) so we can apply same transform to
        test data using the TRAINING statistics (no leakage).
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std = np.where(std == 0, 1.0, std)   # avoid division by zero
        return (X - mean) / std, mean, std

    def _apply_scaler(self, X: np.ndarray) -> np.ndarray:
        """Apply training-time scaler to new data."""
        assert self._scaler_mean is not None and self._scaler_std is not None
        return (np.array(X, dtype=np.float64) - self._scaler_mean) / self._scaler_std

    @staticmethod
    def _get_batch(
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample a random mini-batch."""
        idx = rng.choice(len(X), size=min(batch_size, len(X)), replace=False)
        return X[idx], y[idx]

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
