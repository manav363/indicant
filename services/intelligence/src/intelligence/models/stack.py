"""L2 base learners, L3 meta-learner, L5 meta-labeller.

The layered engine. Its structure is less important than one property:

    **The meta-learner must only ever see PURGED, EMBARGOED out-of-fold
    predictions from the base learners.**

If it sees in-sample base predictions, it learns to trust models that already
know the answer, the stack reports an excellent score, and every number
downstream is fiction. The literature flags this explicitly: improper stacking
"can produce overoptimistic outcomes". It is the single most likely way this
build goes wrong, so `generate_oof` is the only path that produces meta features
and it takes a `PurgedPanelCV`, not a fold count.

Layers:

  L2  heterogeneous base learners, each emitting purged OOF predictions
  L3  meta-learner over those OOF predictions plus regime state
  L5  meta-labeller: 'given L3 says BUY, will THIS BUY be right?'

The ElasticNet in L2 is the honest baseline. If five models and a meta-learner
cannot beat a regularised linear model out of sample, that is the finding, and
`ModelCard.beats_baseline` is built to report it rather than bury it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from intelligence.validation.panel_cv import PurgedPanelCV

logger = logging.getLogger(__name__)

# Below this many training rows a fold's model is fitting noise; skip it and say
# so rather than emitting predictions nobody should trust.
MIN_FOLD_TRAIN_ROWS = 500


class Learner(Protocol):
    """Minimal surface the stack needs. Keeps sklearn/xgboost/lightgbm
    interchangeable without the stack importing any of them by name.
    """

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None) -> Any: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


@dataclass
class BaseLearnerSpec:
    """A named base learner and how to build a fresh one.

    `factory` rather than an instance, because every fold needs an unfitted
    model — reusing one across folds is a subtle way to leak.
    """

    name: str
    factory: Any
    needs_scaling: bool = False


def default_base_learners(random_state: int = 42) -> list[BaseLearnerSpec]:
    """The L2 roster.

    Shallow by design. The asset-pricing literature finds boosted trees "more
    consistent and reliable than deeper ones… likely due to a low
    signal-to-noise ratio". Depth here buys variance, not skill.

    XGBoost and LightGBM are added only if installed, so the stack degrades to
    sklearn-only rather than failing at import in a slim environment.
    """
    specs = [
        # The honest baseline. Everything else has to beat this.
        BaseLearnerSpec(
            name="elasticnet",
            factory=lambda: SGDClassifier(
                loss="log_loss",
                penalty="elasticnet",
                l1_ratio=0.15,
                alpha=1e-4,
                max_iter=2000,
                tol=1e-4,
                random_state=random_state,
            ),
            needs_scaling=True,
        ),
        BaseLearnerSpec(
            name="random_forest",
            factory=lambda: RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=50,
                n_jobs=-1,
                random_state=random_state,
            ),
        ),
        BaseLearnerSpec(
            name="mlp",
            factory=lambda: MLPClassifier(
                hidden_layer_sizes=(32, 16),  # GKX-style: shallow, not deep
                alpha=1e-3,
                max_iter=400,
                early_stopping=True,
                random_state=random_state,
            ),
            needs_scaling=True,
        ),
    ]

    try:
        from xgboost import XGBClassifier

        specs.append(
            BaseLearnerSpec(
                name="xgboost",
                factory=lambda: XGBClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=1.0,
                    n_jobs=-1,
                    random_state=random_state,
                    eval_metric="logloss",
                ),
            )
        )
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("xgboost not installed; excluded from the stack")

    try:
        from lightgbm import LGBMClassifier

        specs.append(
            BaseLearnerSpec(
                name="lightgbm",
                factory=lambda: LGBMClassifier(
                    n_estimators=400,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    n_jobs=-1,
                    random_state=random_state,
                    verbose=-1,
                ),
            )
        )
    except ImportError:  # pragma: no cover - environment dependent
        logger.warning("lightgbm not installed; excluded from the stack")

    return specs


@dataclass
class OOFResult:
    """Out-of-fold predictions, one column per base learner.

    `coverage` matters: rows never in any test fold have no OOF prediction and
    must not reach the meta-learner. Silently filling them with 0.5 would train
    L3 on fabricated inputs.
    """

    predictions: pd.DataFrame
    fold_of_row: np.ndarray
    learner_names: list[str]
    skipped_folds: dict[int, str] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        return float(self.predictions.notna().all(axis=1).mean())

    @property
    def covered_mask(self) -> np.ndarray:
        return self.predictions.notna().all(axis=1).to_numpy()


def generate_oof(
    X: pd.DataFrame,
    y: pd.Series,
    cv: PurgedPanelCV,
    panel: pd.DataFrame,
    *,
    learners: list[BaseLearnerSpec] | None = None,
    sample_weight: pd.Series | None = None,
) -> OOFResult:
    """Purged, embargoed out-of-fold predictions from every base learner.

    THE function that keeps the stack honest. A fold's model never sees its own
    test rows, nor the purged band before them, nor the embargoed band after.
    """
    specs = learners or default_base_learners()
    splits = list(cv.split(panel))
    if not splits:
        raise ValueError("cross-validator produced no folds")

    oof = pd.DataFrame(
        {spec.name: np.full(len(X), np.nan) for spec in specs}, index=X.index
    )
    fold_of_row = np.full(len(X), -1, dtype="int64")
    skipped: dict[int, str] = {}

    X_values = X.to_numpy(dtype="float64")
    y_values = y.to_numpy(dtype="float64")
    weights = None if sample_weight is None else sample_weight.to_numpy(dtype="float64")

    for split in splits:
        tr, te = split.train_idx, split.test_idx

        # Drop unlabelled rows from training; keep the test index intact so
        # coverage reflects reality.
        tr = tr[np.isfinite(y_values[tr])]
        if tr.size < MIN_FOLD_TRAIN_ROWS:
            skipped[split.fold] = f"{tr.size} labelled train rows < {MIN_FOLD_TRAIN_ROWS}"
            continue
        if np.unique(y_values[tr]).size < 2:
            skipped[split.fold] = "training fold is single-class"
            continue

        fold_of_row[te] = split.fold
        X_tr, y_tr = X_values[tr], y_values[tr]
        w_tr = None if weights is None else weights[tr]
        X_te = X_values[te]

        for spec in specs:
            try:
                X_fit, X_pred = X_tr, X_te
                if spec.needs_scaling:
                    # Fitted on TRAIN ONLY. Scaling on the full panel would leak
                    # the test period's distribution into the training features.
                    scaler = StandardScaler().fit(X_tr)
                    X_fit, X_pred = scaler.transform(X_tr), scaler.transform(X_te)

                model = spec.factory()
                try:
                    model.fit(X_fit, y_tr, sample_weight=w_tr)
                except TypeError:
                    model.fit(X_fit, y_tr)

                proba = model.predict_proba(X_pred)[:, 1]
                oof.iloc[te, oof.columns.get_loc(spec.name)] = proba
            except Exception as exc:
                logger.warning("fold %s learner %s failed: %s", split.fold, spec.name, exc)

    return OOFResult(
        predictions=oof,
        fold_of_row=fold_of_row,
        learner_names=[s.name for s in specs],
        skipped_folds=skipped,
    )


@dataclass
class StackResult:
    meta_model: Any
    scaler: StandardScaler | None
    learner_names: list[str]
    oof: OOFResult
    meta_oof: pd.Series
    n_train: int

    def predict_proba(self, base_predictions: pd.DataFrame) -> np.ndarray:
        X = base_predictions[self.learner_names].to_numpy(dtype="float64")
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return self.meta_model.predict_proba(X)[:, 1]


def fit_meta_learner(
    oof: OOFResult,
    y: pd.Series,
    *,
    regime: pd.Series | None = None,
    sample_weight: pd.Series | None = None,
    random_state: int = 42,
) -> StackResult:
    """L3 — stack over base-learner OOF predictions.

    Shallow on purpose. The meta-learner has 5 inputs and a low-signal target;
    anything with capacity will memorise fold idiosyncrasies. Logistic
    regression also keeps the learned weights readable, so 'which base model is
    the stack actually relying on' is an answerable question.
    """
    covered = oof.covered_mask & y.notna().to_numpy()
    if covered.sum() < MIN_FOLD_TRAIN_ROWS:
        raise ValueError(
            f"only {covered.sum()} rows have complete OOF predictions and a label; "
            f"need {MIN_FOLD_TRAIN_ROWS}. Check fold coverage before stacking."
        )

    features = oof.predictions.loc[covered, oof.learner_names].copy()
    if regime is not None:
        # Regime enters as a feature so the meta-learner can weight base models
        # differently by market state — the interaction is non-stationary, which
        # is the documented reason stacking helps here at all.
        features["regime"] = regime.loc[covered].to_numpy()

    X = features.to_numpy(dtype="float64")
    y_train = y.loc[covered].to_numpy(dtype="float64")
    w = None if sample_weight is None else sample_weight.loc[covered].to_numpy()

    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(scaler.transform(X), y_train, sample_weight=w)

    meta_oof = pd.Series(np.nan, index=y.index, name="meta_proba")
    meta_oof.loc[covered] = model.predict_proba(scaler.transform(X))[:, 1]

    return StackResult(
        meta_model=model,
        scaler=scaler,
        learner_names=list(features.columns),
        oof=oof,
        meta_oof=meta_oof,
        n_train=int(covered.sum()),
    )


@dataclass
class MetaLabelResult:
    model: Any
    scaler: StandardScaler
    threshold: float
    n_train: int
    base_rate: float

    def conviction(self, primary_proba: np.ndarray, features: np.ndarray) -> np.ndarray:
        X = np.column_stack([primary_proba, features])
        return self.model.predict_proba(self.scaler.transform(X))[:, 1]


def fit_meta_labeller(
    primary_proba: pd.Series,
    y: pd.Series,
    features: pd.DataFrame,
    *,
    threshold: float = 0.5,
    sample_weight: pd.Series | None = None,
    random_state: int = 42,
) -> MetaLabelResult:
    """L5 — meta-labelling. Side and size, separated.

    The primary model says which way. This model answers a different and easier
    question: *given* the primary said BUY, will this particular BUY be right?

    It does not find more trades. It declines the bad ones — which for a signal
    with a marginal edge is what makes it usable at all.

    Trained ONLY on rows where the primary fired. Training on every row would
    make it a second side model rather than a filter.
    """
    fired = (primary_proba >= threshold) & primary_proba.notna() & y.notna()
    n_fired = int(fired.sum())
    if n_fired < MIN_FOLD_TRAIN_ROWS:
        raise ValueError(
            f"primary model fired on only {n_fired} rows; need {MIN_FOLD_TRAIN_ROWS} "
            f"to train a meta-labeller"
        )

    # Was the primary actually right, on the rows where it acted?
    correct = (y.loc[fired] > 0).astype("float64")
    if correct.nunique() < 2:
        raise ValueError(
            "the primary model was uniformly right or wrong on the rows it fired; "
            "there is nothing for a meta-labeller to learn"
        )

    X = np.column_stack(
        [primary_proba.loc[fired].to_numpy(), features.loc[fired].to_numpy(dtype="float64")]
    )
    w = None if sample_weight is None else sample_weight.loc[fired].to_numpy()

    scaler = StandardScaler().fit(X)
    # class_weight balanced: by construction the positive class is thin — if the
    # primary is 52% accurate, only 52% of its calls are positives here.
    model = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=random_state
    )
    model.fit(scaler.transform(X), correct.to_numpy(), sample_weight=w)

    return MetaLabelResult(
        model=model,
        scaler=scaler,
        threshold=threshold,
        n_train=n_fired,
        base_rate=float(correct.mean()),
    )


def baseline_comparison(
    oof: OOFResult,
    meta_oof: pd.Series,
    y: pd.Series,
    *,
    baseline_name: str = "elasticnet",
) -> dict[str, float | None]:
    """Does the stack beat the honest baseline?

    Reported, not asserted. If a five-model stack with a meta-learner cannot
    beat a regularised linear model, that IS the result and it belongs on the
    /model page — a null finding published is worth more than a fabricated edge.
    """
    from sklearn.metrics import log_loss, roc_auc_score

    mask = oof.covered_mask & y.notna().to_numpy() & meta_oof.notna().to_numpy()
    if mask.sum() < 100:
        return {"baseline_auc": None, "stack_auc": None, "improvement": None}

    y_true = y.to_numpy()[mask]
    if len(np.unique(y_true)) < 2:
        return {"baseline_auc": None, "stack_auc": None, "improvement": None}

    base = oof.predictions[baseline_name].to_numpy()[mask]
    stack = meta_oof.to_numpy()[mask]

    result: dict[str, float | None] = {
        "baseline_auc": float(roc_auc_score(y_true, base)),
        "stack_auc": float(roc_auc_score(y_true, stack)),
        "baseline_logloss": float(log_loss(y_true, base)),
        "stack_logloss": float(log_loss(y_true, stack)),
        "n_compared": int(mask.sum()),
    }
    result["improvement"] = result["stack_auc"] - result["baseline_auc"]  # type: ignore[operator]
    return result
