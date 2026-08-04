"""A stack that can actually serve.

`generate_oof` fits base learners inside each fold and throws them away — that
is correct for producing unbiased out-of-fold predictions, but it leaves nothing
to predict with afterwards. The meta-learner expects base-learner OUTPUTS, not
raw features, so handing it a feature row produces either a crash or, worse,
a shape-compatible nonsense answer.

So training has a second, separate step: after the OOF pass establishes how good
the stack is, fit one final set of base learners on ALL the labelled data and
keep them. Those are what serve.

The distinction matters and is easy to get wrong:

    OOF learners    fitted per fold, never see their own test rows.
                    Used ONLY to measure. Discarded.
    Serving learners fitted once on everything. Used ONLY to predict.
                    Never scored — scoring them on data they trained on is
                    exactly the leak the OOF pass exists to avoid.

Reporting the serving learners' accuracy would be reporting in-sample fit. The
honest numbers come from the OOF pass and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from intelligence.models.stack import BaseLearnerSpec, StackResult


@dataclass
class ServingStack:
    """Final base learners + their scalers + the meta-learner.

    Deliberately NOT the object the OOF pass produced. That one holds fold
    models; this one holds models fitted on everything.
    """

    base_models: dict[str, Any]
    base_scalers: dict[str, StandardScaler | None]
    meta_model: Any
    meta_scaler: StandardScaler | None
    learner_order: list[str]
    feature_names: list[str]
    # Recorded so a mismatch at serve time is a loud error rather than a
    # silently-reordered feature vector.
    meta_input_order: list[str] = field(default_factory=list)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Raw features in, calibrated probability out.

        Runs the base learners, assembles their outputs in the SAME column order
        the meta-learner was fitted on, then runs the meta-learner. Column order
        is explicit because a silently transposed pair of base predictions would
        still produce a plausible number.
        """
        x = (
            features[self.feature_names].to_numpy(dtype="float64")
            if isinstance(features, pd.DataFrame)
            else np.asarray(features, dtype="float64")
        )

        base_out: dict[str, np.ndarray] = {}
        for name in self.learner_order:
            model = self.base_models[name]
            scaler = self.base_scalers.get(name)
            xi = scaler.transform(x) if scaler is not None else x
            base_out[name] = model.predict_proba(xi)[:, 1]

        frame = pd.DataFrame(base_out)

        # The meta-learner may have been fitted with extra inputs (regime).
        # Anything it expects and we cannot supply is filled with the neutral
        # 0.5 — and unlike a feature, that IS defensible here: it says "no
        # opinion from this input", which is the truth when it is absent.
        for col in self.meta_input_order:
            if col not in frame.columns:
                frame[col] = 0.5
        frame = frame[self.meta_input_order or self.learner_order]

        m = frame.to_numpy(dtype="float64")
        if self.meta_scaler is not None:
            m = self.meta_scaler.transform(m)
        return self.meta_model.predict_proba(m)


def fit_serving_stack(
    x: pd.DataFrame,
    y: pd.Series,
    stack: StackResult,
    *,
    learners: list[BaseLearnerSpec],
    sample_weight: pd.Series | None = None,
) -> ServingStack:
    """Fit the final base learners on all labelled data.

    These are never scored. Their only job is to produce the inputs the
    meta-learner already knows how to combine.
    """
    labelled = y.notna().to_numpy()
    x_all = x.to_numpy(dtype="float64")[labelled]
    y_all = y.to_numpy(dtype="float64")[labelled]
    w = None if sample_weight is None else sample_weight.to_numpy()[labelled]

    if np.unique(y_all).size < 2:
        raise ValueError(
            "cannot fit a serving stack on a single-class target — "
            "there is nothing to separate"
        )

    base_models: dict[str, Any] = {}
    base_scalers: dict[str, StandardScaler | None] = {}
    order: list[str] = []

    for spec in learners:
        scaler = StandardScaler().fit(x_all) if spec.needs_scaling else None
        xi = scaler.transform(x_all) if scaler is not None else x_all
        model = spec.factory()
        try:
            model.fit(xi, y_all, sample_weight=w)
        except TypeError:
            model.fit(xi, y_all)
        base_models[spec.name] = model
        base_scalers[spec.name] = scaler
        order.append(spec.name)

    return ServingStack(
        base_models=base_models,
        base_scalers=base_scalers,
        meta_model=stack.meta_model,
        meta_scaler=stack.scaler,
        learner_order=order,
        feature_names=list(x.columns),
        meta_input_order=list(stack.learner_names),
    )
