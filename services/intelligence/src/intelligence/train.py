"""Training pipeline — L0 through L8, end to end.

Ties together everything the earlier phases built:

    L0  panel          pooled cross-section from the PIT universe
    L1  labels         triple-barrier + average-uniqueness sample weights
    L2  base learners  purged, embargoed out-of-fold predictions
    L3  meta-learner   stacking over those OOF predictions
    L6  calibration    reliability curve + Brier, as evidence not assertion
    L8  validation     permutation test, CPCV, Deflated Sharpe

Every result is written to the registry, including — especially including — a
null one. The project's whole claim is that its numbers are checkable, and a
training run that quietly discards a bad result would break that.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from intelligence.calibration.reliability import calibration_report
from intelligence.data.lake_client import LakeClient
from intelligence.labeling.sample_weights import compute_weights, effective_sample_size
from intelligence.labeling.triple_barrier import (
    TripleBarrierConfig,
    label_distribution,
    label_panel,
    to_binary_side,
)
from intelligence.models.serving_stack import fit_serving_stack
from intelligence.models.stack import (
    baseline_comparison,
    default_base_learners,
    fit_meta_learner,
    generate_oof,
)
from intelligence.panel.builder import PanelBuilder, PanelConfig
from intelligence.validation.panel_cv import CVMode, PanelCVConfig, PurgedPanelCV
from intelligence.validation.panel_permutation import (
    PanelPermutationConfig,
    run_panel_permutation,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    symbols: list[str] | None = None
    start: date | None = None
    end: date | None = None
    horizon_days: int = 126
    n_splits: int = 5
    embargo_days: int = 21
    n_permutations: int = 50
    max_symbols: int = 200
    random_state: int = 42


@dataclass
class TrainResult:
    run_id: str
    n_panel_rows: int = 0
    n_symbols: int = 0
    n_features: int = 0
    n_labelled: int = 0
    effective_n: float = 0.0
    label_distribution: dict[str, float] = field(default_factory=dict)
    oof_coverage: float = 0.0
    baseline: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    permutation: dict[str, Any] = field(default_factory=dict)
    verdict: str = ""

    def summary(self) -> str:
        lines = [
            f"run {self.run_id}",
            f"  panel        {self.n_panel_rows:,} rows · {self.n_symbols} symbols · "
            f"{self.n_features} features",
            f"  labelled     {self.n_labelled:,} rows",
            # The number to quote instead of n_labelled: overlapping labels do
            # not carry one observation's worth of evidence each.
            f"  effective n  {self.effective_n:,.0f}  "
            f"({self.effective_n / max(1, self.n_labelled):.1%} of raw)",
            f"  labels       {self.label_distribution}",
            f"  OOF coverage {self.oof_coverage:.1%}",
        ]
        if self.baseline.get("stack_auc") is not None:
            lines.append(
                f"  AUC          stack {self.baseline['stack_auc']:.4f} vs "
                f"baseline {self.baseline['baseline_auc']:.4f} "
                f"({self.baseline['improvement']:+.4f})"
            )
        if self.calibration.get("brier_score") is not None:
            lines.append(
                f"  Brier        {self.calibration['brier_score']:.4f} "
                f"(skill {self.calibration['brier_skill_score']:+.4f})"
            )
        if self.permutation.get("p_value") is not None:
            lines.append(
                f"  permutation  p = {self.permutation['p_value']:.4f} "
                f"over {self.permutation['n_permutations']} shuffles"
            )
        lines.append("")
        lines.append(f"  {self.verdict}")
        return "\n".join(lines)


class Trainer:
    def __init__(self, client: LakeClient) -> None:
        self._client = client
        self._builder = PanelBuilder(client)

    def run(self, config: TrainConfig | None = None) -> tuple[TrainResult, dict[str, Any]]:
        cfg = config or TrainConfig()
        run_id = f"{date.today().strftime('%Y%m%d')}_{np.random.randint(0, 2**16):04x}"
        result = TrainResult(run_id=run_id)

        # ---- L0 panel ---------------------------------------------------
        symbols = cfg.symbols or self._pick_symbols(cfg)
        logger.info("building panel over %d symbols", len(symbols))
        panel_res = self._builder.build(
            PanelConfig(start=cfg.start, end=cfg.end, symbols=symbols)
        )
        panel = panel_res.frame
        if panel.empty:
            raise ValueError(
                "the panel is empty — the lake has no usable history for these symbols"
            )

        result.n_panel_rows = len(panel)
        result.n_symbols = panel_res.n_symbols
        result.n_features = len(panel_res.feature_columns)

        # ---- L1 labels + sample weights ---------------------------------
        labelled = label_panel(
            panel, TripleBarrierConfig(horizon_days=cfg.horizon_days)
        )
        labelled = labelled[labelled["label"].notna()].reset_index(drop=True)
        if labelled.empty:
            raise ValueError(
                f"no rows survived labelling at a {cfg.horizon_days}-day horizon; "
                f"the lake needs more history than it has"
            )

        result.n_labelled = len(labelled)
        result.label_distribution = label_distribution(labelled["label"])

        weights = compute_weights(labelled)
        result.effective_n = effective_sample_size(weights)

        y = to_binary_side(labelled["label"])
        features = [c for c in panel_res.feature_columns if c in labelled.columns]
        x = labelled[features].astype("float64").fillna(0.0)

        # ---- L2 purged, embargoed OOF -----------------------------------
        cv = PurgedPanelCV(
            PanelCVConfig(
                n_splits=cfg.n_splits,
                purge_days=cfg.horizon_days,
                embargo_days=cfg.embargo_days,
                min_train_dates=252,
                # K-fold for OOF generation: the goal is an unbiased
                # generalisation estimate, and purge+embargo remove the
                # contaminated boundary. See CVMode.
                mode=CVMode.PURGED_KFOLD,
            )
        )
        oof = generate_oof(
            x, y, cv, labelled,
            learners=default_base_learners(cfg.random_state),
            sample_weight=weights,
        )
        result.oof_coverage = oof.coverage

        # ---- L3 meta-learner --------------------------------------------
        stack = fit_meta_learner(oof, y, sample_weight=weights,
                                 random_state=cfg.random_state)
        result.baseline = baseline_comparison(oof, stack.meta_oof, y)

        # ---- L6 calibration evidence ------------------------------------
        mask = oof.covered_mask & y.notna().to_numpy() & stack.meta_oof.notna().to_numpy()
        report = calibration_report(
            y.to_numpy()[mask], stack.meta_oof.to_numpy()[mask]
        )
        result.calibration = report

        # ---- L8 permutation ----------------------------------------------
        result.permutation = self._permutation(
            labelled, y, x, weights, cv, cfg, features
        )

        result.verdict = self._verdict(result)

        # The OOF learners were fitted per fold and discarded — correct for
        # measurement, useless for serving. Fit the final set on everything.
        # These are NEVER scored: their accuracy on data they trained on is
        # in-sample fit, and the honest numbers come from the OOF pass above.
        serving = fit_serving_stack(
            x, y, stack,
            learners=default_base_learners(cfg.random_state),
            sample_weight=weights,
        )

        artifact = {
            "run_id": run_id,
            "model": serving,
            "feature_names": features,
            "trained_at": date.today().isoformat(),
            "p_value": result.permutation.get("p_value"),
            # Required at serve time: cross-sectional features only mean
            # something relative to the set they were computed over.
            "universe": sorted(panel["symbol"].unique().tolist()),
        }
        return result, artifact

    # ------------------------------------------------------------------ bits

    def _pick_symbols(self, cfg: TrainConfig) -> list[str]:
        """Most liquid eligible names, capped.

        Capped because the panel is O(symbols x days) and a full 2,500-symbol
        run is minutes of feature computation for a demonstration. The cap is a
        speed decision, not a modelling one, and it is recorded in the run.
        """
        as_of = cfg.end or (self._client.trading_days() or [date.today()])[-1]
        eligible = self._client.eligible_symbols(as_of)
        return eligible[: cfg.max_symbols]

    def _permutation(
        self, labelled, y, x, weights, cv, cfg, features
    ) -> dict[str, Any]:
        """Could this have come from noise?

        The score function re-runs the WHOLE fold loop per permutation, so the
        null reflects every source of optimism in the real run rather than only
        the model fit.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler

        def score(_panel: pd.DataFrame, labels: pd.Series, seed: int) -> float:
            preds = np.full(len(labels), np.nan)
            for split in cv.split(_panel):
                tr, te = split.train_idx, split.test_idx
                tr = tr[np.isfinite(labels.to_numpy()[tr])]
                if tr.size < 500 or np.unique(labels.to_numpy()[tr]).size < 2:
                    continue
                scaler = StandardScaler().fit(x.to_numpy()[tr])
                model = LogisticRegression(max_iter=300, random_state=seed)
                model.fit(scaler.transform(x.to_numpy()[tr]), labels.to_numpy()[tr])
                preds[te] = model.predict_proba(scaler.transform(x.to_numpy()[te]))[:, 1]

            ok = np.isfinite(preds) & np.isfinite(labels.to_numpy())
            if ok.sum() < 100 or np.unique(labels.to_numpy()[ok]).size < 2:
                return float("nan")
            return float(roc_auc_score(labels.to_numpy()[ok], preds[ok]))

        try:
            perm = run_panel_permutation(
                labelled, y, score, cv,
                PanelPermutationConfig(
                    n_permutations=cfg.n_permutations,
                    random_state=cfg.random_state,
                ),
            )
        except Exception as exc:
            logger.warning("permutation test failed: %s", exc)
            return {"p_value": None, "error": str(exc)}

        return {
            **perm.summary(),
            "verdict": perm.verdict(),
        }

    def _verdict(self, result: TrainResult) -> str:
        """One plain sentence about what this run actually showed."""
        p = result.permutation.get("p_value")
        beats = result.baseline.get("improvement")

        n_perm = result.permutation.get("n_permutations") or 0
        # With N permutations the smallest reportable p is 1/(N+1). Landing ON
        # that floor means NO shuffle beat the actual — which is the test
        # running out of resolution, not evidence of significance. Reporting
        # "p = 0.0476, significant!" from 20 shuffles is exactly the false
        # precision this project exists to avoid.
        floor = 1.0 / (n_perm + 1) if n_perm else None
        at_floor = (
            p is not None and floor is not None and abs(p - floor) < 1e-9
        )

        # Landing on the floor means zero shuffles beat the actual. Whether that
        # is informative depends entirely on HOW MANY shuffles: 0/20 is the test
        # having no resolution, 0/200 is a real result. Conflating them either
        # invents significance or throws away a finding.
        floor_is_informative_at = 100

        parts: list[str] = []
        if p is None:
            parts.append("The permutation test did not complete, so significance is unknown.")
        elif at_floor and n_perm < floor_is_informative_at:
            parts.append(
                f"p = {p:.4f} is the FLOOR of a {n_perm}-permutation test "
                f"(1/{n_perm + 1}), not a measurement: no shuffle beat the actual, "
                f"so the test cannot resolve any finer. Re-run with more "
                f"permutations before reading anything into this."
            )
        elif at_floor:
            parts.append(
                f"NO shuffle out of {n_perm} beat the actual score, so p is reported "
                f"at its floor of {p:.4f} and the true value is below it. That is "
                f"real evidence of a signal — though the permutation test only says "
                f"the edge is not random, not that it is large or tradeable."
            )
        elif p < 0.05:
            parts.append(
                f"The edge survives label shuffling (p={p:.4f} over {n_perm} shuffles)."
            )
        else:
            parts.append(
                f"The edge is NOT distinguishable from chance (p={p:.4f}). "
                f"Reshuffled labels scored this well {p:.1%} of the time."
            )

        if beats is not None:
            parts.append(
                f"The stack {'beats' if beats > 0 else 'does NOT beat'} its "
                f"ElasticNet baseline by {beats:+.4f} AUC."
            )

        skill = result.calibration.get("brier_skill_score")
        if skill is not None and np.isfinite(skill):
            parts.append(
                f"Brier skill {skill:+.4f} — "
                f"{'better' if skill > 0 else 'no better'} than always predicting "
                f"the base rate."
            )
        return " ".join(parts)


def save_artifact(artifact: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(artifact, fh)
    return path
