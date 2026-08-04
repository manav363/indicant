"""intelligence CLI.

    indicant-ml train   --symbols 60 --from 2022-01-01
    indicant-ml predict RELIANCE
    indicant-ml screen  --top 20
    indicant-ml status

Exits non-zero on a real problem so a scheduled run fails loudly rather than
logging a warning nobody reads.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from indicant_contracts import LakePaths

from intelligence.data.lake_client import LakeClient
from intelligence.serving import ModelNotTrained, PredictionService, load_model
from intelligence.train import TrainConfig, Trainer, save_artifact


def _lake_root() -> Path:
    return Path(os.environ.get("INDICANT_LAKE_ROOT", "./data/lake")).expanduser().resolve()


def _client() -> LakeClient:
    return LakeClient(LakePaths(root=_lake_root()))


def _artifact_path() -> Path:
    return _lake_root() / "models" / "current.pkl"


def _iso(v: str) -> date:
    return date.fromisoformat(v)


def cmd_train(args: argparse.Namespace) -> int:
    client = _client()
    if not client.is_ready:
        print("the lake has no price data; run 'indicant-md backfill' first",
              file=sys.stderr)
        return 1

    trainer = Trainer(client)
    result, artifact = trainer.run(
        TrainConfig(
            start=args.start,
            end=args.end,
            horizon_days=args.horizon_days,
            n_splits=args.splits,
            n_permutations=args.permutations,
            max_symbols=args.symbols,
        )
    )

    print(result.summary())

    path = save_artifact(artifact, _artifact_path())
    print(f"\nartifact -> {path}")

    # The run is recorded whatever it says. A training pipeline that only
    # persists good results is not a record, it is a highlight reel.
    report = _lake_root() / "models" / f"{result.run_id}.json"
    report.write_text(
        json.dumps(
            {
                "run_id": result.run_id,
                "n_panel_rows": result.n_panel_rows,
                "n_symbols": result.n_symbols,
                "n_features": result.n_features,
                "n_labelled": result.n_labelled,
                "effective_n": result.effective_n,
                "label_distribution": result.label_distribution,
                "oof_coverage": result.oof_coverage,
                "baseline": result.baseline,
                "calibration": {
                    k: v for k, v in result.calibration.items() if k != "bins"
                },
                "calibration_bins": [
                    {
                        "bin_lower": b.bin_lower,
                        "bin_upper": b.bin_upper,
                        "mean_predicted": b.mean_predicted,
                        "observed_rate": b.observed_rate,
                        "count": b.count,
                    }
                    for b in result.calibration.get("bins", [])
                ],
                "permutation": result.permutation,
                "verdict": result.verdict,
            },
            indent=2,
            default=str,
        )
    )
    print(f"report   -> {report}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    try:
        model = load_model(_artifact_path())
    except ModelNotTrained as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    service = PredictionService(_client(), model)
    try:
        p = service.predict(args.symbol, horizon_months=args.horizon)
    except (ModelNotTrained, KeyError) as exc:
        print(f"{args.symbol}: {exc}", file=sys.stderr)
        return 1

    print(f"{p.symbol}  {p.signal.value}  ({p.strength.value})")
    print(f"  P(up in {p.horizon_months}m)  {p.probability_up:.4f}")
    print(f"  confidence         {p.confidence:.4f}")
    print(f"  price              Rs {p.current_price:,.2f}")
    print(f"  suggested size     {p.suggested_position_pct:.2f}%")
    if p.regime:
        print(f"  regime             {p.regime.value}")
    if p.facts:
        print("  drivers:")
        for f in p.facts:
            arrow = "up  " if f.shap > 0 else "down"
            print(f"    {arrow}  {f.display_name}: {f.display_value}  (shap {f.shap:+.4f})")
    return 0


def cmd_screen(args: argparse.Namespace) -> int:
    try:
        model = load_model(_artifact_path())
    except ModelNotTrained as exc:
        print(f"{exc}", file=sys.stderr)
        return 1

    client = _client()
    service = PredictionService(client, model)
    as_of = (client.trading_days() or [date.today()])[-1]
    symbols = client.eligible_symbols(as_of)[: args.universe]

    rows = service.screen(symbols, horizon_months=args.horizon, top=args.top)
    if not rows:
        print("nothing to show — no symbol could be predicted", file=sys.stderr)
        return 1

    print(f"{'symbol':<14}{'signal':<8}{'P(up)':>8}{'conf':>8}{'price':>12}")
    for r in rows:
        print(
            f"{r['symbol']:<14}{r['signal']:<8}"
            f"{r['probability_up']:>8.4f}{r['confidence']:>8.4f}"
            f"{r['current_price']:>12,.2f}"
        )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    client = _client()
    days = client.trading_days()
    print(f"lake:   {_lake_root()}")
    if not days:
        print("  empty — run 'indicant-md backfill'")
        return 0
    print(f"  rows          {client.row_count(adjusted=False):,}")
    print(f"  trading days  {len(days):,}  ({days[0]} .. {days[-1]})")
    print(f"  eligible      {len(client.eligible_symbols(days[-1])):,} symbols")

    path = _artifact_path()
    if path.exists():
        m = load_model(path)
        sig = m.is_significant
        print(f"model:  {m.run_id}  trained {m.trained_at}")
        print(f"  features      {len(m.feature_names)}")
        print(
            "  significance  "
            + ("not tested" if sig is None else ("significant" if sig else "NOT significant"))
            + (f" (p={m.p_value:.4f})" if m.p_value is not None else "")
        )
    else:
        print("model:  none trained yet")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="indicant-ml", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="build the panel, label, fit the stack, validate")
    t.add_argument("--from", dest="start", type=_iso, default=None)
    t.add_argument("--to", dest="end", type=_iso, default=None)
    t.add_argument("--symbols", type=int, default=60,
                   help="cap on universe size (speed, not modelling)")
    t.add_argument("--horizon-days", type=int, default=126)
    t.add_argument("--splits", type=int, default=5)
    t.add_argument("--permutations", type=int, default=50)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="one symbol")
    pr.add_argument("symbol")
    pr.add_argument("--horizon", type=int, default=6)
    pr.set_defaults(func=cmd_predict)

    sc = sub.add_parser("screen", help="rank the universe")
    sc.add_argument("--top", type=int, default=20)
    sc.add_argument("--universe", type=int, default=60)
    sc.add_argument("--horizon", type=int, default=6)
    sc.set_defaults(func=cmd_screen)

    st = sub.add_parser("status", help="lake and model summary")
    st.set_defaults(func=cmd_status)

    return p


def main(argv: list[str] | None = None) -> int:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
