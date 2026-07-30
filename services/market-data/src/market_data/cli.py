"""market-data CLI.

    indicant-md backfill --from 2006-01-01 --to 2026-07-30
    indicant-md ingest --date 2026-07-30
    indicant-md validate --date 2026-07-30
    indicant-md calendar --from 2015-01-01 --to 2015-12-31 --learn
    indicant-md universe --as-of 2026-07-30
    indicant-md continuity --from 2006-01-01
    indicant-md quarantine [--date ...]
    indicant-md status

Every subcommand exits non-zero on a real problem so a nightly cron fails
loudly rather than logging a warning nobody reads.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta

from indicant_contracts import Verdict

from market_data.adjust.factors import continuity_breaks
from market_data.ingest.bhavcopy import HttpBhavcopyFetcher, LocalBhavcopyFetcher
from market_data.ingest.calendar import TradingCalendarService
from market_data.pipeline import DayResult, IngestPipeline
from market_data.quality.quarantine import QuarantineStore
from market_data.settings import get_settings
from market_data.store.catalog import Catalog
from market_data.store.lake import Lake


def _iso(value: str) -> date:
    return date.fromisoformat(value)


def _build_pipeline(args: argparse.Namespace) -> tuple[IngestPipeline, Lake]:
    settings = get_settings()
    lake = Lake(settings.paths)
    fetcher = (
        LocalBhavcopyFetcher(args.local)
        if getattr(args, "local", None)
        else HttpBhavcopyFetcher(
            timeout=settings.request_timeout,
            max_retries=settings.max_retries,
            backoff=settings.retry_backoff,
            rate_limit_seconds=settings.rate_limit_seconds,
        )
    )
    calendar = TradingCalendarService.from_file()
    return IngestPipeline(lake=lake, fetcher=fetcher, calendar=calendar), lake


def _report(result: DayResult) -> None:
    if result.is_holiday:
        return
    if result.error:
        print(f"  {result.trade_date}  ERROR    {result.error}", file=sys.stderr)
        return
    marker = {
        Verdict.PASS: "ok",
        Verdict.PASS_WITH_WARNINGS: "warn",
        Verdict.QUARANTINED: "quar",
        Verdict.REJECTED: "REJECT",
    }.get(result.verdict, "?")
    print(
        f"  {result.trade_date}  {marker:<7} "
        f"{result.rows_accepted:>6} rows"
        + (f"  ({result.rows_quarantined} held)" if result.rows_quarantined else "")
    )


def cmd_backfill(args: argparse.Namespace) -> int:
    pipeline, lake = _build_pipeline(args)
    print(f"backfilling {args.start} -> {args.end} into {lake.paths.root}")
    result = pipeline.backfill(
        start=args.start, end=args.end, resume=not args.no_resume, progress=_report
    )
    print(result.summary())
    for bad in result.rejected:
        print(f"  REJECTED {bad.trade_date}: {bad.error}", file=sys.stderr)
    return 1 if result.errors or result.rejected else 0


def cmd_ingest(args: argparse.Namespace) -> int:
    pipeline, _ = _build_pipeline(args)
    result = pipeline.ingest_day(args.date)
    _report(result)
    if result.is_holiday:
        print(f"{args.date}: no file published (holiday or not yet available)")
        return 0
    return 0 if result.ingested else 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Run the gate without writing. Dry-run for a suspicious date."""
    pipeline, _ = _build_pipeline(args)
    result = pipeline.ingest_day(args.date, write=False)
    _report(result)
    return 0 if result.ingested or result.is_holiday else 1


def cmd_calendar(args: argparse.Namespace) -> int:
    settings = get_settings()
    lake = Lake(settings.paths)
    calendar = TradingCalendarService.from_file()

    observed = lake.observed_trading_days()
    rec = calendar.reconcile(start=args.start, end=args.end, observed=observed)
    print(f"coverage {rec.coverage:.4f} over {args.start}..{args.end}")
    print(f"  missing:    {len(rec.missing_trading_days)}")
    print(f"  unexpected: {len(rec.unexpected_trading_days)}")
    if rec.uncurated_years:
        print(f"  uncurated years (weekends-only prediction): {list(rec.uncurated_years)}")
    for d in rec.missing_trading_days[:20]:
        print(f"    missing {d}")

    if args.learn:
        years = sorted({d.year for d in observed})
        for year in years:
            try:
                derived = calendar.learn_from_observed(observed, year=year)
                print(f"  learned {len(derived)} holidays for {year}")
            except ValueError as exc:
                print(f"  skipped {year}: {exc}")
        path = calendar.save()
        print(f"  wrote {path}")
    return 0


def cmd_universe(args: argparse.Namespace) -> int:
    settings = get_settings()
    lake = Lake(settings.paths)
    catalog = Catalog(lake, settings.eligibility)
    snapshot = catalog.universe_as_of(args.as_of)
    catalog.write_universe(snapshot)
    catalog.write_symbol_registry(as_of=args.as_of)

    delisted = catalog.delisted_symbols(as_of=args.as_of)
    print(f"universe as of {args.as_of}")
    print(f"  total:    {snapshot.total}")
    print(f"  eligible: {snapshot.eligible_count}")
    print(f"  delisted in history: {len(delisted)}")

    # Only meaningful over a long window. On a two-month lake, zero delistings
    # is the expected outcome, and warning about it every run trains the reader
    # to ignore the warning that matters.
    observed = lake.observed_trading_days()
    span_days = (observed[-1] - observed[0]).days if len(observed) > 1 else 0
    if not delisted and snapshot.total and span_days > 365:
        print(
            f"  WARNING: no delisted names across {span_days} days of history. "
            "The lake is only seeing survivors — the universe is "
            "survivorship-biased.",
            file=sys.stderr,
        )
    for symbol, reason in list(snapshot.excluded.items())[:10]:
        print(f"    excluded {symbol}: {reason}")
    return 0


def cmd_continuity(args: argparse.Namespace) -> int:
    """The Tier-4 sweep across the whole lake.

    This is the check that decides whether 20 years of corporate-action
    adjustment is actually correct.
    """
    settings = get_settings()
    lake = Lake(settings.paths)
    prices = lake.read_prices(start=args.start, end=args.end)
    if prices.empty:
        print("no prices in range")
        return 0

    breaks = continuity_breaks(prices)
    unexplained = breaks[~breaks["explained"]] if not breaks.empty else breaks
    print(f"continuity sweep {args.start}..{args.end}")
    print(f"  rows checked: {len(prices):,}")
    print(f"  breaks:       {len(breaks)}")
    print(f"  unexplained:  {len(unexplained)}")
    for _, row in unexplained.head(20).iterrows():
        print(
            f"    {row['symbol']:<12} {row['date']}  "
            f"prev_close={row['prev_close']:.2f} prior_close={row['prior_close']:.2f} "
            f"ratio={row['ratio']:.4f}"
        )
    return 1 if len(unexplained) else 0


def cmd_quarantine(args: argparse.Namespace) -> int:
    settings = get_settings()
    store = QuarantineStore(Lake(settings.paths))
    held = store.rule_ids_held(args.date)
    if not held:
        print("nothing quarantined")
        return 0
    print("rows held, by rule:")
    for rule_id, count in sorted(held.items(), key=lambda kv: -kv[1]):
        print(f"  {count:>6}  {rule_id}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    settings = get_settings()
    lake = Lake(settings.paths)
    days = lake.observed_trading_days()
    print(f"lake: {lake.paths.root}")
    if not days:
        print("  empty — run: indicant-md backfill --from 2006-01-01")
        return 0
    print(f"  rows:          {lake.row_count():,}")
    print(f"  trading days:  {len(days):,}")
    print(f"  range:         {days[0]} .. {days[-1]}")
    span = lake.symbol_span()
    print(f"  symbols:       {len(span):,}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="indicant-md", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def add_local(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--local",
            type=str,
            default=None,
            help="read from a directory of downloaded files instead of the archive",
        )

    p = sub.add_parser("backfill", help="ingest a date range, one parquet per year")
    p.add_argument("--from", dest="start", type=_iso, default=date(2006, 1, 1))
    p.add_argument("--to", dest="end", type=_iso, default=date.today())
    p.add_argument("--no-resume", action="store_true")
    add_local(p)
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("ingest", help="ingest one day (nightly path)")
    p.add_argument("--date", type=_iso, default=date.today() - timedelta(days=1))
    add_local(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("validate", help="run the gate without writing")
    p.add_argument("--date", type=_iso, required=True)
    add_local(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("calendar", help="reconcile predicted vs observed trading days")
    p.add_argument("--from", dest="start", type=_iso, default=date(2006, 1, 1))
    p.add_argument("--to", dest="end", type=_iso, default=date.today())
    p.add_argument("--learn", action="store_true", help="derive holidays from observed days")
    p.set_defaults(func=cmd_calendar)

    p = sub.add_parser("universe", help="build and write the point-in-time universe")
    p.add_argument("--as-of", dest="as_of", type=_iso, default=date.today())
    p.set_defaults(func=cmd_universe)

    p = sub.add_parser("continuity", help="Tier-4 sweep across the lake")
    p.add_argument("--from", dest="start", type=_iso, default=None)
    p.add_argument("--to", dest="end", type=_iso, default=None)
    p.set_defaults(func=cmd_continuity)

    p = sub.add_parser("quarantine", help="what is held, and by which rule")
    p.add_argument("--date", type=_iso, default=None)
    p.set_defaults(func=cmd_quarantine)

    p = sub.add_parser("status", help="lake summary")
    p.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
