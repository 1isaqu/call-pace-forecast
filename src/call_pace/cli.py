"""Command line interface: `call-pace <command>` (or `python -m call_pace <command>`).

    synth        write an invented dataset so everything below can run with nothing connected
    train        fit lookup, KNN and quantile models (and a ticket lookup) into an artifacts folder
    validate     walk-forward evaluation, promotion decision, quantile-band coverage
    projection   compare share-formula and neighbour projections of the end-of-day total
    ablate       feature subsets, lookup variants or a covariate signal report
    montecarlo   stress-test a rolling retraining policy under volume growth
    pace         print today's expected/actual/delta table as JSON
    serve        run the HTTP endpoint

Data comes from `--source csv` (your files, see docs/data-contract.md) or `--source sqlserver`
(DB_* environment variables, see .env.example).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ._version import __version__
from .ablation import (
    compare_lookup_variants,
    covariate_signal_report,
    feature_subset_search,
)
from .artifacts import ArtifactStore
from .config import Config
from .contract import AmountLoader, HourlyLoader, validate_amounts, validate_hourly
from .errors import CallPaceError, ConfigError
from .features import FeatureSpec, build_frame
from .loaders import CsvLoader
from .loaders.sqlserver import SqlOptions, loader_from_env
from .montecarlo import MonteCarloConfig, run_monte_carlo
from .projection import compare_share_keys, evaluate_projection
from .service import PaceService
from .synthetic import SyntheticSpec, make_synthetic_amounts, make_synthetic_hourly
from .training import train_models
from .validation import (
    PromotionCriteria,
    compare_models,
    decide_promotion,
    markdown_table,
    run_walk_forward,
)

EARLIEST, LATEST = date(1900, 1, 1), date(2200, 1, 1)


def _iso_date(text: str) -> date:
    try:
        return date.fromisoformat(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{text!r} is not an ISO date (YYYY-MM-DD)") from exc


def _add_data_options(parser: argparse.ArgumentParser, amounts: bool = True) -> None:
    parser.add_argument("--source", choices=("csv", "sqlserver"), default="csv")
    parser.add_argument("--csv", help="hourly counts file: bucket,date,hour,count[,covariates]")
    if amounts:
        parser.add_argument("--amounts-csv", help="event amounts file: bucket,date,hour,amount")
    parser.add_argument("--metric", default="agreements", help="sqlserver: what to count")
    parser.add_argument("--start", type=_iso_date, help="first day to read (inclusive)")
    parser.add_argument("--end", type=_iso_date, help="last day to read (inclusive)")
    parser.add_argument("--covariate", action="append", default=[], help="exogenous column name")
    parser.add_argument("--artifacts", help="directory for fitted models (default: artifacts)")
    parser.add_argument(
        "--fill-missing-days",
        action="store_true",
        help="treat operating weekdays with no rows as zero-activity days",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="call-pace", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"call-pace {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="write an invented dataset")
    synth.add_argument("--out", default="data/hourly.csv")
    synth.add_argument("--amounts-out", help="also write event amounts here")
    synth.add_argument("--days", type=int, default=120)
    synth.add_argument("--seed", type=int, default=7)
    synth.add_argument("--start", type=_iso_date, default=date(2026, 1, 5))
    synth.add_argument("--with-covariate", action="store_true")

    train = sub.add_parser("train", help="fit models into the artifacts folder")
    _add_data_options(train)
    train.add_argument("--models", default="lookup,knn,quantile")
    train.add_argument("--k", type=int, help="fix k instead of selecting it by cross-validation")

    validate = sub.add_parser("validate", help="walk-forward evaluation and promotion decision")
    _add_data_options(validate, amounts=False)
    validate.add_argument("--challengers", default="knn,quantile")
    validate.add_argument("--per-bucket", action="store_true", help="also decide per bucket")
    validate.add_argument("--report", help="also write the markdown report to this file")

    projection = sub.add_parser("projection", help="end-of-day projection comparison")
    _add_data_options(projection, amounts=False)
    projection.add_argument("--holdout-days", type=int, default=20)
    projection.add_argument("--k", type=int, default=10)
    projection.add_argument(
        "--extra-key",
        action="append",
        default=[],
        help="also test whether adding this column (for example weekday_pos) to the "
        "(bucket, hour) share-curve key earns its thinner cells",
    )

    ablate = sub.add_parser("ablate", help="ablation studies")
    _add_data_options(ablate, amounts=False)
    ablate.add_argument("what", choices=("features", "lookup", "covariate"))
    ablate.add_argument("--top", type=int, default=10)

    mc = sub.add_parser("montecarlo", help="Monte Carlo retraining stress test")
    _add_data_options(mc, amounts=False)
    defaults = MonteCarloConfig()
    mc.add_argument("--scenarios", type=int, default=defaults.n_scenarios)
    mc.add_argument("--months", type=int, default=defaults.n_months)
    mc.add_argument("--seed", type=int, default=defaults.seed)
    mc.add_argument(
        "--growth",
        type=float,
        nargs=3,
        metavar=("LOW", "MODE", "HIGH"),
        default=(defaults.growth_low, defaults.growth_mode, defaults.growth_high),
    )
    mc.add_argument("--window", type=int, default=defaults.window_days)
    mc.add_argument("--k", type=int, default=defaults.k)
    mc.add_argument("--out", help="write every (scenario, month) result to this csv")

    pace = sub.add_parser("pace", help="print the intraday pace table as JSON")
    _add_data_options(pace)
    pace.add_argument("--bucket", default="all")
    pace.add_argument("--now", type=datetime.fromisoformat, help="override the clock (ISO time)")

    serve = sub.add_parser("serve", help="run the HTTP endpoint")
    _add_data_options(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


# --- shared plumbing ------------------------------------------------------------------------


def _config(args: argparse.Namespace) -> Config:
    config = Config.from_env()
    changes: dict = {}
    if getattr(args, "covariate", None):
        changes["covariates"] = tuple(args.covariate)
    if getattr(args, "artifacts", None):
        changes["artifacts_dir"] = args.artifacts
    return config.replace(**changes) if changes else config


def _loaders(args: argparse.Namespace, config: Config) -> tuple[HourlyLoader, AmountLoader | None]:
    if args.source == "sqlserver":
        loader = loader_from_env(os.environ, config, SqlOptions(metric=args.metric))
        return loader, loader
    if not args.csv:
        raise ConfigError("--csv is required with --source csv")
    amounts = getattr(args, "amounts_csv", None)
    loader = CsvLoader(args.csv, amounts)
    return loader, (loader if amounts else None)


def _window(args: argparse.Namespace) -> tuple[date, date]:
    if args.source == "sqlserver":
        end = args.end or date.today() - timedelta(days=1)
        return args.start or end - timedelta(days=120), end
    return args.start or EARLIEST, args.end or LATEST


def _read(args: argparse.Namespace, config: Config) -> pd.DataFrame:
    loader, _ = _loaders(args, config)
    start, end = _window(args)
    return validate_hourly(loader.load_hourly(start, end), config)


def _read_frame(args: argparse.Namespace, config: Config) -> tuple[FeatureSpec, pd.DataFrame]:
    hourly = _read(args, config)
    spec = FeatureSpec.from_data(hourly, config)
    return spec, build_frame(hourly, spec, fill_missing_days=args.fill_missing_days)


def _split(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


# --- commands -------------------------------------------------------------------------------


def cmd_synth(args: argparse.Namespace) -> int:
    config = Config()
    params = SyntheticSpec(covariate_effect=0.5 if args.with_covariate else 0.0)
    hourly = make_synthetic_hourly(
        args.days, args.start, args.seed, config, params, args.with_covariate
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(out, index=False)
    print(f"wrote {len(hourly)} hourly rows to {out}")
    if args.amounts_out:
        amounts = make_synthetic_amounts(hourly, args.seed + 1)
        Path(args.amounts_out).parent.mkdir(parents=True, exist_ok=True)
        amounts.to_csv(args.amounts_out, index=False)
        print(f"wrote {len(amounts)} event amounts to {args.amounts_out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    config = _config(args)
    loader, amount_loader = _loaders(args, config)
    start, end = _window(args)
    hourly = validate_hourly(loader.load_hourly(start, end), config)
    amounts = None
    if amount_loader is not None:
        amounts = validate_amounts(amount_loader.load_amounts(start, end), config)
    store = ArtifactStore(config.artifacts_dir, config.artifact_ttl_seconds)
    report = train_models(
        hourly, config, store, amounts, _split(args.models), args.k, args.fill_missing_days
    )
    print(json.dumps(report.to_dict(), indent=2))
    print(f"artifacts written to {store.directory}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    config = _config(args)
    spec, frame = _read_frame(args, config)
    challengers = _split(args.challengers)
    result = run_walk_forward(frame, config, spec, challengers)
    lines = [result.to_markdown(), "", *_decisions(result, challengers, config)]
    if args.per_bucket:
        for bucket in spec.buckets:
            lines += [
                "",
                f"bucket {bucket}",
                *_decisions(result.for_bucket(bucket), challengers, config),
            ]
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


def _decisions(result, challengers: Sequence[str], config: Config) -> list[str]:
    lines = []
    for name in challengers:
        lines.append(decide_promotion(result, name, PromotionCriteria()).explain())
        paired = compare_models(result, name, "lookup")
        lines.append(
            "  paired test vs lookup over all folds: mean absolute-error difference "
            f"{paired.mean_difference:+.3f} (negative favours {name}), "
            f"p={paired.p_value:.4f}"
        )
    if "quantile" in challengers:
        holdout = result.holdout()
        lines.append(
            f"quantile band: nominal coverage {config.nominal_coverage:.0%}, empirical "
            f"{holdout['coverage']:.1%} on the holdout fold, mean width {holdout['width']:.2f}"
        )
    return lines


def cmd_projection(args: argparse.Namespace) -> int:
    config = _config(args)
    _, frame = _read_frame(args, config)
    table = evaluate_projection(
        frame, args.holdout_days, args.k, config.covariates, config.start_hour, config.end_hour
    )
    print(markdown_table(table))
    if args.extra_key:
        keys = ("bucket", "hour", *args.extra_key)
        print()
        print(compare_share_keys(frame, keys, holdout_days=args.holdout_days).explain())
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    config = _config(args)
    spec, frame = _read_frame(args, config)
    if args.what == "features":
        result = feature_subset_search(frame, spec, config)
        print(result.summary())
        print(markdown_table(result.table.head(args.top)))
    elif args.what == "lookup":
        print(markdown_table(compare_lookup_variants(frame, config)))
    else:
        if not config.covariates:
            raise ConfigError("pass --covariate NAME to test a covariate")
        print(covariate_signal_report(frame, config.covariates[0]).conclusion())
    return 0


def cmd_montecarlo(args: argparse.Namespace) -> int:
    config = _config(args)
    spec, frame = _read_frame(args, config)
    low, mode, high = args.growth
    mc = MonteCarloConfig(
        n_scenarios=args.scenarios,
        n_months=args.months,
        growth_low=low,
        growth_mode=mode,
        growth_high=high,
        window_days=args.window,
        k=args.k,
        seed=args.seed,
    )
    result = run_monte_carlo(frame, config, spec, mc)
    print(markdown_table(result.summary()))
    print(
        f"P(directional accuracy > {mc.directional_threshold:.0%} in month {mc.n_months}) = "
        f"{result.probability_directional_above():.1%}"
    )
    beats = result.probability_beats_baseline()
    print(f"P(model beats the lookup in month {mc.n_months}) = {beats:.1%}")
    if args.out:
        result.runs.to_csv(args.out, index=False)
    return 0


def _service(args: argparse.Namespace, config: Config) -> PaceService:
    if config.covariates:
        raise ConfigError(
            "serving models that use covariates needs a Python covariate_provider: "
            "use PaceService from the library; the command line cannot supply one"
        )
    loader, amount_loader = _loaders(args, config)
    store = ArtifactStore(config.artifacts_dir, config.artifact_ttl_seconds)
    clock = (lambda: args.now) if getattr(args, "now", None) else datetime.now
    return PaceService(config, loader, store, amount_loader, clock=clock)


def cmd_pace(args: argparse.Namespace) -> int:
    print(json.dumps(_service(args, _config(args)).pace(args.bucket), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigError("install the optional extra: pip install '.[api]'") from exc
    from .api import create_app

    uvicorn.run(create_app(_service(args, _config(args))), host=args.host, port=args.port)
    return 0


COMMANDS = {
    "synth": cmd_synth,
    "train": cmd_train,
    "validate": cmd_validate,
    "projection": cmd_projection,
    "ablate": cmd_ablate,
    "montecarlo": cmd_montecarlo,
    "pace": cmd_pace,
    "serve": cmd_serve,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except CallPaceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
