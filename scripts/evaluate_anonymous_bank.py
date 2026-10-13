"""Walk-forward evaluation on the public "Anonymous Bank" call-center data, with the figures.

    pip install -e ".[dev,realdata]"
    python scripts/evaluate_anonymous_bank.py --data data --out data/results --figures docs/img

The first run downloads the archive (about 10 MB) from the Technion host and checks every hash;
`data/` is ignored by git. Results are written as JSON, and `--figures` also draws four PNG files.
Besides the lookup baseline of the package, the script reports a lookup keyed by weekday, as a
reference only, so a gain is not measured against a baseline that ignores the day of the week.
The run is deterministic on one platform and library set.
"""

import argparse
import json
import time
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from call_pace import FeatureSpec, build_frame, decide_promotion, run_walk_forward, validate_hourly
from call_pace.anonymous_bank import (
    MONTH_SHA256,
    anonymous_bank_config,
    fetch,
    hourly_counts,
    read_calls,
)
from call_pace.baseline import keyed_lookup_predict
from call_pace.validation import make_folds, split_frame

DAY_NAMES = {6: "Sun", 0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu"}
COLORS = {
    "actual": "#1b1b1b", "lookup": "#b0762c", "model": "#1f6f8b", "band": "#9ccbdc",
    "knn": "#6aa5bf", "grid": "#d9d9d9", "keep": "#c9c9c9",
}  # fmt: skip
NAMES = {
    "PS": "PS regular service", "NW": "NW prospective customers", "NE": "NE stock exchange",
    "TT": "TT bank calls back", "IN": "IN internet help desk", "PE": "PE regular, English",
}  # fmt: skip


def weekday_lookup_errors(frame: pd.DataFrame, config) -> pd.DataFrame:
    """MAE per (bucket, fold) of a lookup keyed by (bucket, weekday, hour), on the same folds."""
    folds = make_folds(
        frame["date"].unique(), config.n_folds, config.fold_size_days, config.min_train_days
    )
    rows = []
    for number, fold in enumerate(folds, start=1):
        train, test = split_frame(frame, fold)
        windows = []
        for _, group in train.groupby("bucket"):
            days = np.sort(group["date"].unique())[-(config.lookup_window_days or len(group)) :]
            windows.append(group[group["date"].isin(days)])
        keys = ["bucket", "weekday_pos", "hour"]
        predicted = keyed_lookup_predict(
            pd.concat(windows), test, keys, config.lookup_stat, config.lookup_treatment,
            config.postprocess,
        )  # fmt: skip
        errors = pd.DataFrame(
            {
                "bucket": test["bucket"].to_numpy(),
                "ae": np.abs(test["count"].to_numpy() - predicted),
            }
        )
        rows.extend(
            {"bucket": b, "fold": number, "mae": float(g["ae"].mean())}
            for b, g in errors.groupby("bucket")
        )
    return pd.DataFrame(rows).pivot(index="bucket", columns="fold", values="mae")


def summarize(result, hourly: pd.DataFrame, weekday_mae: pd.DataFrame) -> list[dict]:
    """One row per bucket: volume, errors, promotion decisions, band coverage."""
    pred = result.predictions
    inside = (pred["actual"] >= pred["lower"]) & (pred["actual"] <= pred["upper"])
    volume = hourly.groupby("bucket")["count"].mean().sort_values(ascending=False)
    rows = []
    for bucket in volume.index:
        part = result.for_bucket(bucket)
        fm = part.fold_metrics
        reference = weekday_mae.loc[bucket]
        rows.append(
            {
                "bucket": bucket,
                "calls_total": int(hourly.loc[hourly["bucket"] == bucket, "count"].sum()),
                "mean_per_hour": round(float(volume[bucket]), 2),
                "mae_lookup": round(float(fm["mae_lookup"].mean()), 3),
                "mae_lookup_weekday": round(float(reference.mean()), 3),
                "mae_knn": round(float(fm["mae_knn"].mean()), 3),
                "mae_quantile": round(float(fm["mae_quantile"].mean()), 3),
                "last_fold_lookup": round(float(fm["mae_lookup"].iloc[-1]), 3),
                "last_fold_lookup_weekday": round(float(reference.iloc[-1]), 3),
                "last_fold_knn": round(float(fm["mae_knn"].iloc[-1]), 3),
                "last_fold_quantile": round(float(fm["mae_quantile"].iloc[-1]), 3),
                "knn": decide_promotion(part, "knn").explain().splitlines()[0],
                "quantile": decide_promotion(part, "quantile").explain().splitlines()[0],
                "band_coverage": round(float(inside[pred["bucket"] == bucket].mean()), 3),
            }
        )
    return rows


def draw_figures(result, hourly: pd.DataFrame, rows: list[dict], target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.6,
            "figure.dpi": 110,
        }  # fmt: skip
    )
    target.mkdir(parents=True, exist_ok=True)
    pred = result.predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    pred["inside"] = (pred["actual"] >= pred["lower"]) & (pred["actual"] <= pred["upper"])
    order = [r["bucket"] for r in rows]
    volume = {r["bucket"]: r["mean_per_hour"] for r in rows}
    work = hourly.assign(weekday=pd.to_datetime(hourly["date"]).dt.weekday)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    profile = (
        work[work["bucket"] == "PS"].groupby(["weekday", "hour"])["count"].mean().reset_index()
    )
    shades = ("#1f6f8b", "#6aa5bf", "#9ccbdc", "#c79a5c", "#b0762c")
    for weekday, color in zip((6, 0, 1, 2, 3), shades, strict=True):
        part = profile[profile["weekday"] == weekday]
        axes[0].plot(part["hour"], part["count"], color=color, lw=2, label=DAY_NAMES[weekday])
    axes[0].set(xlabel="hour of day", ylabel="mean calls per hour")
    axes[0].set_title("PS: the same shape every day, Sunday slightly higher", fontsize=10)
    axes[0].legend(
        frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18)
    )
    for bucket, color in zip(("PS", "NW", "TT", "IN"), ("#1f6f8b", "#b0762c", "#7f7f7f", "#6aa5bf"),
                             strict=True):  # fmt: skip
        daily = (
            work[work["bucket"] == bucket].groupby(["date", "hour"])["count"].sum().unstack("hour")
        )
        share = daily.div(daily.sum(axis=1), axis=0).mean() * 100
        axes[1].plot(share.index, share.values, color=color, lw=2, label=bucket)
    axes[1].set(xlabel="hour of day", ylabel="share of the day's calls (%)")
    axes[1].set_title("Each service type has its own shape", fontsize=10)
    axes[1].legend(
        frameon=False, ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18)
    )
    fig.suptitle("Anonymous Bank call center, 1999: calls arriving per hour, Sunday to Thursday, "
                 "7h to 24h", y=1.03, fontsize=11)  # fmt: skip
    fig.tight_layout()
    fig.savefig(target / "anonymous_bank_1_shape_by_day_and_type.png", bbox_inches="tight")
    plt.close(fig)

    def wednesday(fold):
        days = sorted(pred[pred["fold"] == fold]["date"].unique())
        return next(d for d in days if pd.Timestamp(d).weekday() == 2)

    picks = [(pred["fold"].min(), wednesday(pred["fold"].min())),
             (pred["fold"].max(), wednesday(pred["fold"].max()))]  # fmt: skip
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.4), sharex=True)
    for i, bucket in enumerate(("PS", "IN")):
        for j, (fold, day) in enumerate(picks):
            ax = axes[i, j]
            d = pred[(pred["fold"] == fold) & (pred["bucket"] == bucket) & (pred["date"] == day)]
            d = d.sort_values("hour")
            ax.fill_between(d["hour"], d["lower"], d["upper"], color=COLORS["band"], alpha=0.85,
                            label="quantile band (25-75%)")  # fmt: skip
            ax.plot(
                d["hour"], d["quantile"], color=COLORS["model"], lw=2, label="LightGBM quantile"
            )
            ax.plot(d["hour"], d["lookup"], color=COLORS["lookup"], lw=1.6, ls="--",
                    label="lookup baseline")  # fmt: skip
            ax.plot(d["hour"], d["actual"], color=COLORS["actual"], lw=0, marker="o", ms=4,
                    label="actual")  # fmt: skip
            ax.set_title(f"{NAMES[bucket]}, {pd.Timestamp(day).date()} (fold {fold})", fontsize=10)
            if i == 1:
                ax.set_xlabel("hour of day")
            if j == 0:
                ax.set_ylabel("calls per hour")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle("Held-out Wednesdays: a high-volume type (top) and a sparse one (bottom)",
                 y=1.0, fontsize=11)  # fmt: skip
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(target / "anonymous_bank_2_held_out_days.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=60)
    for ax, key, title in zip(axes, ("knn", "quantile"), ("KNN", "LightGBM quantile"), strict=True):
        matrix = []
        for bucket in order:
            fm = result.for_bucket(bucket).fold_metrics
            matrix.append(
                ((fm["mae_lookup"] - fm[f"mae_{key}"]) / fm["mae_lookup"] * 100).to_numpy()
            )
        matrix = np.array(matrix)
        image = ax.imshow(matrix, cmap="RdBu", norm=norm, aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:+.0f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(matrix.shape[1]), [str(n) for n in range(1, matrix.shape[1] + 1)])
        ax.set_xlabel("walk-forward fold")
        ax.set_title(title, fontsize=10)
        ax.grid(False)
        ax.set_yticks(
            range(len(order)),
            [f"{r['bucket']}  {r['mean_per_hour']:.1f}/h  "
             + ("PROMOTED" if r[key].startswith("PROMOTE") else "kept baseline") for r in rows],
            fontsize=8.5,
        )  # fmt: skip
    fig.colorbar(image, ax=axes, shrink=0.8, label="MAE gain over lookup (%)\nblue = model better")
    fig.suptitle("Gain over the baseline by fold: promotion needs a clear win and no clearly worse "
                 "fold", y=1.02, fontsize=11)  # fmt: skip
    fig.savefig(target / "anonymous_bank_3_gain_by_bucket_and_fold.png", bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    coverage = [r["band_coverage"] for r in rows]
    ax.bar(x, coverage, color=[COLORS["model"] if abs(c - 0.5) < 0.09 else COLORS["keep"]
                               for c in coverage])  # fmt: skip
    ax.axhline(0.5, color=COLORS["actual"], ls=":", lw=1.4, label="nominal 50%")
    for position, value in zip(x, coverage, strict=True):
        ax.text(position, value + 0.02, f"{value:.0%}", ha="center", fontsize=9)
    ax.set_xticks(x, [f"{b}\n{volume[b]:.1f}/h" for b in order])
    ax.set(ylim=(0, 1.1), ylabel="share of actuals inside the band")
    ax.set_title("Band coverage by service type: within 9 points of 50% for PS, NE and IN",
                 fontsize=10)  # fmt: skip
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(target / "anonymous_bank_4_band_coverage_by_bucket.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default="data", help="folder that receives anonymous_bank/")
    parser.add_argument("--out", default="data/results", help="folder for the JSON results")
    parser.add_argument("--figures", help="folder for the four PNG figures (needs matplotlib)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    folder = Path(args.data) / "anonymous_bank"
    if not all((folder / name).is_file() for name in MONTH_SHA256):
        fetch(folder)
    calls, quality = read_calls(folder)
    config = anonymous_bank_config()
    hourly = validate_hourly(hourly_counts(calls), config)
    hourly["date"] = pd.to_datetime(hourly["date"])
    spec = FeatureSpec.from_data(hourly, config)
    frame = build_frame(hourly, spec)
    started = time.time()
    result = run_walk_forward(frame, config, spec)
    seconds = round(time.time() - started, 1)
    rows = summarize(result, hourly, weekday_lookup_errors(frame, config))
    summary = {
        "quality": asdict(quality),
        "operating_days": int(hourly["date"].nunique()),
        "calls_in_window": int(hourly["count"].sum()),
        "walk_forward_seconds": seconds,
        "band_coverage_overall": round(
            float(
                (
                    (result.predictions["actual"] >= result.predictions["lower"])
                    & (result.predictions["actual"] <= result.predictions["upper"])
                ).mean()
            ),
            3,
        ),  # fmt: skip
        "versions": {n: version(n) for n in ("numpy", "pandas", "scikit-learn", "lightgbm")},
        "by_bucket": rows,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "anonymous_bank.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "by_bucket"}, indent=1))
    print(pd.DataFrame(rows).drop(columns=["calls_total"]).to_string(index=False))
    if args.figures:
        draw_figures(result, hourly, rows, Path(args.figures))
        print("figures written to", args.figures)


if __name__ == "__main__":
    main()
