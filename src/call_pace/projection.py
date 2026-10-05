"""Projecting the day's final total from the volume so far.

Two ways to answer "how many will we have by closing time?":

  * `ProportionProjector`: the formula. If by this hour a bucket typically has completed a
    share `s` of its day, then projected total = cumulative so far / s.
  * `NeighborProjector`: a similarity model. Find past days that looked like today at this
    point and take the median of their final totals.

The comparison is the interesting part. By construction, share = cumulative / total, so the
formula inverts exactly the identity the similarity model has to learn indirectly. A model that
approximates a relation cannot beat a formula that states it unless it brings information the
formula ignores (weekday effects, seasonality, external events). `evaluate_projection` lets you
check that on your own data instead of taking it on faith.

A related question is how finely the share curve itself should be keyed. Adding the weekday to
the key can sharpen the curve, but every extra key value splits the history into thinner cells
whose medians are noisier. `compare_share_keys` weighs the gain against `cell_sparsity`, with
the same discipline as the rest of the library: a finer key is adopted only when it clearly
helps on held-out days AND its cells stay populated.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .errors import ConfigError, InsufficientDataError
from .metrics import mae
from .validation import PairedTest, paired_error_test

MODES = ("hour_feature", "hour_partition")
_CHUNK = 512


class ProportionProjector:
    """Projected total = cumulative / median share of the day completed by that hour."""

    name = "proportion"

    def __init__(self, window_days: int | None = None) -> None:
        self.window_days = window_days
        self.shares: dict[tuple[str, int], float] = {}
        self.fallback: dict[int, float] = {}

    def fit(self, frame: pd.DataFrame) -> ProportionProjector:
        """Fit on a modelling frame (needs `share`); silent days have no share and are skipped."""
        usable = frame[frame["share"].notna()]
        if usable.empty:
            raise InsufficientDataError("no day with events to learn a share curve from")
        if self.window_days is not None:
            usable = self._trailing(usable)
        grouped = usable.groupby(["bucket", "hour"])["share"].median()
        self.shares = {(b, int(h)): float(v) for (b, h), v in grouped.items()}
        self.fallback = {
            int(h): float(v) for h, v in usable.groupby("hour")["share"].median().items()
        }
        return self

    def _trailing(self, usable: pd.DataFrame) -> pd.DataFrame:
        keep = (
            usable.groupby("bucket")["date"].transform(
                lambda s: s.rank(method="dense", ascending=False)
            )
            <= self.window_days
        )
        return usable[keep]

    def share_for(self, bucket: str, hour: int) -> float:
        return self.shares.get((bucket, hour), self.fallback.get(hour, float("nan")))

    def project_one(self, bucket: str, hour: int, cumulative: float) -> float:
        """Projected day total, or NaN when the typical share at this hour is zero or unknown."""
        share = self.share_for(bucket, hour)
        return cumulative / share if share and share > 0 else float("nan")

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        rows = zip(frame["bucket"], frame["hour"], frame["cumulative"], strict=True)
        return np.array([self.project_one(b, int(h), c) for b, h, c in rows], dtype=float)


class NeighborProjector:
    """Median final total of the k most similar past days, in a min-max normalised space.

    Candidates are always restricted to the same bucket. With mode "hour_partition" they are
    also restricted to the same hour (exact match, distance on the rest). With mode
    "hour_feature" the hour is one more dimension of the distance and nothing is filtered.
    """

    name = "neighbors"

    def __init__(
        self,
        k: int = 10,
        mode: str = "hour_feature",
        covariates: Sequence[str] = (),
        start_hour: int = 8,
        end_hour: int = 19,
    ) -> None:
        if mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}")
        if k < 1:
            raise ConfigError("k must be positive")
        self.k, self.mode = k, mode
        self.covariates = tuple(covariates)
        self.start_hour, self.end_hour = start_hour, end_hour
        self.ood_count = 0
        self._ranges: dict[str, tuple[float, float]] = {}
        self._train: pd.DataFrame | None = None
        self._fallback = ProportionProjector()

    @property
    def _numeric(self) -> list[str]:
        return ["cumulative", *self.covariates]

    def fit(self, frame: pd.DataFrame) -> NeighborProjector:
        usable = frame[frame["day_total"] > 0]
        if usable.empty:
            raise InsufficientDataError("no day with events to learn from")
        self._ranges = {c: (float(usable[c].min()), float(usable[c].max())) for c in self._numeric}
        self._train = usable.reset_index(drop=True)
        self._fallback.fit(frame)
        return self

    def _normalise(self, frame: pd.DataFrame) -> np.ndarray:
        span = max(self.end_hour - self.start_hour, 1)
        columns = []
        if self.mode == "hour_feature":
            columns.append((frame["hour"].to_numpy(float) - self.start_hour) / span)
        for name in self._numeric:
            low, high = self._ranges[name]
            scaled = (
                (frame[name].to_numpy(float) - low) / (high - low)
                if high > low
                else np.zeros(len(frame))
            )
            self.ood_count += int(np.sum((scaled < 0) | (scaled > 1)))
            columns.append(np.clip(scaled, 0.0, 1.0))
        return np.column_stack(columns)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Projected day total per row. Cells without k candidates fall back to the formula."""
        if self._train is None:
            raise InsufficientDataError("the neighbour projector has not been fitted")
        self.ood_count = 0
        out = np.full(len(frame), np.nan)
        test_x, train_x = self._normalise(frame), self._normalise(self._train)
        keys = ["bucket", "hour"] if self.mode == "hour_partition" else ["bucket"]
        train_keys = self._train[keys]
        train_y = self._train["day_total"].to_numpy(float)
        by = keys if len(keys) > 1 else keys[0]
        for key, rows in frame[keys].reset_index(drop=True).groupby(by).indices.items():
            key = key if isinstance(key, tuple) else (key,)
            same_cell = np.logical_and.reduce(
                [train_keys[c].to_numpy() == v for c, v in zip(keys, key, strict=True)]
            )
            if same_cell.sum() < self.k:
                out[rows] = self._fallback.predict(frame.iloc[rows])
            else:
                out[rows] = _nearest_median(
                    train_x[same_cell], train_y[same_cell], test_x[rows], self.k
                )
        return out


def _nearest_median(
    train_x: np.ndarray, train_y: np.ndarray, query: np.ndarray, k: int
) -> np.ndarray:
    """Median target of the k nearest training rows for each query row (Euclidean)."""
    result = np.empty(len(query))
    for start in range(0, len(query), _CHUNK):
        block = query[start : start + _CHUNK]
        distances = np.sqrt(((block[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2))
        nearest = np.argpartition(distances, k - 1, axis=1)[:, :k]
        result[start : start + len(block)] = np.median(train_y[nearest], axis=1)
    return result


def evaluate_projection(
    frame: pd.DataFrame,
    holdout_days: int = 20,
    k: int = 10,
    covariates: Sequence[str] = (),
    start_hour: int = 8,
    end_hour: int = 19,
) -> pd.DataFrame:
    """MAE of each projector per hour on the last `holdout_days` days (earlier days train).

    Rows where any projector has no prediction (NaN) are left out for ALL projectors, so every
    column is computed on exactly the same observations. The last row, hour "all", pools them.
    """
    train, test = _holdout_split(frame, holdout_days)
    projections = {"proportion": ProportionProjector().fit(train).predict(test)}
    for mode in MODES:
        model = NeighborProjector(k, mode, covariates, start_hour, end_hour).fit(train)
        projections[mode] = model.predict(test)
    valid = np.all([np.isfinite(p) for p in projections.values()], axis=0)
    scored = test[valid].assign(**{name: p[valid] for name, p in projections.items()})
    rows = [_projection_row(str(hour), group) for hour, group in scored.groupby("hour")]
    rows.append(_projection_row("all", scored))
    return pd.DataFrame(rows)


def _projection_row(label: str, group: pd.DataFrame) -> dict[str, float | str]:
    row: dict[str, float | str] = {"hour": label, "n": len(group)}
    for name in ("proportion", *MODES):
        row[f"mae_{name}"] = mae(group["day_total"], group[name])
    return row


def _holdout_split(frame: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Earlier days train, the last `holdout_days` test; silent test days are left out."""
    dates = sorted(frame["date"].unique())
    if len(dates) <= holdout_days:
        raise InsufficientDataError(f"need more than {holdout_days} days, found {len(dates)}")
    test_dates = dates[-holdout_days:]
    train, test = frame[~frame["date"].isin(test_dates)], frame[frame["date"].isin(test_dates)]
    return train, test[test["day_total"] > 0].reset_index(drop=True)


@dataclass(frozen=True)
class CellSparsity:
    """How thinly a set of key columns spreads the rows of a frame.

    Every combination of key values is a cell. A cell with fewer than `min_obs` rows gives a
    noisy median; a cell with none is absent. Both count as thin, because both force a lookup
    to guess.
    """

    cells_possible: int
    cells_present: int
    cells_thin: int
    min_obs: int
    median_obs: float

    @property
    def thin_share(self) -> float:
        return self.cells_thin / self.cells_possible if self.cells_possible else 0.0


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ConfigError(f"columns {missing} are not in the frame")


def cell_sparsity(frame: pd.DataFrame, keys: Sequence[str], min_obs: int = 5) -> CellSparsity:
    """Cells that `keys` create in `frame`, and how many have fewer than `min_obs` rows.

    The number of possible cells is the product of the distinct values of each key, so a cell
    that never occurs is counted rather than silently ignored.
    """
    _require_columns(frame, keys)
    sizes = frame.groupby(list(keys)).size()
    possible = int(np.prod([frame[k].nunique() for k in keys]))
    thin = int((sizes < min_obs).sum()) + (possible - len(sizes))
    return CellSparsity(possible, len(sizes), thin, min_obs, float(sizes.median()))


@dataclass(frozen=True)
class KeyComparison:
    """A finer share-curve key against a coarser one, on the same held-out rows."""

    coarse_keys: tuple[str, ...]
    fine_keys: tuple[str, ...]
    n: int
    mae_coarse: float
    mae_fine: float
    fallbacks: int
    sparsity: CellSparsity
    paired: PairedTest
    min_gain_pct: float
    max_thin_share: float

    @property
    def gain_pct(self) -> float:
        """Relative error reduction of the finer key; zero when the coarse key is exact."""
        if self.mae_coarse == 0:
            return 0.0
        return (self.mae_coarse - self.mae_fine) / self.mae_coarse * 100.0

    @property
    def adopt_fine(self) -> bool:
        gain = self.gain_pct >= self.min_gain_pct
        return gain and self.sparsity.thin_share <= self.max_thin_share

    def explain(self) -> str:
        fine, coarse = "+".join(self.fine_keys), "+".join(self.coarse_keys)
        cells = self.sparsity
        if self.adopt_fine:
            verdict = f"ADOPT {fine}"
        elif self.gain_pct >= self.min_gain_pct:
            verdict = (
                f"KEEP {coarse}: the gain is there but {cells.thin_share:.0%} of the cells are "
                f"thin (limit {self.max_thin_share:.0%})"
            )
        else:
            verdict = f"KEEP {coarse}: gain below the {self.min_gain_pct:g}% needed"
        return "\n".join(
            [
                verdict,
                f"  holdout MAE: {fine} {self.mae_fine:.3f} vs {coarse} {self.mae_coarse:.3f} "
                f"({self.gain_pct:+.1f}% gain, n={self.n})",
                f"  paired test (negative favours {fine}): mean difference "
                f"{self.paired.mean_difference:+.3f}, p={self.paired.p_value:.4f}",
                f"  cells of {fine}: {cells.cells_present} of {cells.cells_possible} present, "
                f"{cells.cells_thin} thin (under {cells.min_obs} rows, absent ones included), "
                f"median {cells.median_obs:.1f} rows per cell",
                f"  rows that fell back to the coarser curve: {self.fallbacks}",
            ]
        )


def _share_lookup(train: pd.DataFrame, test: pd.DataFrame, keys: Sequence[str]) -> np.ndarray:
    """Median share per cell of `keys` learned on `train`, looked up for each `test` row."""
    cells = train[train["share"].notna()].groupby(list(keys), as_index=False)["share"].median()
    return test[list(keys)].merge(cells, on=list(keys), how="left")["share"].to_numpy(float)


def _project(cumulative: np.ndarray, share: np.ndarray) -> np.ndarray:
    """cumulative / share where the share is positive, NaN elsewhere."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(share > 0, cumulative / share, np.nan)


def compare_share_keys(
    frame: pd.DataFrame,
    fine_keys: Sequence[str],
    coarse_keys: Sequence[str] = ("bucket", "hour"),
    holdout_days: int = 20,
    min_obs: int = 5,
    min_gain_pct: float = 5.0,
    max_thin_share: float = 0.10,
) -> KeyComparison:
    """Does keying the share curve more finely (say by weekday) earn its thinner cells?

    Both curves are learned on the earlier days and projected on the last `holdout_days` days.
    A row whose finer cell was never seen falls back to the coarser curve and is counted. The
    finer key is recommended only if it reduces the error by at least `min_gain_pct` percent AND
    at most `max_thin_share` of its cells are thin (see `cell_sparsity`, measured on the training
    days). The thresholds are invented defaults. Rows where either projection is undefined are
    left out of both, so the two errors cover exactly the same observations.
    """
    fine, coarse = tuple(fine_keys), tuple(coarse_keys)
    if not set(coarse) < set(fine):
        raise ConfigError("fine_keys must contain every coarse key plus at least one more")
    _require_columns(frame, [*fine, "share", "cumulative", "day_total"])
    train, test = _holdout_split(frame, holdout_days)
    coarse_share = _share_lookup(train, test, coarse)
    fine_share = _share_lookup(train, test, fine)
    fell_back = ~(fine_share > 0)
    cumulative = test["cumulative"].to_numpy(float)
    by_coarse = _project(cumulative, coarse_share)
    by_fine = _project(cumulative, np.where(fell_back, coarse_share, fine_share))
    valid = np.isfinite(by_coarse) & np.isfinite(by_fine)
    if valid.sum() < 2:
        raise InsufficientDataError("fewer than two held-out rows can be projected by both keys")
    actual = test["day_total"].to_numpy(float)[valid]
    error_coarse, error_fine = np.abs(by_coarse[valid] - actual), np.abs(by_fine[valid] - actual)
    return KeyComparison(
        coarse,
        fine,
        int(valid.sum()),
        mae(actual, by_coarse[valid]),
        mae(actual, by_fine[valid]),
        int((fell_back & valid).sum()),
        cell_sparsity(train, fine, min_obs),
        paired_error_test(error_fine, error_coarse),
        min_gain_pct,
        max_thin_share,
    )
