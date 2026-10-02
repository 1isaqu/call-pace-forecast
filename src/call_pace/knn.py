"""Phase 2: KNN regression on time-of-day curves.

The model looks for past (bucket, hour) situations that resemble the current one: same hour,
similar weekday, similar volume so far, similar early-morning total. Their average count is
the expectation. Unlike the lookup it reacts to how the day is going.

Features are standardised (zero mean, unit variance) inside a pipeline fitted on training
rows only, so a feature with a large scale cannot dominate the distance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from .errors import ArtifactError, ConfigError, InsufficientDataError
from .features import FeatureSpec, feature_matrix
from .metrics import apply_postprocess, mae
from .persistence import read_payload, write_payload


class KnnForecaster:
    """StandardScaler + KNeighborsRegressor over the engineered features."""

    name = "knn"

    def __init__(
        self,
        spec: FeatureSpec,
        k: int = 10,
        postprocess: str = "ceil",
        features: Sequence[str] | None = None,
    ) -> None:
        self.spec = spec
        self.k = k
        self.postprocess = postprocess
        self.features = tuple(features) if features is not None else spec.names
        unknown = set(self.features) - set(spec.names)
        if unknown or not self.features:
            raise ConfigError(f"features must be a non-empty subset of {spec.names}")
        self.pipeline: Pipeline | None = None

    def fit(self, frame: pd.DataFrame) -> KnnForecaster:
        if len(frame) == 0:
            raise InsufficientDataError("cannot fit a KNN on an empty frame")
        neighbors = min(self.k, len(frame))
        regressor = KNeighborsRegressor(n_neighbors=neighbors, metric="euclidean")
        self.pipeline = make_pipeline(StandardScaler(), regressor)
        self.pipeline.fit(feature_matrix(frame, self.features), frame["count"].to_numpy(float))
        return self

    def predict_raw(self, frame: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise ArtifactError("the KNN model has not been fitted")
        return self.pipeline.predict(feature_matrix(frame, self.features))

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return apply_postprocess(self.predict_raw(frame), self.postprocess)

    def save(self, path: str | Path) -> Path:
        if self.pipeline is None:
            raise ArtifactError("refusing to save an unfitted KNN model")
        payload = {
            "model": "knn",
            "sklearn_version": sklearn.__version__,
            "spec": self.spec.to_dict(),
            "k": self.k,
            "features": list(self.features),
            "postprocess": self.postprocess,
            "pipeline": self.pipeline,
        }
        return write_payload(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> KnnForecaster:
        """Load an artifact written by `save`. Only load files you produced yourself:
        joblib uses pickle, which can execute code from an untrusted file."""
        payload = read_payload(path, "knn")
        model = cls(
            FeatureSpec.from_dict(payload["spec"]),
            k=payload["k"],
            postprocess=payload["postprocess"],
            features=payload["features"],
        )
        model.pipeline = payload["pipeline"]
        return model


@dataclass(frozen=True)
class KnnCV:
    """Cross-validation over walk-forward splits for a grid of k values."""

    table: pd.DataFrame
    predictions: dict[int, list[np.ndarray]]
    best_k: int


def knn_cross_validate(
    splits: Sequence[tuple[pd.DataFrame, pd.DataFrame]],
    spec: FeatureSpec,
    k_grid: Sequence[int],
    postprocess: str = "ceil",
    features: Sequence[str] | None = None,
) -> KnnCV:
    """MAE of every k on every (train, test) split, plus the k with the lowest mean MAE.

    Ties go to the smaller k: fewer neighbours is the more local model, but the tie-break only
    matters when the means are exactly equal.
    """
    predictions: dict[int, list[np.ndarray]] = {}
    rows = []
    for k in sorted(set(k_grid)):
        model_preds = []
        for train, test in splits:
            model = KnnForecaster(spec, k, postprocess, features).fit(train)
            model_preds.append(model.predict(test))
        predictions[k] = model_preds
        errors = [mae(test["count"], p) for (_, test), p in zip(splits, model_preds, strict=True)]
        rows.append({"k": k, **{f"fold_{i + 1}": e for i, e in enumerate(errors)}})
        rows[-1]["mean_mae"] = float(np.mean(errors))
    table = pd.DataFrame(rows)
    best_k = int(table.sort_values(["mean_mae", "k"]).iloc[0]["k"])
    return KnnCV(table=table, predictions=predictions, best_k=best_k)
