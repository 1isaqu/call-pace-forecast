"""Phase 3: LightGBM central forecast with quantile bands.

Three gradient-boosted models share one feature set:
  * a Poisson-objective model for the central expectation (the target is a count),
  * a lower-quantile model and an upper-quantile model for the band.

The band is a range of plausible hourly counts, not a guarantee. A band fitted at the 25th and
75th percentiles has a nominal coverage of 50 percent; what it actually covers on unseen days
is an empirical question answered by `validation.run_walk_forward` (`coverage` column).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from .config import DEFAULT_LGBM_PARAMS
from .errors import ArtifactError, ConfigError, InsufficientDataError
from .features import FeatureSpec, feature_matrix
from .metrics import apply_postprocess
from .persistence import read_payload, write_payload


class QuantileForecaster:
    """Poisson centre plus lower/upper quantile regressors."""

    name = "quantile"

    def __init__(
        self,
        spec: FeatureSpec,
        quantiles: tuple[float, float] = (0.25, 0.75),
        params: Mapping[str, Any] | None = None,
        postprocess: str = "ceil",
        features: Sequence[str] | None = None,
    ) -> None:
        low, high = quantiles
        if not 0.0 < low < high < 1.0:
            raise ConfigError("quantiles must satisfy 0 < lower < upper < 1")
        self.spec = spec
        self.quantiles = (float(low), float(high))
        self.params = {**DEFAULT_LGBM_PARAMS, **dict(params or {})}
        self.postprocess = postprocess
        self.features = tuple(features) if features is not None else spec.names
        self.center: LGBMRegressor | None = None
        self.lower: LGBMRegressor | None = None
        self.upper: LGBMRegressor | None = None

    def fit(self, frame: pd.DataFrame) -> QuantileForecaster:
        if len(frame) == 0:
            raise InsufficientDataError("cannot fit on an empty frame")
        target = frame["count"].to_numpy(float)
        if (target < 0).any():
            raise ConfigError("the Poisson objective needs non-negative targets")
        if target.sum() == 0:
            raise InsufficientDataError("the Poisson objective cannot fit a series with no events")
        x = feature_matrix(frame, self.features)
        self.center = self._regressor("poisson").fit(x, target)
        self.lower = self._regressor("quantile", self.quantiles[0]).fit(x, target)
        self.upper = self._regressor("quantile", self.quantiles[1]).fit(x, target)
        return self

    def _regressor(self, objective: str, alpha: float | None = None) -> LGBMRegressor:
        extra = {"alpha": alpha} if alpha is not None else {}
        return LGBMRegressor(objective=objective, **self.params, **extra)

    def predict_bands(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Post-processed centre, lower and upper expectation for every row."""
        if self.center is None or self.lower is None or self.upper is None:
            raise ArtifactError("the quantile model has not been fitted")
        x = feature_matrix(frame, self.features)
        low = apply_postprocess(self.lower.predict(x), self.postprocess)
        high = apply_postprocess(self.upper.predict(x), self.postprocess)
        centre = apply_postprocess(self.center.predict(x), self.postprocess)
        # Independently fitted quantile models can cross; order them so lower <= upper.
        return pd.DataFrame(
            {"center": centre, "lower": np.minimum(low, high), "upper": np.maximum(low, high)},
            index=frame.index,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_bands(frame)["center"].to_numpy()

    def save(self, path: str | Path) -> Path:
        if self.center is None:
            raise ArtifactError("refusing to save an unfitted quantile model")
        payload = {
            "model": "quantile",
            "lightgbm_version": lightgbm.__version__,
            "spec": self.spec.to_dict(),
            "quantiles": list(self.quantiles),
            "params": self.params,
            "features": list(self.features),
            "postprocess": self.postprocess,
            "center": self.center,
            "lower": self.lower,
            "upper": self.upper,
        }
        return write_payload(path, payload)

    @classmethod
    def load(cls, path: str | Path) -> QuantileForecaster:
        """Load an artifact written by `save` (pickle based: only load files you made)."""
        payload = read_payload(path, "quantile")
        model = cls(
            FeatureSpec.from_dict(payload["spec"]),
            quantiles=tuple(payload["quantiles"]),
            params=payload["params"],
            postprocess=payload["postprocess"],
            features=payload["features"],
        )
        model.center, model.lower, model.upper = (
            payload["center"],
            payload["lower"],
            payload["upper"],
        )
        return model
