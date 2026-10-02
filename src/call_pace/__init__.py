"""Intraday forecasting of cumulative outreach or agreement volume.

The names below are the stable entry points; everything else is importable from its module.
"""

from ._version import __version__
from .ablation import compare_lookup_variants, covariate_signal_report, feature_subset_search
from .artifacts import ArtifactStore
from .baseline import LookupForecaster
from .config import Config
from .contract import AmountLoader, HourlyLoader, validate_amounts, validate_hourly
from .features import FeatureSpec, build_frame
from .knn import KnnForecaster
from .montecarlo import MonteCarloConfig, run_monte_carlo
from .pace import Band, aggregate_bands, build_bands, summarize
from .projection import NeighborProjector, ProportionProjector, evaluate_projection
from .quantile import QuantileForecaster
from .service import PaceService
from .training import train_models
from .validation import PromotionCriteria, decide_promotion, run_walk_forward
from .value import TicketLookup

__all__ = [
    "AmountLoader",
    "ArtifactStore",
    "Band",
    "Config",
    "FeatureSpec",
    "HourlyLoader",
    "KnnForecaster",
    "LookupForecaster",
    "MonteCarloConfig",
    "NeighborProjector",
    "PaceService",
    "PromotionCriteria",
    "ProportionProjector",
    "QuantileForecaster",
    "TicketLookup",
    "__version__",
    "aggregate_bands",
    "build_bands",
    "build_frame",
    "compare_lookup_variants",
    "covariate_signal_report",
    "decide_promotion",
    "evaluate_projection",
    "feature_subset_search",
    "run_monte_carlo",
    "run_walk_forward",
    "summarize",
    "train_models",
    "validate_amounts",
    "validate_hourly",
]
