"""Shared fixtures. All data here is invented, generated in memory."""

from __future__ import annotations

import pandas as pd
import pytest

from call_pace.artifacts import ArtifactStore
from call_pace.config import DEFAULT_LGBM_PARAMS, Config
from call_pace.contract import validate_hourly
from call_pace.features import FeatureSpec, build_frame
from call_pace.synthetic import make_synthetic_amounts, make_synthetic_hourly
from call_pace.training import train_models

FAST_LGBM = {**DEFAULT_LGBM_PARAMS, "n_estimators": 40, "min_child_samples": 5}


@pytest.fixture(scope="session")
def config() -> Config:
    """Small protocol so walk-forward tests run in seconds."""
    return Config(
        n_folds=3,
        fold_size_days=5,
        min_train_days=20,
        k_grid=(3, 5, 8),
        default_k=5,
        lookup_window_days=40,
        lgbm_params=FAST_LGBM,
    )


@pytest.fixture(scope="session")
def hourly(config: Config) -> pd.DataFrame:
    """About three months of validated synthetic hourly counts for two buckets."""
    return validate_hourly(make_synthetic_hourly(n_days=70, seed=3, config=config), config)


@pytest.fixture(scope="session")
def spec(hourly: pd.DataFrame, config: Config) -> FeatureSpec:
    return FeatureSpec.from_data(hourly, config)


@pytest.fixture(scope="session")
def frame(hourly: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    return build_frame(hourly, spec)


@pytest.fixture(scope="session")
def routed(config: Config) -> Config:
    """One bucket served by the lookup, the other by the KNN."""
    return config.replace(routing={"auto": "lookup", "consumer": "knn"})


@pytest.fixture(scope="session")
def amounts(hourly: pd.DataFrame) -> pd.DataFrame:
    return make_synthetic_amounts(hourly, seed=21)


@pytest.fixture(scope="session")
def trained_store(tmp_path_factory, hourly: pd.DataFrame, config: Config, amounts: pd.DataFrame):
    """An artifacts folder with every model kind trained on the synthetic data."""
    store = ArtifactStore(tmp_path_factory.mktemp("trained_artifacts"))
    train_models(hourly, config, store, amounts=amounts)
    return store


@pytest.fixture(scope="session")
def day(hourly: pd.DataFrame):
    """The last day of the synthetic data, used as 'today' in serving tests."""
    return hourly["date"].max().date()
