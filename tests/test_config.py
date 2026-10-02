import pytest

from call_pace.config import Config, parse_pairs
from call_pace.errors import ConfigError


def test_defaults_describe_a_twelve_hour_five_day_schedule():
    cfg = Config()
    assert list(cfg.hours) == list(range(8, 20))
    assert cfg.weekday_period == 5
    assert cfg.nominal_coverage == pytest.approx(0.5)


@pytest.mark.parametrize(
    "changes",
    [
        {"start_hour": 20, "end_hour": 8},
        {"operating_weekdays": (0, 0, 1)},
        {"operating_weekdays": (4, 2)},
        {"operating_weekdays": (7,)},
        {"first_window_hours": 0},
        {"first_window_hours": 12},
        {"postprocess": "floor"},
        {"lookup_stat": "mode"},
        {"lookup_treatment": "trim"},
        {"default_model": "arima"},
        {"routing": {"auto": "svr"}},
        {"quantiles": (0.8, 0.2)},
        {"quantiles": (0.0, 0.5)},
        {"n_folds": 1},
        {"k_grid": ()},
        {"lookup_window_days": 0},
    ],
)
def test_invalid_values_are_rejected(changes):
    with pytest.raises(ConfigError):
        Config(**changes)


def test_per_bucket_routing_and_cutoff_fall_back_to_defaults():
    cfg = Config(default_model="knn", routing={"auto": "lookup"}, maturity_cutoff_days={"auto": 3})
    assert cfg.model_for("auto") == "lookup"
    assert cfg.model_for("consumer") == "knn"
    assert cfg.cutoff_for("auto") == 3
    assert cfg.cutoff_for("consumer") == 0


def test_replace_returns_a_validated_copy():
    cfg = Config().replace(k_grid=(4, 6))
    assert cfg.k_grid == (4, 6)
    with pytest.raises(ConfigError):
        Config().replace(n_folds=0)


def test_from_env_reads_overrides_and_keeps_other_defaults():
    env = {
        "CALL_PACE_START_HOUR": "9",
        "CALL_PACE_END_HOUR": "18",
        "CALL_PACE_LOOKUP_WINDOW_DAYS": "30",
        "CALL_PACE_ARTIFACTS_DIR": "out",
        "CALL_PACE_DEFAULT_MODEL": "lookup",
        "CALL_PACE_MATURITY_CUTOFF_DAYS": "auto=2, consumer=5",
        "CALL_PACE_ROUTING": "auto=knn",
    }
    cfg = Config.from_env(env)
    assert (cfg.start_hour, cfg.end_hour, cfg.lookup_window_days) == (9, 18, 30)
    assert cfg.artifacts_dir == "out"
    assert cfg.default_model == "lookup"
    assert cfg.maturity_cutoff_days == {"auto": 2, "consumer": 5}
    assert cfg.routing == {"auto": "knn"}
    assert cfg.k_grid == Config().k_grid


def test_from_env_with_nothing_set_equals_defaults():
    assert Config.from_env({}) == Config()


@pytest.mark.parametrize(
    "env",
    [
        {"CALL_PACE_START_HOUR": "eight"},
        {"CALL_PACE_MATURITY_CUTOFF_DAYS": "auto=two"},
        {"CALL_PACE_ROUTING": "auto"},
    ],
)
def test_from_env_reports_malformed_values(env):
    with pytest.raises(ConfigError):
        Config.from_env(env)


def test_parse_pairs_handles_blank_and_spacing():
    assert parse_pairs("") == {}
    assert parse_pairs(" a = 1 ,b=2,") == {"a": "1", "b": "2"}
    with pytest.raises(ConfigError):
        parse_pairs("a=")
