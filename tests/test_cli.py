import json

import pandas as pd
import pytest

from call_pace.cli import build_parser, main


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    """A scratch folder with an invented dataset already written by `call-pace synth`."""
    folder = tmp_path_factory.mktemp("cli")
    code = main(
        [
            "synth",
            "--out",
            str(folder / "hourly.csv"),
            "--amounts-out",
            str(folder / "amounts.csv"),
            "--days",
            "90",
            "--seed",
            "3",
        ]
    )
    assert code == 0
    return folder


@pytest.fixture(scope="module")
def trained(workspace):
    args = [
        "train",
        "--csv",
        str(workspace / "hourly.csv"),
        "--amounts-csv",
        str(workspace / "amounts.csv"),
        "--artifacts",
        str(workspace / "artifacts"),
    ]
    assert main(args) == 0
    return workspace / "artifacts"


def data_args(workspace):
    return ["--csv", str(workspace / "hourly.csv")]


def test_synth_writes_the_contract_columns_and_the_requested_volume(workspace):
    hourly = pd.read_csv(workspace / "hourly.csv")
    assert list(hourly.columns) == ["bucket", "date", "hour", "count"]
    assert len(hourly) == 90 * 2 * 12
    amounts = pd.read_csv(workspace / "amounts.csv")
    assert list(amounts.columns) == ["bucket", "date", "hour", "amount"] and len(amounts) > 1000


def test_synth_can_add_a_covariate_column(tmp_path):
    out = tmp_path / "with_cov.csv"
    assert main(["synth", "--out", str(out), "--days", "30", "--with-covariate"]) == 0
    assert "days_since_event" in pd.read_csv(out).columns


def test_train_reports_what_it_fitted_and_writes_artifacts(trained):
    names = sorted(p.name for p in trained.iterdir())
    assert names == ["knn.joblib", "lookup.json", "manifest.json", "quantile.joblib", "ticket.json"]
    manifest = json.loads((trained / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["buckets"] == ["auto", "consumer"] and manifest["days"] == 90


def test_fill_missing_days_turns_a_silent_weekday_into_a_zero_day(workspace, tmp_path):
    hourly = pd.read_csv(workspace / "hourly.csv")
    gapped = hourly[hourly["date"] != sorted(hourly["date"].unique())[40]]
    path = tmp_path / "gapped.csv"
    gapped.to_csv(path, index=False)
    days = {}
    for flag in ([], ["--fill-missing-days"]):
        folder = tmp_path / ("filled" if flag else "plain")
        assert (
            main(
                [
                    "train",
                    "--csv",
                    str(path),
                    "--models",
                    "lookup",
                    "--artifacts",
                    str(folder),
                    *flag,
                ]
            )
            == 0
        )
        days[bool(flag)] = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))[
            "days"
        ]
    assert days == {False: 89, True: 90}


def test_validate_prints_folds_decisions_and_band_coverage(workspace, capsys, tmp_path):
    report = tmp_path / "report.md"
    assert main(["validate", *data_args(workspace), "--report", str(report)]) == 0
    out = capsys.readouterr().out
    assert "Fold metrics" in out and "Selected k:" in out
    assert "PROMOTE: knn" in out or "KEEP BASELINE: knn" in out
    assert "paired test vs lookup" in out and "empirical" in out
    assert report.read_text(encoding="utf-8").startswith("Fold metrics")


def test_validate_can_also_decide_per_bucket(workspace, capsys):
    assert main(["validate", *data_args(workspace), "--per-bucket"]) == 0
    out = capsys.readouterr().out
    assert "bucket auto" in out and "bucket consumer" in out
    assert out.count("paired test vs lookup") == 6  # pooled + two buckets, two challengers each


def test_validate_with_only_the_baseline_and_one_challenger(workspace, capsys):
    assert main(["validate", *data_args(workspace), "--challengers", "knn"]) == 0
    out = capsys.readouterr().out
    assert "mae_quantile" not in out and "mae_knn" in out


def test_projection_prints_a_table_per_hour_and_overall(workspace, capsys):
    assert main(["projection", *data_args(workspace), "--holdout-days", "15"]) == 0
    out = capsys.readouterr().out
    assert "mae_proportion" in out and "| all |" in out


def test_ablate_features_prints_a_selection_and_a_ranked_table(workspace, capsys):
    assert main(["ablate", "features", *data_args(workspace), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "selected:" in out and "holdout MAE" in out and "cv_mae" in out


def test_ablate_lookup_variants(workspace, capsys):
    assert main(["ablate", "lookup", *data_args(workspace)]) == 0
    assert "hour+bucket" in capsys.readouterr().out


def test_ablate_covariate_needs_a_covariate_name(workspace, capsys):
    assert main(["ablate", "covariate", *data_args(workspace)]) == 2
    assert "--covariate" in capsys.readouterr().err


def test_ablate_covariate_reports_a_verdict(tmp_path, capsys):
    out = tmp_path / "cov.csv"
    main(["synth", "--out", str(out), "--days", "90", "--with-covariate"])
    capsys.readouterr()
    assert main(["ablate", "covariate", "--csv", str(out), "--covariate", "days_since_event"]) == 0
    assert "signal" in capsys.readouterr().out


def test_montecarlo_prints_percentiles_and_probabilities_and_can_save_runs(
    workspace, capsys, tmp_path
):
    runs = tmp_path / "runs.csv"
    args = ["montecarlo", *data_args(workspace), "--scenarios", "3", "--months", "2"]
    assert main([*args, "--window", "40", "--seed", "4", "--out", str(runs)]) == 0
    out = capsys.readouterr().out
    assert "mae_p50" in out and "P(directional accuracy" in out and "beats the lookup" in out
    assert len(pd.read_csv(runs)) == 3 * 2


def test_pace_prints_json_for_a_replayed_moment(workspace, trained, capsys):
    hourly = pd.read_csv(workspace / "hourly.csv")
    last_day = hourly["date"].max()
    args = [
        "pace",
        *data_args(workspace),
        "--artifacts",
        str(trained),
        "--now",
        f"{last_day}T11:20:00",
        "--bucket",
        "consumer",
    ]
    assert main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["meta"]["in_operation"] is True and len(payload["data"]["bands"]) == 12


def test_errors_are_reported_cleanly_with_a_non_zero_exit(workspace, trained, capsys):
    assert main(["pace", *data_args(workspace), "--artifacts", str(trained), "--bucket", "x"]) == 2
    assert "unknown bucket" in capsys.readouterr().err
    assert main(["train"]) == 2
    assert "--csv is required" in capsys.readouterr().err


def test_sqlserver_source_without_environment_is_a_clean_error(monkeypatch, capsys):
    for name in ("DB_HOST", "DB_USER", "DB_PASSWORD", "DB_NAMES"):
        monkeypatch.delenv(name, raising=False)
    assert main(["train", "--source", "sqlserver"]) == 2
    assert "DB_NAMES" in capsys.readouterr().err


def test_serve_hands_the_app_to_uvicorn(workspace, trained, monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    captured = {}
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.update(app=app, **kwargs))
    args = ["serve", *data_args(workspace), "--artifacts", str(trained), "--port", "8123"]
    assert main(args) == 0
    assert captured["host"] == "127.0.0.1" and captured["port"] == 8123
    assert captured["app"].title == "call-pace-forecast"


def test_bad_dates_and_missing_commands_exit_through_argparse(workspace):
    with pytest.raises(SystemExit) as bad_date:
        main(["train", *data_args(workspace), "--start", "yesterday"])
    assert bad_date.value.code == 2
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0 and "call-pace" in capsys.readouterr().out
