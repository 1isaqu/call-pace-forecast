"""Keep the documentation true: the README's examples run and its tables match the code."""

import dataclasses
import re
from pathlib import Path

from call_pace.cli import COMMANDS
from call_pace.config import Config

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_python_examples_in_the_readme_run_in_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the examples write ./artifacts
    blocks = re.findall(r"```python\n(.*?)```", README, re.DOTALL)
    assert len(blocks) == 2
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, "README.md", "exec"), namespace)  # noqa: S102
    assert isinstance(namespace["payload"]["data"]["projected_close"], int | float)
    assert (tmp_path / "artifacts" / "manifest.json").is_file()


def test_every_command_is_documented():
    for name in COMMANDS:
        assert f"call-pace {name}" in README, name


def test_every_config_field_is_documented():
    for field in dataclasses.fields(Config):
        assert f"`{field.name}`" in README, field.name


def test_environment_variables_in_the_readme_match_env_example():
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    documented = set(re.findall(r"\b(?:CALL_PACE|DB)_[A-Z_]+\b", README))
    declared = set(re.findall(r"^#?\s*((?:CALL_PACE|DB)_[A-Z_]+)=", example, re.MULTILINE))
    assert documented == declared


def test_the_variables_config_reads_are_all_in_the_readme():
    source = (ROOT / "src" / "call_pace" / "config.py").read_text(encoding="utf-8")
    for name in set(re.findall(r'"(CALL_PACE_[A-Z_]+)"', source)):
        assert name in README, name


def test_required_repository_files_exist():
    for name in ("LICENSE", ".gitignore", ".env.example", ".github/workflows/ci.yml"):
        assert (ROOT / name).is_file(), name
    for name in ("architecture", "data-contract", "evaluation", "decisions", "glossary"):
        assert (ROOT / "docs" / f"{name}.md").is_file(), name


def test_gitignore_keeps_data_secrets_and_models_out_of_the_repository():
    rules = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for rule in (
        ".env",
        ".venv/",
        "data/",
        "artifacts/",
        "*.csv",
        "*.parquet",
        "*.pkl",
        "*.joblib",
    ):
        assert rule in rules, rule
