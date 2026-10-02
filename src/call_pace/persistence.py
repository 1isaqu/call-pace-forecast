"""Reading and writing fitted-model files.

Models with scikit-learn or LightGBM internals are stored with joblib, which is pickle based.
Pickle can execute arbitrary code when loading, so only ever load artifacts that your own
training run wrote into a directory you control.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from .errors import ArtifactError

FORMAT_VERSION = 1


def write_payload(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write `payload` (which must carry a `model` key) to `path`, creating parent folders."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**payload, "format_version": FORMAT_VERSION}, target)
    return target


def read_payload(path: str | Path, expected_model: str) -> dict[str, Any]:
    """Read a payload back and check it is the kind of model the caller expects."""
    try:
        payload = joblib.load(path)
    except Exception as exc:
        raise ArtifactError(f"cannot read artifact {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("model") != expected_model:
        raise ArtifactError(f"{path} is not a {expected_model} artifact")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ArtifactError(f"{path} has an unsupported format version")
    return payload
