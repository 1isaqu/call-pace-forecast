"""Where fitted models live at runtime, and how serving reads them.

Models are written into a directory chosen at run time (default `artifacts/`, ignored by git).
Nothing fitted is ever committed to the repository. A manifest next to the models records what
they were trained on, so a service can start without re-deriving the feature layout.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ._version import __version__
from .baseline import LookupForecaster
from .errors import ArtifactError
from .features import FeatureSpec
from .knn import KnnForecaster
from .quantile import QuantileForecaster
from .value import TicketLookup

KINDS = ("lookup", "knn", "quantile", "ticket")
_FILES = {
    "lookup": "lookup.json",
    "knn": "knn.joblib",
    "quantile": "quantile.joblib",
    "ticket": "ticket.json",
}
_LOADERS: dict[str, Callable[[Path], Any]] = {
    "lookup": LookupForecaster.load,
    "knn": KnnForecaster.load,
    "quantile": QuantileForecaster.load,
    "ticket": TicketLookup.load,
}
MANIFEST_FILE = "manifest.json"
MANIFEST_VERSION = 1


class TtlCache:
    """A small thread-safe cache whose entries expire after `ttl_seconds`.

    The lock is held while a missing entry is being loaded, so a burst of concurrent requests
    triggers one load instead of many. `clock` is injectable so tests need no sleeping.
    """

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._items: dict[Any, tuple[Any, float]] = {}
        self._lock = threading.Lock()

    def get_or_load(self, key: Any, loader: Callable[[], Any]) -> Any:
        with self._lock:
            now = self._clock()
            self._items = {k: v for k, v in self._items.items() if now - v[1] <= self.ttl_seconds}
            if key not in self._items:
                self._items[key] = (loader(), now)
            return self._items[key][0]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class ArtifactStore:
    """Save and load fitted models and the manifest in one directory."""

    def __init__(
        self,
        directory: str | Path,
        ttl_seconds: float = 86400,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.directory = Path(directory)
        self._cache = TtlCache(ttl_seconds, clock)

    def path(self, kind: str) -> Path:
        if kind not in _FILES:
            raise ArtifactError(f"unknown artifact kind {kind!r}; choose from {KINDS}")
        return self.directory / _FILES[kind]

    def exists(self, kind: str) -> bool:
        return self.path(kind).is_file()

    def available(self) -> list[str]:
        return [kind for kind in KINDS if self.exists(kind)]

    def save(self, kind: str, model: Any) -> Path:
        self._cache.clear()
        return model.save(self.path(kind))

    def load(self, kind: str) -> Any:
        """Load a model, cached for the store's TTL. Raises ArtifactError when it is missing."""
        path = self.path(kind)
        if not path.is_file():
            raise ArtifactError(f"no {kind} artifact at {path}; train first (call-pace train)")
        return self._cache.get_or_load(("model", kind), lambda: _LOADERS[kind](path))

    def write_manifest(self, spec: FeatureSpec, summary: Mapping[str, Any] | None = None) -> Path:
        """Record the feature layout and what was trained, next to the models."""
        self._cache.clear()
        payload = {
            "format_version": MANIFEST_VERSION,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "library_version": __version__,
            "spec": spec.to_dict(),
            "buckets": list(spec.buckets),
            "kinds": self.available(),
            **dict(summary or {}),
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / MANIFEST_FILE
        target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return target

    def manifest(self) -> dict[str, Any]:
        path = self.directory / MANIFEST_FILE
        if not path.is_file():
            raise ArtifactError(f"no manifest at {path}; train first (call-pace train)")
        return self._cache.get_or_load(("manifest",), lambda: _read_manifest(path))

    def spec(self) -> FeatureSpec:
        return FeatureSpec.from_dict(self.manifest()["spec"])

    def known_buckets(self) -> list[str]:
        return list(self.manifest()["buckets"])

    def clear_cache(self) -> None:
        self._cache.clear()


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactError(f"cannot read manifest {path}: {exc}") from exc
    if data.get("format_version") != MANIFEST_VERSION or "spec" not in data:
        raise ArtifactError(f"{path} is not a supported manifest")
    return data
