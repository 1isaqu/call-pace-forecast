"""Optional HTTP endpoint (needs the `api` extra: pip install '.[api]').

    GET /health          liveness
    GET /pace            pace for all buckets
    GET /pace/{bucket}   pace for one bucket

Response envelope: {"meta": {...}, "data": {...}, "errors": [...]}. Status codes: 400 for an
unknown bucket, 500 when models or configuration are not usable. A failure while reading
today's data is NOT an error status: it returns 200 with empty bands and a message in
`errors`, so a display that polls this endpoint keeps working.
"""

from __future__ import annotations

import logging
from typing import Any

from . import __version__
from .errors import ArtifactError, ConfigError, InvalidRequestError
from .service import ALL, PaceService

log = logging.getLogger(__name__)


def create_app(service: PaceService) -> Any:
    """Build the FastAPI application around a `PaceService`."""
    try:
        from fastapi import FastAPI, HTTPException
    except ImportError as exc:
        raise ConfigError("install the optional extra: pip install '.[api]'") from exc

    app = FastAPI(title="call-pace-forecast", version=__version__)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/pace")
    @app.get("/pace/{bucket}")
    def pace(bucket: str = ALL) -> dict[str, Any]:
        try:
            return service.pace(bucket)
        except InvalidRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except (ArtifactError, ConfigError) as exc:
            log.error("service not ready: %s", exc)
            raise HTTPException(
                status_code=500, detail="models or configuration unavailable"
            ) from exc

    return app
