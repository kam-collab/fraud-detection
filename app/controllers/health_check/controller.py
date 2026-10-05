"""Liveness, readiness and Prometheus metrics."""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/readiness")
def readiness(request: Request, response: Response) -> dict:
    service = getattr(request.app.state, "service", None)
    if service is None:
        response.status_code = 503
        return {"status": "not_ready", "reason": getattr(request.app.state, "load_error", "model not loaded")}
    return {"status": "ready", "model_version": service.model.version, "history_rows": len(service.df)}


@router.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
