"""Fraud scoring API.

Run:  uvicorn app.main:app --port 8000      (or `make api`)
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.controllers.health_check.controller import router as health_router
from app.controllers.scoring.controller import router as scoring_router
from app.utils import metrics
from app.utils.logger import get_logger, trace_id_var

load_dotenv()
log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.service = None
    try:
        from app.controllers.scoring.service import ScoringService

        app.state.service = ScoringService()
    except Exception as e:  # keep the process up so /health and /readiness can report the cause
        app.state.load_error = f"{type(e).__name__}: {e}"
        log.error("scoring service failed to load: %s", app.state.load_error)
    yield


app = FastAPI(title="Payment Fraud Detection API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id"],
)


@app.middleware("http")
async def observe(request: Request, call_next):
    """Attach a trace id to every request, log it, and record latency and status."""
    trace_id = request.headers.get("X-Trace-Id") or uuid.uuid4().hex[:16]
    token = trace_id_var.set(trace_id)
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Trace-Id"] = trace_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        path = getattr(route, "path", request.url.path)  # route template keeps label cardinality low
        metrics.HTTP_REQUESTS.labels(method=request.method, path=path, status=str(status)).inc()
        metrics.HTTP_LATENCY.labels(path=path).observe(elapsed)
        if path not in ("/health", "/metrics"):
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": status,
                    "duration_ms": round(elapsed * 1000, 1),
                },
            )
        trace_id_var.reset(token)


app.include_router(health_router)
app.include_router(scoring_router)
