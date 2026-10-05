"""API routes: scoring, review queue, explanations, model card, monitoring, EDA overview."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query, Request

from app.controllers.scoring.schemas import ReviewDecision, Transaction
from app.controllers.scoring.service import ScoringService, ValidationFailed
from app.utils import config

router = APIRouter(prefix="/api/v1", tags=["fraud"])


def _service(request: Request) -> ScoringService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(503, "Model not loaded. Run `make pipeline` first.")
    return service


def _report(name: str) -> dict:
    path = config.REPORTS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found. Run `make pipeline` first.")
    return json.loads(path.read_text())


@router.post("/score")
def score(txn: Transaction, request: Request) -> dict:
    """Score one transaction and return the decision band with the main reasons."""
    try:
        return _service(request).score(txn.model_dump())
    except ValidationFailed as e:
        raise HTTPException(422, {"errors": e.result.errors, "warnings": e.result.warnings})


@router.get("/transactions/sample")
def sample(request: Request, kind: str = Query("random", pattern="^(random|fraud|genuine|review|blocked)$")) -> dict:
    return _service(request).sample(kind)


@router.get("/review-queue")
def review_queue(request: Request, limit: int = Query(25, ge=1, le=200), offset: int = Query(0, ge=0)) -> dict:
    return _service(request).review_queue(limit, offset)


@router.post("/review-queue/{request_id}/decision")
def review_decision(request_id: str, body: ReviewDecision, request: Request) -> dict:
    out = _service(request).decide(request_id, body.decision)
    if out is None:
        raise HTTPException(404, "request_id is not in the review queue")
    return out


@router.get("/explain/{request_id}")
def explain(request_id: str, request: Request, llm: bool | None = None) -> dict:
    """Reasons for a transaction's score. `llm=true` asks for the Claude-written note
    (falls back to the template when the LLM is not configured)."""
    out = _service(request).explain(request_id, use_llm=llm)
    if out is None:
        raise HTTPException(404, "Unknown request_id")
    return out


@router.get("/model")
def model(request: Request) -> dict:
    """Model card plus the release evaluation on the June hold-out."""
    card = json.loads(config.MODEL_CARD_PATH.read_text()) if config.MODEL_CARD_PATH.exists() else {}
    report = _report("training_report.json")
    return {
        "card": card,
        "champion": report["champion"],
        "baseline": report["baseline"],
        "selection_table": report["selection_table"],
        "rows": report["rows"],
        "windows": report["windows"],
    }


@router.get("/monitoring")
def monitoring() -> dict:
    return _report("monitoring_report.json")


@router.get("/overview")
def overview(request: Request) -> dict:
    return _service(request).overview()
