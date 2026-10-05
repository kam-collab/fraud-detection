"""Prometheus metrics for the scoring service (exposed at /metrics)."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

HTTP_REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
SCORES = Histogram(
    "fraud_score",
    "Distribution of fraud scores returned by the API",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9),
)
DECISIONS = Counter("fraud_decisions_total", "Decisions by band", ["band"])
VALIDATION_FAILURES = Counter("fraud_validation_failures_total", "Requests rejected by input validation")
EXPLANATIONS = Counter("fraud_explanations_total", "Explanations generated", ["source"])
REVIEW_DECISIONS = Counter("fraud_review_decisions_total", "Analyst decisions on the review queue", ["decision"])
MODEL_INFO = Gauge("fraud_model_info", "Loaded model version and thresholds", ["version"])
REVIEW_THRESHOLD = Gauge("fraud_review_threshold", "Score at or above which a payment is reviewed")
BLOCK_THRESHOLD = Gauge("fraud_block_threshold", "Score at or above which a payment is blocked")
