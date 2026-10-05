# Usage

## Pipeline

| Command | What it does | Output |
|---|---|---|
| `make data` | Generates the seeded synthetic transactions | `data/synthetic_transactions.csv.gz` |
| `make train` | Fits 11 candidates, selects on May, calibrates, sets thresholds, evaluates on June | `models/fraud_model.joblib`, `models/model_card.json`, `reports/training_report.json` |
| `make monitor` | Drift, performance and business monitoring for July-September | `reports/monitoring_report.json` |
| `make pipeline` | All three | |
| `make notebook` | Rebuilds and executes the assignment notebook | `notebooks/fraud_detection.ipynb` |

## API

Interactive docs: `http://localhost:8000/docs`.

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/score` | Validate and score one transaction; returns score, band, reasons, explanation |
| GET | `/api/v1/transactions/sample?kind=` | A stored September transaction to replay (`random`, `fraud`, `genuine`, `review`, `blocked`) |
| GET | `/api/v1/review-queue` | Transactions in the review band over the last 3 days of data |
| POST | `/api/v1/review-queue/{id}/decision` | Record an analyst decision (`approve` / `decline`) |
| GET | `/api/v1/explain/{id}?llm=` | Reasons and explanation for a transaction |
| GET | `/api/v1/model` | Model card and June hold-out evaluation |
| GET | `/api/v1/monitoring` | Monthly drift report |
| GET | `/api/v1/overview` | EDA summaries for the development period |
| GET | `/health`, `/readiness`, `/metrics` | Liveness, readiness, Prometheus metrics |

Score a transaction:

```bash
curl -s -X POST localhost:8000/api/v1/score -H 'content-type: application/json' -d '{
  "service_type": "wallet", "device_id": "DEV0003142", "merchant_id": "M100296",
  "merchant_type": "high", "mcc_code": 7995, "mcc_title": "Betting and Gambling",
  "issuer_bank": "NOT_APPLICABLE", "amount": 45000
}'
```

`request_id` and `request_time` are generated when omitted. `request_status` is optional and
never used for the current transaction: the decision is made before the outcome is known.

The API has no authentication. It is a local demonstration, not a deployable service.

## Observability

- **Logs**: one JSON object per line on stderr, each carrying a `trace_id`. Send an
  `X-Trace-Id` header to propagate your own; the response echoes it.
- **Metrics**: `GET /metrics` in Prometheus format: request counts and latency by route,
  score distribution, decisions per band, validation failures, explanation source, analyst
  decisions, and the loaded model version and thresholds.
- **Prediction log**: every scored transaction is appended to `logs/predictions.jsonl` with
  its features, score, band and model version. Joined later with labels, this is what
  drift and performance monitoring run on in production.
