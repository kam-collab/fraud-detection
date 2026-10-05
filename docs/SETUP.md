# Setup

## Requirements

- Python 3.10 or newer (developed on 3.13)
- Node.js 20 or newer, only for the analyst console in `frontend/`
- No database, no Docker, no API keys

## Backend

```bash
make setup        # creates .venv and installs app/requirements.txt
make pipeline     # generate data -> train -> monitor (about 2 minutes)
make test         # unit tests
make api          # scoring API on http://localhost:8000
```

Without `make`:

```bash
python3 -m venv .venv
.venv/bin/pip install -r app/requirements.txt
.venv/bin/python scripts/run_pipeline.py
.venv/bin/python -m uvicorn app.main:app --port 8000
```

The tree model is scikit-learn's `HistGradientBoostingClassifier`, so there is no
`libomp` / compiler requirement (which LightGBM and XGBoost have on macOS).

## Frontend

```bash
make frontend-setup   # npm install
make frontend         # http://localhost:3000 (expects the API on :8000)
```

## Optional: LLM explanations

Copy `.env.example` to `.env`, set `EXPLANATIONS_LLM=true` and provide Anthropic
credentials (`ANTHROPIC_API_KEY`). Without them the API returns template explanations and
nothing else changes. Only the template path has been verified in this submission.
