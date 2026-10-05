# =============================================================================
# FRAUD-DETECTION — Makefile
# =============================================================================
# Quick start:  make setup && make pipeline && make api     (then: make frontend)
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help
PY := .venv/bin/python

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

.PHONY: setup
setup: ## Create .venv and install Python dependencies
	python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r app/requirements.txt

.PHONY: frontend-setup
frontend-setup: ## Install frontend dependencies
	cd frontend && npm install

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

.PHONY: data
data: ## Generate the synthetic transactions (seeded)
	$(PY) -m app.controllers.data_generation.generator

.PHONY: train
train: ## Train, select, calibrate and evaluate; writes models/ and reports/
	$(PY) -m app.controllers.training.trainer

.PHONY: monitor
monitor: ## Drift and performance monitoring for July-September
	$(PY) -m app.controllers.monitoring.report

.PHONY: pipeline
pipeline: ## data -> train -> monitor in one go
	$(PY) scripts/run_pipeline.py

.PHONY: notebook
notebook: ## Rebuild and execute the assignment notebook
	$(PY) scripts/build_notebook.py

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

.PHONY: api
api: ## Start the scoring API on :8000 (docs at /docs, metrics at /metrics)
	$(PY) -m uvicorn app.main:app --port 8000

.PHONY: frontend
frontend: ## Start the analyst console on :3000
	cd frontend && npm run dev

# ---------------------------------------------------------------------------
# Testing & code quality
# ---------------------------------------------------------------------------

.PHONY: test
test: ## Run unit tests
	PYTHONPATH=. $(PY) -m pytest tests/unit -q

.PHONY: test-all
test-all: ## Run unit + integration tests (needs the trained model and data)
	PYTHONPATH=. $(PY) -m pytest tests -q

.PHONY: lint
lint: ## Check formatting
	$(PY) -m black --check --line-length 120 app/ tests/ scripts/
	$(PY) -m isort --check-only --profile black --line-length 120 app/ tests/ scripts/

.PHONY: fmt
fmt: ## Auto-format code
	$(PY) -m black --line-length 120 app/ tests/ scripts/
	$(PY) -m isort --profile black --line-length 120 app/ tests/ scripts/

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
