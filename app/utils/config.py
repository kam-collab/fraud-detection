"""Central configuration: paths, time windows, column roles and policy targets."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

SAMPLE_CSV = DATA_DIR / "fraud_detection_sample_transactions.csv"
SYNTHETIC_CSV = DATA_DIR / "synthetic_transactions.csv.gz"
MODEL_PATH = MODELS_DIR / "fraud_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "model_card.json"

SEED = 42
TARGET = "fraud_label"

RAW_COLUMNS = [
    "request_id",
    "request_time",
    "service_type",
    "device_id",
    "merchant_id",
    "merchant_state",
    "merchant_city",
    "merchant_type",
    "mcc_code",
    "mcc_title",
    "issuer_bank",
    "currency_code",
    "amount",
    "request_status",
    "fraud_label",
]

# Time windows (half-open: start <= request_time < end).
# Development period is January-June, "live" period is July-September.
# January is feature warm-up: behavioural features look back 30 days, so the first
# month has incomplete history and is used as history only, never as training rows.
TRAIN_START, TRAIN_END = "2026-01-31", "2026-05-01"  # fit
VALID_START, VALID_END = "2026-05-08", "2026-06-01"  # model selection, calibration, thresholds
TEST_START, TEST_END = "2026-06-08", "2026-07-01"  # untouched hold-out = release baseline
LIVE_START, LIVE_END = "2026-07-01", "2026-10-01"  # production monitoring
# The 7-day gaps before validation and test mirror the label delay: labels for the
# last week before a window starts would not be known yet when that window begins.

# Fraud labels (chargebacks / customer disputes) arrive late. Any feature built
# from labels only uses labels older than this many days.
LABEL_DELAY_DAYS = 7

# Decision policy targets, applied on the validation window to pick thresholds.
REVIEW_MIN_PRECISION = 0.50  # at least 1 in 2 cases sent to manual review is fraud
BLOCK_MIN_PRECISION = 0.95  # auto-decline only when at most 1 in 20 is a false decline

# Illustrative cost assumptions (INR) for the business view. Not real figures.
REVIEW_COST = 30.0  # analyst cost per manually reviewed transaction
FALSE_DECLINE_MARGIN = 0.03  # share of amount lost when a genuine payment is blocked
FALSE_DECLINE_FIXED = 150.0  # fixed goodwill / churn cost per false decline
REVIEW_CATCH_RATE = 0.90  # share of reviewed frauds the analyst correctly stops

# Drift alert thresholds (rule-of-thumb values; see README for caveats).
PSI_WARN, PSI_ALERT = 0.10, 0.25
KS_WARN, KS_ALERT = 0.05, 0.10  # on the KS D statistic (effect size), not the p-value
FLAG_RATE_WARN, FLAG_RATE_ALERT = 0.25, 0.50  # relative change in the share of payments flagged
