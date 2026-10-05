"""Monthly monitoring of the live period (July-September) against the release baseline.

Four layers, cheapest and earliest first:
  1. data drift         - needs no labels, available immediately
  2. prediction drift   - needs no labels, available immediately
  3. performance drift  - needs labels, so it lags by the label delay
  4. business metrics   - partly immediate (approval rate, queue), partly delayed (fraud loss)
Each alert is mapped to a concrete action.
"""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd

from app.controllers.evaluation import metrics as ev
from app.controllers.features.builder import build_features, load_transactions, window_mask
from app.controllers.monitoring.drift import feature_drift, psi_numeric, status
from app.controllers.training.model import FraudModel, make_gbm
from app.controllers.training.trainer import build_model, policy_thresholds, split_masks
from app.controllers.validation.schema import validate
from app.utils import config
from app.utils.logger import get_logger

log = get_logger(__name__)

RAW_NUMERIC = ["amount", "hour"]
RAW_CATEGORICAL = ["service_type", "merchant_type", "mcc_code", "issuer_bank"]
ENGINEERED = ["dev_cnt_24h", "dev_logamt_z", "dev_new_merchant", "mer_logamt_z", "mer_fraud_rate_lag", "is_night"]
LIVE_MONTHS = [
    ("2026-07", "2026-07-01", "2026-08-01"),
    ("2026-08", "2026-08-01", "2026-09-01"),
    ("2026-09", "2026-09-01", "2026-10-01"),
]
ORDER = ["unknown", "ok", "warn", "alert"]


def _worst(statuses) -> str:
    return max(statuses, key=ORDER.index) if statuses else "ok"


def _raw_view(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["amount"] + RAW_CATEGORICAL].copy()
    out["hour"] = df["request_time"].dt.hour
    return out


def _actions(month: dict) -> list[dict]:
    """Translate the month's findings into the runbook actions."""
    out = []

    def add(severity, issue, action):
        out.append({"severity": severity, "issue": issue, "action": action})

    for severity, key in (("alert", "errors"), ("warn", "warnings")):
        for w in month["data_quality"][key]:
            add(
                severity,
                f"Data quality: {w}",
                "Investigate data quality with the upstream owner before trusting scores; "
                "do not retrain on the affected rows until fixed.",
            )
    drifted = [f for f in month["data_drift"] if f["status"] != "ok"]
    if drifted:
        names = ", ".join(
            f"{f['feature']} (PSI {f['psi']:.2f}" + (f", KS {f['ks']:.2f})" if f["ks"] is not None else ")")
            for f in drifted
        )
        add(
            _worst([f["status"] for f in drifted]),
            f"Input drift: {names}",
            "Confirm it is a real business change (pricing, promotion, product mix) and not a pipeline "
            "fault; if real, check prediction drift and schedule a challenger trained on recent data.",
        )
    pd_ = month["prediction_drift"]
    if pd_["status"] != "ok":
        add(
            pd_["status"],
            f"Prediction drift: {pd_['flag_rate']:.2%} of payments flagged for review or block vs "
            f"{pd_['baseline_flag_rate']:.2%} at release ({pd_['flag_rate_change']:+.0%}); score PSI {pd_['score_psi']:.2f}",
            "Adjust the decision thresholds to keep the review queue within analyst capacity, "
            "then confirm with labels once they arrive.",
        )
    perf = month["performance"]
    if perf["precision_status"] != "ok":
        add(
            perf["precision_status"],
            f"Precision {perf['precision']:.2f} vs baseline {perf['baseline_precision']:.2f} " "(more false positives)",
            "Re-tune the decision thresholds on the most recent labelled window to restore the precision floor.",
        )
    if perf["recall_status"] != "ok" or perf["pr_auc_status"] != "ok":
        add(
            _worst([perf["recall_status"], perf["pr_auc_status"]]),
            f"Recall {perf['recall']:.2f} vs {perf['baseline_recall']:.2f}, PR-AUC {perf['pr_auc']:.2f} vs "
            f"{perf['baseline_pr_auc']:.2f} (fraud being missed)",
            "Train a challenger on recent data, run it in shadow mode against the champion, and "
            "promote it if it wins at the same alert rate. Retrain on a fixed schedule afterwards.",
        )
    if perf["pr_auc_status"] == "alert" and perf["pr_auc"] < 0.6 * perf["baseline_pr_auc"]:
        add(
            "alert",
            "Severe performance collapse (PR-AUC below 60% of the release baseline)",
            "If this followed a model or pipeline release, roll back to the previous version. If it built up "
            "gradually (a new fraud pattern), add a temporary rule for the pattern and fast-track the challenger.",
        )
    biz = month["business"]
    if biz["false_decline_rate"] > 2 * biz["baseline_false_decline_rate"] and biz["false_declines"] >= 20:
        add(
            "warn",
            f"False declines {biz['false_decline_rate']:.3%} of traffic vs "
            f"{biz['baseline_false_decline_rate']:.3%} at release",
            "Raise the block threshold (send borderline cases to review instead of declining).",
        )
    if not out:
        add("ok", "No drift detected", "No action; continue monitoring.")
    return out


def monitor(df: pd.DataFrame | None = None, model: FraudModel | None = None, save: bool = True) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_transactions() if df is None else df
    model = joblib.load(config.MODEL_PATH) if model is None else model
    X = build_features(df)
    y = df[config.TARGET].to_numpy()
    score = model.score(X)
    masks = split_masks(df)
    raw = _raw_view(df)

    # References: inputs are compared with what the model was fitted on (January-April);
    # scores and metrics with the out-of-sample June hold-out, because scores on the
    # training window itself are optimistic.
    ref_in, ref_out = masks["train"], masks["test"]
    base = ev.evaluate(y[ref_out], score[ref_out], model.review_threshold)
    base_ci = ev.bootstrap_ci(y[ref_out], score[ref_out], model.review_threshold)
    base_biz = ev.business_metrics(
        y[ref_out], score[ref_out], df.loc[ref_out, "amount"], model.review_threshold, model.block_threshold
    )
    base_biz["false_decline_rate"] = base_biz["false_declines"] / base_biz["transactions"]
    base_days = (pd.Timestamp(config.TEST_END) - pd.Timestamp(config.TEST_START)).days
    base_biz["review_queue_per_day"] = base_biz["review_queue"] / base_days

    def perf_status(value: float, name: str) -> str:
        lo = base_ci[name][0]
        return "ok" if value >= lo else "alert" if value < 0.85 * base[name] else "warn"

    months = []
    for label, start, end in LIVE_MONTHS:
        cur = window_mask(df, start, end).to_numpy()
        dq = validate(df[cur].drop(columns=["fraud_pattern"], errors="ignore"))
        drift = feature_drift(raw[ref_in], raw[cur], RAW_NUMERIC, RAW_CATEGORICAL) + feature_drift(
            X[ref_in], X[cur], ENGINEERED, []
        )
        score_psi = psi_numeric(score[ref_out], score[cur])
        flag_rate = float((score[cur] >= model.review_threshold).mean())
        base_flag_rate = float((score[ref_out] >= model.review_threshold).mean())
        m = ev.evaluate(y[cur], score[cur], model.review_threshold)
        biz = ev.business_metrics(
            y[cur], score[cur], df.loc[cur, "amount"], model.review_threshold, model.block_threshold
        )
        biz["false_decline_rate"] = biz["false_declines"] / biz["transactions"]
        biz.update(
            {
                f"baseline_{k}": base_biz[k]
                for k in ("approval_rate", "review_rate", "false_decline_rate", "fraud_value_recall")
            }
        )
        days = (pd.Timestamp(end) - pd.Timestamp(start)).days
        biz["review_queue_per_day"] = biz["review_queue"] / days
        month = {
            "month": label,
            "rows": int(cur.sum()),
            "data_quality": dq.to_dict(),
            "data_drift": drift,
            "prediction_drift": {
                "score_psi": score_psi,
                "mean_score": float(score[cur].mean()),
                "baseline_mean_score": float(score[ref_out].mean()),
                # share of payments at or above each threshold. Almost all scores sit near zero,
                # so PSI on the whole distribution barely moves when the tail grows; the
                # flagged share is the sensitive, label-free signal.
                "flag_rate": flag_rate,
                "baseline_flag_rate": base_flag_rate,
                "flag_rate_change": flag_rate / base_flag_rate - 1,
                "block_rate": float((score[cur] >= model.block_threshold).mean()),
                "baseline_block_rate": float((score[ref_out] >= model.block_threshold).mean()),
                "status": _worst(
                    [
                        status(score_psi, config.PSI_WARN, config.PSI_ALERT),
                        status(abs(flag_rate / base_flag_rate - 1), config.FLAG_RATE_WARN, config.FLAG_RATE_ALERT),
                    ]
                ),
            },
            "performance": {
                **{
                    k: m[k]
                    for k in (
                        "pr_auc",
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1",
                        "false_positive_rate",
                        "tp",
                        "fp",
                        "fn",
                        "tn",
                        "fraud_rate",
                    )
                },
                **{f"baseline_{k}": base[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")},
                **{f"{k}_status": perf_status(m[k], k) for k in ("pr_auc", "precision", "recall", "f1")},
                "labels_available_from": str((pd.Timestamp(end) + pd.Timedelta(days=config.LABEL_DELAY_DAYS)).date()),
            },
            "business": biz,
        }
        if "fraud_pattern" in df:
            month["recall_by_pattern"] = ev.recall_by_group(
                y[cur], score[cur], df.loc[cur, "fraud_pattern"], model.review_threshold
            ).to_dict("records")
        month["actions"] = _actions(month)
        month["status"] = _worst([a["severity"] for a in month["actions"]])
        months.append(month)
        log.info(
            "%s status=%s score_psi=%.3f pr_auc=%.3f precision=%.3f recall=%.3f",
            label,
            month["status"],
            score_psi,
            m["pr_auc"],
            m["precision"],
            m["recall"],
        )

    report = {
        "model_version": model.version,
        "reference": {
            "inputs": [config.TRAIN_START, config.TRAIN_END],
            "scores_and_metrics": [config.TEST_START, config.TEST_END],
        },
        "thresholds": {
            "psi_warn": config.PSI_WARN,
            "psi_alert": config.PSI_ALERT,
            "ks_warn": config.KS_WARN,
            "ks_alert": config.KS_ALERT,
        },
        "baseline": {"metrics": base, "ci95": base_ci, "business": base_biz},
        "months": months,
        "score_histogram": _score_histograms(score, masks, df),
        "remediation": remediation_demo(df, X, y, model, score),
    }
    if save:
        config.REPORTS_DIR.mkdir(exist_ok=True)
        (config.REPORTS_DIR / "monitoring_report.json").write_text(json.dumps(report, indent=2, default=float))
    return report


def _score_histograms(score: np.ndarray, masks: dict, df: pd.DataFrame) -> dict:
    edges = np.linspace(0, 1, 21)
    out = {
        "edges": edges.tolist(),
        "baseline_june": (np.histogram(score[masks["test"]], edges)[0] / masks["test"].sum()).tolist(),
    }
    for label, start, end in LIVE_MONTHS:
        cur = window_mask(df, start, end).to_numpy()
        out[label] = (np.histogram(score[cur], edges)[0] / cur.sum()).tolist()
    return out


def remediation_demo(df, X, y, champion: FraudModel, champion_score: np.ndarray) -> dict:
    """Demonstrate the two main actions on September, using only data available at the time.

    Labels up to the end of August exist by 8 September. So on September traffic we compare:
      1. the champion unchanged;
      2. the champion with thresholds re-tuned on August (cheap, fixes precision only);
      3. a challenger trained through July (same warm-up rule), calibrated on August, run in shadow.
    """
    fit = window_mask(df, config.TRAIN_START, "2026-08-01").to_numpy()
    tune = window_mask(df, "2026-08-08", "2026-09-01").to_numpy()
    shadow = window_mask(df, "2026-09-08", "2026-10-01").to_numpy()
    amount = df.loc[shadow, "amount"]

    pipe = make_gbm({0: 1.0, 1: float(np.sqrt((1 - y[fit].mean()) / y[fit].mean()))}, 0.05, 300)
    pipe.fit(X[fit], y[fit])
    challenger = build_model(pipe, X[fit], X[tune], y[tune], version="1.1.0-challenger")
    challenger_score = challenger.score(X[shadow])
    retuned = policy_thresholds(y[tune], champion_score[tune])

    def row(name, s, review, block):
        m = ev.evaluate(y[shadow], s, review)
        b = ev.business_metrics(y[shadow], s, amount, review, block)
        return {
            "model": name,
            **{k: m[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "fp", "fn")},
            "review_threshold": review,
            "block_threshold": block,
            "false_declines": b["false_declines"],
            "review_queue": b["review_queue"],
            "fraud_loss_amount": b["fraud_loss_amount"],
            "total_cost": b["total_cost"],
        }

    return {
        "window": ["2026-09-08", "2026-10-01"],
        "rows": [
            row(
                "champion v1.0.0 (unchanged)",
                champion_score[shadow],
                champion.review_threshold,
                champion.block_threshold,
            ),
            row("champion v1.0.0, thresholds re-tuned on August", champion_score[shadow], *retuned),
            row(
                "challenger v1.1.0 (trained to end of July, shadow)",
                challenger_score,
                challenger.review_threshold,
                challenger.block_threshold,
            ),
        ],
    }


if __name__ == "__main__":
    r = monitor()
    for mth in r["months"]:
        p = mth["performance"]
        print(
            f"\n{mth['month']}  [{mth['status'].upper()}]  score PSI {mth['prediction_drift']['score_psi']:.3f}  "
            f"PR-AUC {p['pr_auc']:.3f}  precision {p['precision']:.3f}  recall {p['recall']:.3f}"
        )
        for a in mth["actions"]:
            print(f"   - [{a['severity']}] {a['issue']}\n       -> {a['action']}")
    print("\nSeptember remediation:")
    print(pd.DataFrame(r["remediation"]["rows"]).round(3).to_string(index=False))
