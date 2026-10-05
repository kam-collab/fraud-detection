"""Train, select, calibrate and evaluate the fraud model.

Windows (see config): January is feature warm-up, fit on February-April (from 31 January),
select/calibrate/set thresholds on May,
report once on the untouched June hold-out. Nothing about June influences any choice.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.controllers.evaluation import metrics as ev
from app.controllers.features.builder import (
    CATEGORICAL,
    FEATURES,
    NUMERIC,
    build_features,
    load_transactions,
    window_mask,
)
from app.controllers.training.model import FraudModel, _logit, make_gbm, make_logreg
from app.utils import config
from app.utils.logger import get_logger

log = get_logger(__name__)
MODEL_VERSION = "1.0.0"


def split_masks(df: pd.DataFrame) -> dict:
    return {
        "train": window_mask(df, config.TRAIN_START, config.TRAIN_END).to_numpy(),
        "valid": window_mask(df, config.VALID_START, config.VALID_END).to_numpy(),
        "test": window_mask(df, config.TEST_START, config.TEST_END).to_numpy(),
        "live": window_mask(df, config.LIVE_START, config.LIVE_END).to_numpy(),
    }


def candidates(pos_ratio: float) -> dict:
    """Baseline + tree model, each under different class-imbalance treatments."""
    sqrt_w = {0: 1.0, 1: float(np.sqrt(pos_ratio))}
    out = {
        "logreg | no weighting": make_logreg(None),
        "logreg | balanced weights": make_logreg("balanced"),
    }
    for label, weight in (("no weighting", None), ("sqrt weights", sqrt_w), ("balanced weights", "balanced")):
        for lr, iters in ((0.05, 300), (0.05, 600), (0.1, 300)):
            out[f"gbm | {label} | lr={lr} trees={iters}"] = make_gbm(weight, lr, iters)
    return out


def fit_platt(raw_valid: np.ndarray, y_valid: np.ndarray) -> tuple[float, float]:
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_valid).reshape(-1, 1), y_valid)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def reference_values(X_train: pd.DataFrame) -> dict:
    """Typical value of each feature on the training window, used for explanations."""
    ref = {f: float(np.nanmedian(X_train[f].to_numpy(dtype=float))) for f in NUMERIC}
    ref.update({f: str(X_train[f].mode(dropna=True).iloc[0]) for f in CATEGORICAL})
    return ref


def build_model(
    pipeline, X_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: np.ndarray, version: str = MODEL_VERSION
) -> FraudModel:
    """Wrap a fitted pipeline: calibrate on the validation window and set the policy thresholds."""
    a, b = fit_platt(pipeline.predict_proba(X_valid)[:, 1], y_valid)
    model = FraudModel(pipeline=pipeline, calib_a=a, calib_b=b, version=version, reference=reference_values(X_train))
    model.review_threshold, model.block_threshold = policy_thresholds(y_valid, model.score(X_valid))
    return model


def policy_thresholds(y_valid: np.ndarray, score_valid: np.ndarray) -> tuple[float, float]:
    review = ev.threshold_for_precision(y_valid, score_valid, config.REVIEW_MIN_PRECISION)
    block = max(ev.threshold_for_precision(y_valid, score_valid, config.BLOCK_MIN_PRECISION), review)
    return review, block


def train(df: pd.DataFrame | None = None, save: bool = True) -> dict:
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_transactions() if df is None else df
    X = build_features(df)
    y = df[config.TARGET].to_numpy()
    m = split_masks(df)
    Xtr, ytr, Xva, yva, Xte, yte = (
        X[m["train"]],
        y[m["train"]],
        X[m["valid"]],
        y[m["valid"]],
        X[m["test"]],
        y[m["test"]],
    )
    log.info(
        "train=%d (fraud %d) valid=%d (fraud %d) test=%d (fraud %d)",
        len(ytr),
        ytr.sum(),
        len(yva),
        yva.sum(),
        len(yte),
        yte.sum(),
    )

    # ---- model selection on the validation window (PR-AUC)
    pos_ratio = (1 - ytr.mean()) / ytr.mean()
    fitted, table = {}, []
    for name, pipe in candidates(pos_ratio).items():
        pipe.fit(Xtr, ytr)
        fitted[name] = pipe
        raw = pipe.predict_proba(Xva)[:, 1]
        r = ev.ranking_metrics(yva, raw)
        table.append(
            {
                "model": name,
                "valid_pr_auc": r["pr_auc"],
                "valid_roc_auc": r["roc_auc"],
                "valid_brier_uncalibrated": r["brier"],
            }
        )
        log.info("%-48s valid PR-AUC %.4f", name, r["pr_auc"])
    table = sorted(table, key=lambda d: -d["valid_pr_auc"])
    best_gbm = next(d["model"] for d in table if d["model"].startswith("gbm"))
    best_lr = next(d["model"] for d in table if d["model"].startswith("logreg"))

    def finalise(name: str) -> FraudModel:
        return build_model(fitted[name], Xtr, Xva, yva)

    champion, baseline = finalise(best_gbm), finalise(best_lr)

    # ---- one-off report on the June hold-out
    def report(model: FraudModel) -> dict:
        s_va, s_te = model.score(Xva), model.score(Xte)
        amount = df.loc[m["test"], "amount"].to_numpy()
        out = {
            "review_threshold": model.review_threshold,
            "block_threshold": model.block_threshold,
            "valid": ev.evaluate(yva, s_va, model.review_threshold),
            "test": ev.evaluate(yte, s_te, model.review_threshold),
            "test_ci95": ev.bootstrap_ci(yte, s_te, model.review_threshold),
            "test_block_band": ev.threshold_metrics(yte, s_te, model.block_threshold),
            "test_default_0.5": ev.threshold_metrics(yte, s_te, 0.5),
            "test_max_f1_threshold": ev.threshold_metrics(yte, s_te, ev.threshold_for_max_f1(yva, s_va)),
            "test_business": ev.business_metrics(yte, s_te, amount, model.review_threshold, model.block_threshold),
            "test_pr_curve": ev.pr_curve_points(yte, s_te),
        }
        raw_te = model.raw_score(Xte)
        out["test_brier_uncalibrated"] = ev.ranking_metrics(yte, raw_te)["brier"]
        if "fraud_pattern" in df:
            out["test_recall_by_pattern"] = ev.recall_by_group(
                yte, s_te, df.loc[m["test"], "fraud_pattern"], model.review_threshold
            ).to_dict("records")
        return out

    result = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data": {
            "source": "synthetic (app/controllers/data_generation/generator.py)",
            "seed": config.SEED,
            "rows": int(len(df)),
            "fraud_rate": float(y.mean()),
        },
        "windows": {
            "train": [config.TRAIN_START, config.TRAIN_END],
            "valid": [config.VALID_START, config.VALID_END],
            "test": [config.TEST_START, config.TEST_END],
            "label_delay_days": config.LABEL_DELAY_DAYS,
        },
        "rows": {
            k: {"n": int(v.sum()), "frauds": int(y[v].sum()), "fraud_rate": float(y[v].mean())} for k, v in m.items()
        },
        "selection_table": table,
        "champion": {"name": best_gbm, **report(champion)},
        "baseline": {"name": best_lr, **report(baseline)},
        "policy": {
            "review_min_precision": config.REVIEW_MIN_PRECISION,
            "block_min_precision": config.BLOCK_MIN_PRECISION,
        },
        "features": FEATURES,
    }

    if save:
        config.MODELS_DIR.mkdir(exist_ok=True)
        config.REPORTS_DIR.mkdir(exist_ok=True)
        joblib.dump(champion, config.MODEL_PATH)
        joblib.dump(baseline, config.MODELS_DIR / "baseline_model.joblib")
        (config.REPORTS_DIR / "training_report.json").write_text(json.dumps(result, indent=2))
        card = {k: result[k] for k in ("model_version", "trained_at", "data", "windows", "policy", "features")}
        card.update(
            {
                "algorithm": best_gbm,
                "thresholds": {"review": champion.review_threshold, "block": champion.block_threshold},
                "release_baseline": {
                    k: result["champion"]["test"][k]
                    for k in ("pr_auc", "roc_auc", "precision", "recall", "f1", "false_positive_rate")
                },
                "release_baseline_ci95": result["champion"]["test_ci95"],
                "rollback": "Keep the previous models/fraud_model.joblib as fraud_model.prev.joblib; "
                "roll back by restoring it and restarting the API (see docs/MONITORING.md).",
            }
        )
        config.MODEL_CARD_PATH.write_text(json.dumps(card, indent=2))
        log.info("saved %s", config.MODEL_PATH)
    return {"report": result, "champion": champion, "baseline": baseline, "X": X, "y": y, "masks": m, "df": df}


if __name__ == "__main__":
    r = train()["report"]
    for who in ("baseline", "champion"):
        t = r[who]["test"]
        print(
            f"{who:9s} {r[who]['name']}\n   June hold-out: PR-AUC {t['pr_auc']:.3f}  ROC-AUC {t['roc_auc']:.3f}  "
            f"precision {t['precision']:.3f}  recall {t['recall']:.3f}  F1 {t['f1']:.3f}  "
            f"TP {t['tp']} FP {t['fp']} FN {t['fn']} TN {t['tn']}"
        )
