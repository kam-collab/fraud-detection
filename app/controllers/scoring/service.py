"""Online scoring: validate -> build features from recent history -> score -> decide -> explain.

The service reuses `build_features`, the exact function used for training, on the slice of
history that can influence one transaction (its device's rows and its merchant's recent
rows). That removes the usual source of training/serving skew: there is one feature
implementation, not two. Here the history lives in memory; in production it would be an
online feature store with 37 days of retention (30-day lookback + 7-day label delay) plus
one first-seen timestamp per device.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from app.controllers.explanations import explainer
from app.controllers.features.builder import DAY, LOOKBACK_DAYS, build_features, load_transactions
from app.controllers.training.model import BLOCK, REVIEW, FraudModel
from app.controllers.validation.schema import validate
from app.utils import config, metrics
from app.utils.logger import get_logger, trace_id_var

log = get_logger(__name__)
RAW = [c for c in config.RAW_COLUMNS]
PUBLIC = [
    "request_id",
    "request_time",
    "service_type",
    "merchant_state",
    "merchant_city",
    "merchant_type",
    "mcc_code",
    "mcc_title",
    "issuer_bank",
    "currency_code",
    "amount",
]
PREDICTION_LOG = config.ROOT / "logs" / "predictions.jsonl"
GATEWAY_TZ = "Asia/Kolkata"


class ValidationFailed(Exception):
    def __init__(self, result):
        super().__init__("; ".join(result.errors))
        self.result = result


def _clean(value):
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, np.generic):
        return value.item()
    return value


class ScoringService:
    def __init__(self, model: FraudModel | None = None, df: pd.DataFrame | None = None):
        self.model: FraudModel = joblib.load(config.MODEL_PATH) if model is None else model
        self.df = load_transactions() if df is None else df
        self.X = build_features(self.df)
        self.scores = self.model.score(self.X)
        self.bands = self.model.decide(self.scores)
        self._dev_idx = self.df.groupby("device_id").indices
        self._mer_idx = self.df.groupby("merchant_id").indices
        self._pos = pd.Series(np.arange(len(self.df)), index=self.df["request_id"].to_numpy())
        self._secs = (self.df["request_time"].astype("int64") // 10**9).to_numpy()
        self.now: pd.Timestamp = self.df["request_time"].max()
        self.session = pd.DataFrame(columns=RAW)  # transactions scored through the API
        self.session_results: dict[str, dict] = {}
        self.decisions: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._counter = 0
        metrics.MODEL_INFO.labels(version=self.model.version).set(1)
        metrics.REVIEW_THRESHOLD.set(self.model.review_threshold)
        metrics.BLOCK_THRESHOLD.set(self.model.block_threshold)
        log.info("scoring service ready", extra={"rows": len(self.df), "model_version": self.model.version})

    # ------------------------------------------------------------------ scoring
    def _history(self, device_id: str, merchant_id: str, at: pd.Timestamp, before_pos: int | None) -> pd.DataFrame:
        """Rows that can influence this transaction's features."""
        at_s = int(at.value // 10**9)
        dev = self._dev_idx.get(device_id, np.array([], dtype=int))
        mer = self._mer_idx.get(merchant_id, np.array([], dtype=int))
        horizon = at_s - (LOOKBACK_DAYS + config.LABEL_DELAY_DAYS) * DAY
        recent = np.union1d(dev, mer)
        recent = recent[self._secs[recent] >= horizon]
        # plus the device's first row ever: dev_age_days needs its first-seen time
        idx = np.union1d(recent, dev[:1])
        idx = idx[idx < before_pos] if before_pos is not None else idx[self._secs[idx] <= at_s]
        hist = self.df.iloc[idx][RAW]
        if len(self.session):
            s = self.session
            extra = s[((s["device_id"] == device_id) | (s["merchant_id"] == merchant_id)) & (s["request_time"] <= at)]
            hist = pd.concat([hist, extra], ignore_index=True)
        return hist

    def score(self, txn: dict) -> dict:
        txn = dict(txn)
        with self._lock:
            self._counter += 1
            if not txn.get("request_id"):
                txn["request_id"] = f"TXN9{self._counter:06d}"
            if not txn.get("request_time"):
                self.now = self.now + pd.Timedelta(seconds=1)
                txn["request_time"] = self.now
        row = pd.DataFrame([{c: txn.get(c) for c in RAW if c != config.TARGET}])
        result = validate(row, batch=False)
        if not result.ok:
            metrics.VALIDATION_FAILURES.inc()
            log.warning("validation failed", extra={"errors": result.errors})
            raise ValidationFailed(result)

        ts = pd.Timestamp(row.loc[0, "request_time"])
        if ts.tzinfo is not None:  # stored times are naive gateway-local (IST)
            ts = ts.tz_convert(GATEWAY_TZ).tz_localize(None)
        row["request_time"] = pd.Series([ts], dtype="datetime64[ns]")
        row["amount"] = row["amount"].astype(float)
        row["mcc_code"] = row["mcc_code"].astype("Int64")
        row[config.TARGET] = 0  # label unknown at decision time
        at = row.loc[0, "request_time"]
        known = self._pos.get(txn["request_id"])  # replay of a stored transaction
        hist = self._history(txn["device_id"], txn["merchant_id"], at, None if known is None else int(known))
        frame = pd.concat([hist.astype({"mcc_code": "Int64"}), row[RAW]], ignore_index=True)
        feats = build_features(frame).iloc[[-1]]
        score = float(self.model.score(feats)[0])
        band = str(self.model.decide(np.array([score]))[0])
        reasons = self.model.contributions(feats)
        raw = {k: _clean(v) for k, v in row.iloc[0].to_dict().items()}
        explanation = explainer.explain(score, band, reasons, raw, use_llm=False)

        out = {
            "request_id": txn["request_id"],
            "score": score,
            "band": band,
            "thresholds": {"review": self.model.review_threshold, "block": self.model.block_threshold},
            "model_version": self.model.version,
            "reasons": reasons,
            "explanation": explanation,
            "warnings": result.warnings,
        }
        if known is None:
            with self._lock:
                self.session = pd.concat([self.session, row[RAW]], ignore_index=True)
                self.session_results[txn["request_id"]] = {"raw": raw, "features": feats, **out}
        metrics.SCORES.observe(score)
        metrics.DECISIONS.labels(band=band).inc()
        metrics.EXPLANATIONS.labels(source=explanation["source"]).inc()
        self._log_prediction(out, feats)
        return out

    def _log_prediction(self, out: dict, feats: pd.DataFrame) -> None:
        """Append-only prediction log: the raw material for later drift and performance analysis."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "trace_id": trace_id_var.get(),
            "request_id": out["request_id"],
            "model_version": out["model_version"],
            "score": round(out["score"], 6),
            "band": out["band"],
            "features": {k: _clean(v) for k, v in feats.iloc[0].to_dict().items()},
        }
        try:
            PREDICTION_LOG.parent.mkdir(exist_ok=True)
            with PREDICTION_LOG.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError as e:
            log.warning("prediction log write failed: %s", e)
        log.info(
            "scored", extra={"request_id": out["request_id"], "score": round(out["score"], 4), "band": out["band"]}
        )

    # ------------------------------------------------------------- explanations
    def explain(self, request_id: str, use_llm: bool | None = None) -> dict | None:
        if request_id in self.session_results:
            r = self.session_results[request_id]
            score, band, reasons, raw = r["score"], r["band"], r["reasons"], r["raw"]
        elif request_id in self._pos.index:
            i = int(self._pos[request_id])
            score, band = float(self.scores[i]), str(self.bands[i])
            reasons = self.model.contributions(self.X.iloc[[i]])
            raw = {k: _clean(v) for k, v in self.df.iloc[i][RAW].to_dict().items()}
        else:
            return None
        explanation = explainer.explain(score, band, reasons, raw, use_llm=use_llm)
        metrics.EXPLANATIONS.labels(source=explanation["source"]).inc()
        return {
            "request_id": request_id,
            "score": score,
            "band": band,
            "reasons": reasons,
            "reason_phrases": explainer.reason_phrases(reasons, raw),
            "explanation": explanation,
        }

    # ------------------------------------------------------------- review queue
    def review_queue(self, limit: int = 25, offset: int = 0, days: int = 3) -> dict:
        """Transactions in the review band over the most recent days, highest score first."""
        since = self.df["request_time"].max() - pd.Timedelta(days=days)
        mask = (self.bands == REVIEW) & (self.df["request_time"] >= since).to_numpy()
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-self.scores[idx])]
        items = []
        for i in idx[offset : offset + limit]:
            row = self.df.iloc[i]
            rid = row["request_id"]
            decision = self.decisions.get(rid)
            item = {c: _clean(row[c]) for c in PUBLIC}
            item.update(
                {
                    "score": float(self.scores[i]),
                    "band": REVIEW,
                    "decision": decision["decision"] if decision else None,
                    # the true label is only revealed once the analyst has decided
                    "actual_label": int(row[config.TARGET]) if decision else None,
                }
            )
            items.append(item)
        pending = int(sum(1 for i in idx if self.df.iloc[i]["request_id"] not in self.decisions))
        return {"total": int(len(idx)), "pending": pending, "since": _clean(since), "items": items}

    def decide(self, request_id: str, decision: str) -> dict | None:
        if request_id not in self._pos.index:
            return None
        i = int(self._pos[request_id])
        if self.bands[i] != REVIEW:  # only transactions in the review band are decided here
            return None
        self.decisions[request_id] = {
            "decision": decision,
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        metrics.REVIEW_DECISIONS.labels(decision=decision).inc()
        actual = int(self.df.iloc[i][config.TARGET])
        log.info("review decision", extra={"request_id": request_id, "decision": decision})
        return {
            "request_id": request_id,
            "decision": decision,
            "actual_label": actual,
            "correct": (decision == "decline") == (actual == 1),
        }

    # ------------------------------------------------------------------ samples
    def sample(self, kind: str = "random", seed: int | None = None) -> dict:
        """A stored September transaction to replay through the scoring form."""
        rng = np.random.default_rng(seed)
        live = (self.df["request_time"] >= pd.Timestamp("2026-09-01")).to_numpy()
        y = self.df[config.TARGET].to_numpy()
        pick = {
            "fraud": live & (y == 1),
            "genuine": live & (y == 0),
            "blocked": live & (self.bands == BLOCK),
            "review": live & (self.bands == REVIEW),
        }.get(kind, live)
        i = int(rng.choice(np.flatnonzero(pick)))
        row = self.df.iloc[i]
        return {c: _clean(row[c]) for c in RAW if c not in (config.TARGET, "request_status")}

    # ----------------------------------------------------------------- overview
    def overview(self) -> dict:
        """EDA summaries on the development period (January-June) for the dashboard."""
        if hasattr(self, "_overview"):
            return self._overview
        df = self.df
        dev = df[df["request_time"] < pd.Timestamp(config.LIVE_START)].copy()
        X = self.X.loc[dev.index]
        y = dev[config.TARGET]

        def rate_by(series, name, top=None):
            g = pd.DataFrame({"k": series.to_numpy(), "y": y.to_numpy()}).groupby("k")["y"].agg(["size", "sum", "mean"])
            g = g.sort_values("size", ascending=False).head(top) if top else g
            return [
                {
                    name: _clean(k),
                    "transactions": int(r["size"]),
                    "frauds": int(r["sum"]),
                    "fraud_rate": float(r["mean"]),
                }
                for k, r in g.iterrows()
            ]

        amount_bins = [0, 500, 2000, 5000, 20000, 50000, np.inf]
        amount_labels = ["<500", "500-2k", "2k-5k", "5k-20k", "20k-50k", ">50k"]
        velocity = pd.cut(X["dev_cnt_1h"], [-1, 0, 1, 2, np.inf], labels=["0", "1", "2", "3+"])
        monthly = df.groupby(df["request_time"].dt.strftime("%Y-%m"))[config.TARGET].agg(["size", "sum", "mean"])
        self._overview = {
            "period": [str(dev["request_time"].min().date()), str(dev["request_time"].max().date())],
            "transactions": int(len(dev)),
            "frauds": int(y.sum()),
            "fraud_rate": float(y.mean()),
            "fraud_amount_share": float(dev.loc[y == 1, "amount"].sum() / dev["amount"].sum()),
            "monthly": [
                {"month": m, "transactions": int(r["size"]), "frauds": int(r["sum"]), "fraud_rate": float(r["mean"])}
                for m, r in monthly.iterrows()
            ],
            "by_service_type": rate_by(dev["service_type"], "service_type"),
            "by_merchant_type": rate_by(dev["merchant_type"], "merchant_type"),
            "by_request_status": rate_by(dev["request_status"], "request_status"),
            "by_mcc": sorted(rate_by(dev["mcc_title"].fillna("Unknown"), "mcc_title"), key=lambda d: -d["fraud_rate"]),
            "by_hour": sorted(rate_by(dev["request_time"].dt.hour, "hour"), key=lambda d: d["hour"]),
            "by_amount": [
                d
                for lab in amount_labels
                for d in rate_by(pd.cut(dev["amount"], amount_bins, labels=amount_labels).astype(str), "amount")
                if d["amount"] == lab
            ],
            "by_device_velocity_1h": sorted(
                rate_by(velocity.astype(str), "payments_last_hour"), key=lambda d: d["payments_last_hour"]
            ),
            "by_new_merchant": rate_by(
                X["dev_new_merchant"].map({0.0: "seen before", 1.0: "new for device"}), "merchant_relationship"
            ),
        }
        return self._overview
