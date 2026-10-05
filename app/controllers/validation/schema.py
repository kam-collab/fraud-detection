"""Input validation run before scoring and before any retraining.

Two levels:
  errors    the batch / request must be rejected (wrong schema, wrong types, impossible values)
  warnings  the data is usable but something moved (unseen categories, missing-rate jump)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

REQUIRED = {
    "request_id": "string",
    "request_time": "datetime",
    "service_type": "string",
    "device_id": "string",
    "merchant_id": "string",
    "merchant_state": "string",
    "merchant_city": "string",
    "merchant_type": "string",
    "mcc_code": "integer",
    "mcc_title": "string",
    "issuer_bank": "string",
    "currency_code": "string",
    "amount": "number",
    "request_status": "string",
}
NOT_NULL = [
    "request_id",
    "request_time",
    "service_type",
    "device_id",
    "merchant_id",
    "merchant_type",
    "mcc_code",
    "amount",
]
ALLOWED = {
    "service_type": {"upi", "wallet", "imps", "netbanking"},
    "merchant_type": {"low", "medium", "high"},
    "request_status": {"SUCCESS", "FAILED", "DECLINED"},
    "currency_code": {"INR"},
}
AMOUNT_RANGE = (0.0, 10_000_000.0)  # exclusive lower bound
MCC_RANGE = (1, 9999)
# Maximum tolerated share of missing values per nullable column (training level was <= 1%).
MAX_MISSING = {
    "issuer_bank": 0.05,
    "merchant_state": 0.03,
    "merchant_city": 0.03,
    "mcc_title": 0.03,
    "request_status": 0.0,
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}


def validate(df: pd.DataFrame, batch: bool = True) -> ValidationResult:
    """Validate transactions in the raw assignment schema. `batch=False` skips rate checks."""
    res = ValidationResult()

    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        res.errors.append(f"missing columns: {missing_cols}")
        return res
    if df.empty:
        res.errors.append("no rows")
        return res

    # ---- data types
    times = pd.to_datetime(df["request_time"], errors="coerce", format="mixed")
    bad = int((times.isna() & df["request_time"].notna()).sum())
    if bad:
        res.errors.append(f"request_time: {bad} unparseable timestamps")
    amount = pd.to_numeric(df["amount"], errors="coerce")
    bad = int((amount.isna() & df["amount"].notna()).sum())
    if bad:
        res.errors.append(f"amount: {bad} non-numeric values")
    mcc = pd.to_numeric(df["mcc_code"], errors="coerce")
    bad = int(((mcc.isna() & df["mcc_code"].notna()) | (mcc.notna() & (mcc % 1 != 0))).sum())
    if bad:
        res.errors.append(f"mcc_code: {bad} non-integer values")

    # ---- required values
    for col in NOT_NULL:
        n = int(df[col].isna().sum())
        if n:
            res.errors.append(f"{col}: {n} missing values in a required field")

    # ---- ranges and domains
    n = int((amount <= AMOUNT_RANGE[0]).sum())
    if n:
        res.errors.append(f"amount: {n} values <= 0")
    n = int((amount > AMOUNT_RANGE[1]).sum())
    if n:
        res.errors.append(f"amount: {n} values above {AMOUNT_RANGE[1]:,.0f}")
    n = int(((mcc < MCC_RANGE[0]) | (mcc > MCC_RANGE[1])).sum())
    if n:
        res.errors.append(f"mcc_code: {n} values outside {MCC_RANGE}")
    if "fraud_label" in df.columns:
        n = int((~df["fraud_label"].dropna().isin([0, 1])).sum())
        if n:
            res.errors.append(f"fraud_label: {n} values other than 0/1")
    for col, allowed in ALLOWED.items():
        unseen = sorted(set(df[col].dropna().astype(str)) - allowed)
        if unseen:
            # unseen categories do not break scoring (encoders map them to "unknown"),
            # but they usually mean an upstream change, so surface them
            res.warnings.append(f"{col}: unseen values {unseen[:5]}")

    if batch:
        n = int(df["request_id"].duplicated().sum())
        if n:
            res.errors.append(f"request_id: {n} duplicates")
        for col, limit in MAX_MISSING.items():
            rate = float(df[col].isna().mean())
            if rate > limit:
                res.warnings.append(f"{col}: {rate:.1%} missing (limit {limit:.0%})")
        if times.notna().any() and not times.dropna().is_monotonic_increasing:
            res.warnings.append("request_time: rows are not in time order (they will be sorted)")
    return res
