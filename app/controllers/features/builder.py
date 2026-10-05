"""Point-in-time feature engineering.

Every behavioural feature for a transaction is computed only from transactions that
happened strictly before it (same device or same merchant), so the same numbers
could be produced at serving time. The only label-based feature (merchant fraud
rate) additionally waits LABEL_DELAY_DAYS, because fraud labels arrive late.

Fields that are never fed to the model directly:
  request_id      unique per row, no signal
  device_id       identifier / quasi-PII, would be memorised -> used only for velocity/history
  merchant_id     identifier, high cardinality -> used only for velocity/history/risk
  merchant_city   mostly redundant with merchant_state, higher cardinality
  mcc_title       duplicate of mcc_code
  currency_code   constant
  request_status  outcome of the current payment, not known when the fraud decision
                  is made (leakage) -> only the status of EARLIER payments is used
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.utils import config

CATEGORICAL = ["service_type", "merchant_type", "mcc_code", "issuer_bank", "merchant_state"]
NUMERIC = [
    # amount and time
    "log_amount",
    "amount_is_round",
    "hour",
    "day_of_week",
    "is_night",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    # device behaviour
    "dev_cnt_10m",
    "dev_cnt_1h",
    "dev_cnt_24h",
    "dev_cnt_7d",
    "dev_amt_sum_1h",
    "dev_amt_sum_24h",
    "dev_secs_since_last",
    "dev_cnt_30d",
    "dev_age_days",
    "dev_logamt_z",
    "dev_amt_ratio",
    "dev_new_merchant",
    "dev_new_merchants_24h",
    "dev_new_service",
    "dev_fail_cnt_1h",
    "dev_fail_cnt_24h",
    # merchant behaviour
    "mer_cnt_1h",
    "mer_cnt_30d",
    "mer_logamt_z",
    "mer_fraud_rate_lag",
]
FEATURES = NUMERIC + CATEGORICAL

_KEY_BASE = 10**10  # larger than any epoch-second value plus any window
PRIOR_FRAUD_RATE = 0.01  # prior for smoothing the merchant fraud rate
PRIOR_STRENGTH = 50
MINUTE, HOUR, DAY = 60, 3600, 86400
LOOKBACK_DAYS = 30  # fixed history window for all behavioural features


class _Grouped:
    """Rows of one entity type (device or merchant) laid out contiguously in time order."""

    def __init__(self, codes: np.ndarray, secs: np.ndarray):
        n = len(codes)
        self.order = np.lexsort((np.arange(n), secs, codes))
        g, t = codes[self.order].astype(np.int64), secs[self.order]
        idx = np.arange(n)
        first = np.r_[True, g[1:] != g[:-1]]
        self.start = np.maximum.accumulate(np.where(first, idx, 0))  # first row of own group
        self.key = g * _KEY_BASE + t
        self.idx = idx

    def _excl_cumsum(self, values: np.ndarray) -> np.ndarray:
        # NaN counts as zero: one bad value must not poison every later row's running sum
        return np.r_[0.0, np.cumsum(np.nan_to_num(values[self.order].astype(float)))]

    def _back(self, sorted_values: np.ndarray) -> np.ndarray:
        out = np.empty(len(sorted_values), dtype=float)
        out[self.order] = sorted_values
        return out

    def window(self, seconds: int, values: np.ndarray | None = None) -> np.ndarray:
        """Count (or sum of `values`) over this entity's earlier rows in [t - seconds, t)."""
        lo = np.maximum(np.searchsorted(self.key, self.key - seconds, side="left"), self.start)
        if values is None:
            return self._back(self.idx - lo)
        cs = self._excl_cumsum(values)
        return self._back(cs[self.idx] - cs[lo])

    def prior_older_than(self, seconds: int, values: np.ndarray | None = None) -> np.ndarray:
        """Count (or sum) over this entity's rows with time < t - seconds."""
        hi = np.maximum(np.searchsorted(self.key, self.key - seconds, side="left"), self.start)
        if values is None:
            return self._back(hi - self.start)
        cs = self._excl_cumsum(values)
        return self._back(cs[hi] - cs[self.start])

    def secs_since_last(self, secs: np.ndarray) -> np.ndarray:
        t = secs[self.order].astype(float)
        prev = np.r_[np.nan, t[:-1]]
        prev[self.idx == self.start] = np.nan
        return self._back(t - prev)

    def first_seen(self, secs: np.ndarray) -> np.ndarray:
        t = secs[self.order]
        return self._back(t[self.start])


def _zscore(value: np.ndarray, n: np.ndarray, s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """z-score of `value` against a history summarised by count, sum and sum of squares."""
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = s1 / n
        var = np.maximum(s2 / n - mean**2, 0)
        z = (value - mean) / np.sqrt(var + 0.25)  # +0.25 keeps short histories from exploding
    return np.where(n >= 3, z, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model feature frame for `df` (raw assignment schema), same index.

    `df` must contain the full history needed for the behavioural features; rows
    are processed in (request_time, request_id) order regardless of input order.
    """
    # work on a positional index so duplicate or unordered index labels cannot duplicate rows
    raw = df.reset_index(drop=True)
    if not pd.api.types.is_datetime64_any_dtype(raw["request_time"]):
        raw["request_time"] = pd.to_datetime(raw["request_time"], format="mixed")
    raw = raw.sort_values(["request_time", "request_id"], kind="stable")
    ts = raw["request_time"]
    secs = (ts.astype("int64") // 10**9).to_numpy()
    amount = raw["amount"].to_numpy(dtype=float)
    log_amount = np.log1p(np.clip(amount, 0, None))

    out = pd.DataFrame(index=raw.index)
    hour = ts.dt.hour.to_numpy()
    out["log_amount"] = log_amount
    out["amount_is_round"] = (np.round(amount) % 500 == 0).astype(float)
    out["hour"] = hour.astype(float)
    out["day_of_week"] = ts.dt.dayofweek.to_numpy().astype(float)
    out["is_night"] = ((hour >= 23) | (hour < 6)).astype(float)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dev_code = pd.factorize(raw["device_id"].fillna("__missing__"))[0]
    mer_code = pd.factorize(raw["merchant_id"].fillna("__missing__"))[0]
    dev = _Grouped(dev_code, secs)
    mer = _Grouped(mer_code, secs)

    # ---- device velocity and recent history
    # All history features use a fixed LOOKBACK window rather than "everything so far".
    # Unbounded history grows with calendar time, so such features drift by construction
    # and would need unbounded storage in a feature store. Every feature below can be
    # rebuilt from the last LOOKBACK_DAYS + LABEL_DELAY_DAYS days of transactions, with one
    # exception: dev_age_days needs the device's first-seen time (one stored value per device).
    #
    # Ordering: "earlier" means earlier in (request_time, request_id) order, so two
    # payments in the same second are ordered by id. Windows are [t - w, t).
    look = LOOKBACK_DAYS * DAY
    out["dev_cnt_10m"] = dev.window(10 * MINUTE)
    out["dev_cnt_1h"] = dev.window(HOUR)
    out["dev_cnt_24h"] = dev.window(DAY)
    out["dev_cnt_7d"] = dev.window(7 * DAY)
    out["dev_amt_sum_1h"] = dev.window(HOUR, amount)
    out["dev_amt_sum_24h"] = dev.window(DAY, amount)
    # capped at the lookback; a device with no earlier payment also gets the cap
    # (dev_cnt_30d == 0 tells the two cases apart)
    out["dev_secs_since_last"] = np.fmin(dev.secs_since_last(secs), look)
    n_recent = dev.window(look)
    out["dev_cnt_30d"] = n_recent
    # days since the device was first seen, capped at the lookback so the feature stays
    # stationary. Needs the device's first-seen time, which may be older than the lookback.
    out["dev_age_days"] = np.minimum((secs - dev.first_seen(secs)) / DAY, LOOKBACK_DAYS)
    out["dev_logamt_z"] = _zscore(log_amount, n_recent, dev.window(look, log_amount), dev.window(look, log_amount**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dev_amt_ratio"] = np.where(n_recent > 0, amount / (dev.window(look, amount) / n_recent), np.nan)

    # device has not paid this merchant / used this payment method in the lookback window
    svc_code = pd.factorize(raw["service_type"].fillna("__missing__"))[0]
    dev64 = dev_code.astype(np.int64)
    since_pair = _Grouped(dev64 * (mer_code.max() + 1) + mer_code, secs).secs_since_last(secs)
    since_svc = _Grouped(dev64 * (svc_code.max() + 1) + svc_code, secs).secs_since_last(secs)
    new_merchant = (np.isnan(since_pair) | (since_pair > look)).astype(float)
    out["dev_new_merchant"] = new_merchant
    out["dev_new_merchants_24h"] = dev.window(DAY, new_merchant)
    out["dev_new_service"] = (np.isnan(since_svc) | (since_svc > look)).astype(float)

    # status of EARLIER payments only (the current payment's status is not known yet)
    failed = raw["request_status"].isin(["FAILED", "DECLINED"]).to_numpy().astype(float)
    out["dev_fail_cnt_1h"] = dev.window(HOUR, failed)
    out["dev_fail_cnt_24h"] = dev.window(DAY, failed)

    # ---- merchant velocity, usual ticket size and delayed fraud history
    m_recent = mer.window(look)
    out["mer_cnt_1h"] = mer.window(HOUR)
    out["mer_cnt_30d"] = m_recent
    out["mer_logamt_z"] = _zscore(log_amount, m_recent, mer.window(look, log_amount), mer.window(look, log_amount**2))
    # fraud rate over the 30 days of labels that have matured: [t - delay - 30d, t - delay)
    delay = config.LABEL_DELAY_DAYS * DAY
    if config.TARGET in raw:
        label = raw[config.TARGET].fillna(0).to_numpy(dtype=float)
    else:
        label = np.zeros(len(raw))
    known_n = mer.prior_older_than(delay) - mer.prior_older_than(delay + look)
    known_fraud = mer.prior_older_than(delay, label) - mer.prior_older_than(delay + look, label)
    out["mer_fraud_rate_lag"] = (known_fraud + PRIOR_FRAUD_RATE * PRIOR_STRENGTH) / (known_n + PRIOR_STRENGTH)

    # ---- categoricals, passed through as strings; encoders handle missing / unseen values
    for col in CATEGORICAL:
        values = raw[col]
        if pd.api.types.is_float_dtype(values):  # 5411.0 must encode as "5411"
            values = values.round().astype("Int64")
        out[col] = values.astype("string").astype(object).where(raw[col].notna(), np.nan)

    out = out.sort_index()[FEATURES]
    out.index = df.index
    return out


def load_transactions(path=config.SYNTHETIC_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["request_time"], dtype={"mcc_code": "Int64"})
    return df.sort_values(["request_time", "request_id"], kind="stable").reset_index(drop=True)


def window_mask(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    t = df["request_time"]
    return (t >= pd.Timestamp(start)) & (t < pd.Timestamp(end))
