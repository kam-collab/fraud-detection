"""Point-in-time correctness of `build_features`.

The core of this file is `reference_features`: a deliberately slow, loop-based
re-implementation of every behavioural feature that shares no code with builder.py.
Semantics, taken from the builder's docstrings and comments:

* history of a row = rows of the same device (or merchant) that come strictly earlier
  in (request_time, request_id) order;
* a window of w seconds covers earlier rows with time in [t - w, t];  (rows at the same
  second but earlier in id order are "earlier")
* history features look back 30 days;
* the merchant fraud rate only uses labels older than LABEL_DELAY_DAYS:
  rows in [t - delay - 30d, t - delay), smoothed towards a 1% prior with strength 50.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.controllers.features.builder import CATEGORICAL, FEATURES, NUMERIC, build_features
from app.utils import config
from tests.conftest import make_frame, make_txn

MIN, HOUR, DAY = 60, 3600, 86400
LOOK = 30 * DAY
DELAY = config.LABEL_DELAY_DAYS * DAY
PRIOR_RATE, PRIOR_N = 0.01, 50

BEHAVIOURAL = [
    "dev_cnt_10m",
    "dev_cnt_1h",
    "dev_cnt_24h",
    "dev_cnt_7d",
    "dev_cnt_30d",
    "dev_amt_sum_1h",
    "dev_amt_sum_24h",
    "dev_secs_since_last",
    "dev_age_days",
    "dev_logamt_z",
    "dev_amt_ratio",
    "dev_new_merchant",
    "dev_new_merchants_24h",
    "dev_new_service",
    "dev_fail_cnt_1h",
    "dev_fail_cnt_24h",
    "mer_cnt_1h",
    "mer_cnt_30d",
    "mer_logamt_z",
    "mer_fraud_rate_lag",
]


# --------------------------------------------------------------------------- reference
def _zscore(value: float, history: list[float]) -> float:
    if len(history) < 3:
        return math.nan
    mean = sum(history) / len(history)
    var = sum((h - mean) ** 2 for h in history) / len(history)
    return (value - mean) / math.sqrt(var + 0.25)


def reference_features(df: pd.DataFrame) -> pd.DataFrame:
    """Brute-force behavioural features, one row at a time, in event order."""
    ordered = df.sort_values(["request_time", "request_id"], kind="stable")
    epoch = pd.Timestamp("1970-01-01")
    dev_hist: dict[str, list[dict]] = {}
    mer_hist: dict[str, list[dict]] = {}
    out = {}
    for label, r in zip(ordered.index, ordered.to_dict("records")):
        t = int((pd.Timestamp(r["request_time"]) - epoch) // pd.Timedelta(seconds=1))
        amount = float(r["amount"])
        la = math.log1p(amount)
        dh = dev_hist.setdefault(r["device_id"], [])
        mh = mer_hist.setdefault(r["merchant_id"], [])

        def last(hist, seconds):
            return [h for h in hist if h["t"] >= t - seconds]

        d10, d1h, d24, d7, d30 = (last(dh, w) for w in (10 * MIN, HOUR, DAY, 7 * DAY, LOOK))
        m30 = last(mh, LOOK)
        matured = [h for h in mh if t - DELAY - LOOK <= h["t"] < t - DELAY]
        new_merchant = float(not any(h["merchant"] == r["merchant_id"] for h in d30))
        new_service = float(not any(h["service"] == r["service_type"] for h in d30))

        out[label] = {
            "dev_cnt_10m": len(d10),
            "dev_cnt_1h": len(d1h),
            "dev_cnt_24h": len(d24),
            "dev_cnt_7d": len(d7),
            "dev_cnt_30d": len(d30),
            "dev_amt_sum_1h": sum(h["amount"] for h in d1h),
            "dev_amt_sum_24h": sum(h["amount"] for h in d24),
            # no previous payment is reported as the 30-day cap (builder uses np.fmin)
            "dev_secs_since_last": min(t - dh[-1]["t"], LOOK) if dh else LOOK,
            "dev_age_days": min((t - dh[0]["t"]) / DAY, 30) if dh else 0.0,
            "dev_logamt_z": _zscore(la, [h["la"] for h in d30]),
            "dev_amt_ratio": amount / (sum(h["amount"] for h in d30) / len(d30)) if d30 else math.nan,
            "dev_new_merchant": new_merchant,
            "dev_new_merchants_24h": sum(h["new_merchant"] for h in d24),
            "dev_new_service": new_service,
            "dev_fail_cnt_1h": sum(h["failed"] for h in d1h),
            "dev_fail_cnt_24h": sum(h["failed"] for h in d24),
            "mer_cnt_1h": len(last(mh, HOUR)),
            "mer_cnt_30d": len(m30),
            "mer_logamt_z": _zscore(la, [h["la"] for h in m30]),
            "mer_fraud_rate_lag": (sum(h["label"] for h in matured) + PRIOR_RATE * PRIOR_N) / (len(matured) + PRIOR_N),
        }
        event = {
            "t": t,
            "amount": amount,
            "la": la,
            "merchant": r["merchant_id"],
            "service": r["service_type"],
            "failed": r["request_status"] in ("FAILED", "DECLINED"),
            "label": r["fraud_label"],
            "new_merchant": new_merchant,
        }
        dh.append(event)
        mh.append(event)
    return pd.DataFrame.from_dict(out, orient="index").loc[df.index]


def dense_frame(n: int = 900, seed: int = 3) -> pd.DataFrame:
    """Few devices and merchants, bursts, exact ties and gaps that sit exactly on every
    window boundary (10 min, 1 h, 24 h, 7 d, 30 d, 37 d), ids NOT in time order."""
    rng = np.random.default_rng(seed)
    centres = rng.integers(0, 80 * DAY, 70)
    offsets = np.array(
        [
            0,
            0,
            0,
            45,
            599,
            600,
            601,
            3599,
            3600,
            3601,
            DAY - 1,
            DAY,
            DAY + 1,
            7 * DAY,
            7 * DAY - 1,
            30 * DAY,
            30 * DAY + 1,
            37 * DAY,
            37 * DAY - 1,
        ]
    )
    secs = rng.choice(centres, n) + rng.choice(offsets, n)
    ids = rng.permutation(n)
    df = pd.DataFrame(
        {
            "request_id": [f"R{i:05d}" for i in ids],
            "request_time": pd.Timestamp("2026-01-01") + pd.to_timedelta(secs, unit="s"),
            "service_type": rng.choice(["upi", "wallet", "imps"], n),
            "device_id": rng.choice([f"D{i}" for i in range(6)], n),
            "merchant_id": rng.choice([f"M{i}" for i in range(5)], n),
            "merchant_state": "Kerala",
            "merchant_city": "Kochi",
            "merchant_type": "low",
            "mcc_code": 5411,
            "mcc_title": "Grocery Stores",
            "issuer_bank": "SBI",
            "currency_code": "INR",
            "amount": np.round(rng.lognormal(6, 1.2, n), 2),
            "request_status": rng.choice(["SUCCESS", "FAILED", "DECLINED"], n, p=[0.7, 0.2, 0.1]),
            "fraud_label": (rng.random(n) < 0.2).astype(int),
        }
    )
    return df.sample(frac=1.0, random_state=seed)  # shuffled rows, non-trivial index


@pytest.fixture(scope="module")
def cases(txns, txn_features):
    dense = dense_frame()
    return {
        "generated": (txn_features, reference_features(txns)),
        "dense": (build_features(dense), reference_features(dense)),
    }


@pytest.mark.parametrize("feature", BEHAVIOURAL)
@pytest.mark.parametrize("dataset", ["generated", "dense"])
def test_behavioural_feature_matches_brute_force(cases, dataset, feature):
    actual, expected = cases[dataset]
    np.testing.assert_allclose(
        actual[feature].to_numpy(dtype=float),
        expected[feature].to_numpy(dtype=float),
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
        err_msg=f"{feature} on {dataset}",
    )


def test_reference_datasets_exercise_every_window(cases):
    """Guard against a vacuous comparison: the windows must actually contain rows."""
    for name, (_, expected) in cases.items():
        for col in (
            "dev_cnt_10m",
            "dev_cnt_1h",
            "dev_cnt_24h",
            "dev_fail_cnt_24h",
            "mer_cnt_1h",
            "dev_new_merchants_24h",
        ):
            assert expected[col].max() >= 1, f"{col} never non-zero on {name}"
        assert expected["dev_logamt_z"].notna().sum() > 100
        assert expected["mer_fraud_rate_lag"].nunique() > 5


# ------------------------------------------------------------ stateless features
def test_amount_and_time_features_by_hand():
    df = make_frame(
        [
            make_txn("2026-03-07 23:30:00", amount=1000),  # Saturday night, round amount
            make_txn("2026-03-09 05:59:59", amount=999.99),  # Monday, still night
            make_txn("2026-03-09 06:00:00", amount=1234.5),  # Monday morning
            make_txn("2026-03-11 22:59:59", amount=250),  # Wednesday, not yet night
        ]
    )
    X = build_features(df)
    np.testing.assert_allclose(X["log_amount"], np.log1p(df["amount"]))
    assert X["hour"].tolist() == [23, 5, 6, 22]
    assert X["day_of_week"].tolist() == [5, 0, 0, 2]
    assert X["is_night"].tolist() == [1, 1, 0, 0]
    assert X["is_weekend"].tolist() == [1, 0, 0, 0]
    assert X["amount_is_round"].tolist()[0] == 1 and X["amount_is_round"].tolist()[2:] == [0, 0]
    np.testing.assert_allclose(X["hour_sin"], np.sin(2 * np.pi * np.array([23, 5, 6, 22]) / 24))
    np.testing.assert_allclose(X["hour_cos"], np.cos(2 * np.pi * np.array([23, 5, 6, 22]) / 24))


def test_output_has_model_feature_columns_in_order(txn_features, txns):
    assert list(txn_features.columns) == FEATURES == NUMERIC + CATEGORICAL
    assert len(txn_features) == len(txns)
    # identifiers and the current payment's outcome are never model inputs
    assert not {"request_id", "device_id", "merchant_id", "request_status", "fraud_label"} & set(FEATURES)


# ------------------------------------------------------------------ no look-ahead
def test_no_look_ahead_features_do_not_depend_on_later_rows(txns, txn_features):
    cutoff = txns["request_time"].quantile(0.6)
    past = txns[txns["request_time"] <= cutoff]
    assert 0 < len(past) < len(txns)
    alone = build_features(past)
    pd.testing.assert_frame_equal(alone, txn_features.loc[past.index], check_exact=False, rtol=1e-7, atol=1e-6)


def test_no_look_ahead_on_dense_frame_with_ties():
    df = dense_frame()
    full = build_features(df)
    ordered = df.sort_values(["request_time", "request_id"])
    for k in (1, 50, 400, len(df) - 1):  # prefixes in event order, incl. mid-tie cuts
        prefix = ordered.iloc[:k]
        pd.testing.assert_frame_equal(
            build_features(prefix), full.loc[prefix.index], check_exact=False, rtol=1e-7, atol=1e-6
        )


# --------------------------------------------------------------- label leakage
def _merchant_history_frame() -> pd.DataFrame:
    """One merchant, different devices; the target is the last row (10 Mar 12:00)."""
    return make_frame(
        [
            make_txn("2026-01-20 12:00:00", device="A", label=0),  # 49 days old: outside delay + 30 d
            make_txn("2026-02-15 12:00:00", device="B", label=0),  # 23 days old: matured label
            make_txn("2026-03-01 12:00:00", device="C", label=0),  # 9 days old: matured label
            make_txn("2026-03-03 12:00:00", device="D", label=0),  # exactly 7 days old: NOT matured
            make_txn("2026-03-08 12:00:00", device="E", label=0),  # 2 days old
            make_txn("2026-03-10 11:59:00", device="F", label=0),  # a minute earlier
            make_txn("2026-03-10 12:00:00", device="G", label=0),  # target
            make_txn("2026-02-20 12:00:00", device="H", merchant="OTHER", label=0),
        ]
    )


def test_merchant_fraud_rate_uses_only_matured_labels():
    df = _merchant_history_frame()
    target = 6
    base = build_features(df).loc[target]
    assert base["mer_fraud_rate_lag"] == pytest.approx(0.5 / 52)  # two matured rows, no fraud

    recent = df.copy()
    recent.loc[[3, 4, 5, 6], "fraud_label"] = 1  # within the label delay + itself
    pd.testing.assert_series_equal(build_features(recent).loc[target], base)

    for row in (1, 2):  # an older label of the same merchant
        older = df.copy()
        older.loc[row, "fraud_label"] = 1
        changed = build_features(older).loc[target]
        assert changed["mer_fraud_rate_lag"] == pytest.approx(1.5 / 52)
        pd.testing.assert_series_equal(changed.drop("mer_fraud_rate_lag"), base.drop("mer_fraud_rate_lag"))

    for row in (0, 7):  # too old / another merchant
        unrelated = df.copy()
        unrelated.loc[row, "fraud_label"] = 1
        pd.testing.assert_series_equal(build_features(unrelated).loc[target], base)


def test_labels_inside_the_delay_never_reach_any_feature(txns, txn_features):
    """Flipping every label of the final LABEL_DELAY_DAYS changes no feature of any row."""
    recent = txns["request_time"] > txns["request_time"].max() - pd.Timedelta(days=config.LABEL_DELAY_DAYS)
    flipped = txns.copy()
    flipped.loc[recent, "fraud_label"] = 1 - flipped.loc[recent, "fraud_label"]
    assert recent.sum() > 50
    pd.testing.assert_frame_equal(build_features(flipped), txn_features)


def test_older_labels_do_reach_the_merchant_fraud_rate(txns, txn_features):
    """The mirror image of the test above, so that it cannot pass by labels being ignored."""
    flipped = txns.copy()
    flipped["fraud_label"] = 1
    changed = build_features(flipped)
    assert (changed["mer_fraud_rate_lag"] > txn_features["mer_fraud_rate_lag"]).mean() > 0.3
    others = [f for f in FEATURES if f != "mer_fraud_rate_lag"]
    pd.testing.assert_frame_equal(changed[others], txn_features[others])


def test_features_work_without_a_label_column(txns, txn_features):
    """Serving frames may carry no labels: that must behave exactly like 'no known fraud'."""
    head = txns.head(1500)
    X = build_features(head.drop(columns=["fraud_label"]))
    pd.testing.assert_frame_equal(X, build_features(head.assign(fraud_label=0)))
    assert (X["mer_fraud_rate_lag"] <= PRIOR_RATE).all() and (X["mer_fraud_rate_lag"] < PRIOR_RATE).any()


# -------------------------------------------- current status, row order, index
def test_own_request_status_does_not_affect_own_features():
    df = make_frame(
        [
            make_txn("2026-03-10 12:00:00", status="SUCCESS"),
            make_txn("2026-03-10 12:05:00", status="SUCCESS"),
            make_txn("2026-03-10 12:10:00", status="SUCCESS"),
        ]
    )
    base = build_features(df)
    for status in ("FAILED", "DECLINED", None):
        changed = df.copy()
        changed.loc[1, "request_status"] = status
        X = build_features(changed)
        pd.testing.assert_series_equal(X.loc[1], base.loc[1])
        pd.testing.assert_series_equal(X.loc[0], base.loc[0])
    # ... but an earlier failure is visible to the next payment of the device
    changed = df.copy()
    changed.loc[1, "request_status"] = "FAILED"
    X = build_features(changed)
    assert (X.loc[2, "dev_fail_cnt_1h"], X.loc[2, "dev_fail_cnt_24h"]) == (1, 1)
    assert (base.loc[2, "dev_fail_cnt_1h"], base.loc[2, "dev_fail_cnt_24h"]) == (0, 0)


def test_own_status_never_matters_on_generated_data(txns, txn_features):
    """Changing the status of the last payment of every device changes none of those rows."""
    last = txns.groupby("device_id").tail(1).index
    changed = txns.copy()
    changed.loc[last, "request_status"] = "FAILED"
    pd.testing.assert_frame_equal(build_features(changed).loc[last], txn_features.loc[last])


def test_row_order_of_the_input_does_not_matter(txns, txn_features):
    shuffled = txns.sample(frac=1.0, random_state=11)
    X = build_features(shuffled)
    assert X.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(X.loc[txns.index], txn_features)


def test_output_index_matches_an_arbitrary_input_index():
    df = make_frame([make_txn(f"2026-03-10 12:0{i}:00", amount=100 + i) for i in range(5)])
    df.index = ["e", "a", "d", "b", "c"]
    X = build_features(df)
    assert list(X.index) == ["e", "a", "d", "b", "c"]
    assert X["dev_cnt_10m"].tolist() == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(X["log_amount"], np.log1p(df["amount"]))


def test_duplicate_index_labels_do_not_duplicate_rows():
    df = make_frame([make_txn(f"2026-03-10 12:0{i}:00", amount=100 + i) for i in range(4)])
    expected = build_features(df)
    df.index = [0, 0, 1, 2]  # e.g. after pd.concat without ignore_index
    X = build_features(df)
    assert len(X) == len(df) and list(X.index) == [0, 0, 1, 2]
    pd.testing.assert_frame_equal(X.reset_index(drop=True), expected)  # rows stay aligned by position


# ---------------------------------------------------------------- categoricals
def test_missing_categoricals_become_nan_and_others_are_strings():
    df = make_frame(
        [
            make_txn("2026-03-10 12:00:00", issuer_bank=None, merchant_state=np.nan),
            make_txn("2026-03-10 12:01:00", service=None),
            make_txn("2026-03-10 12:02:00"),
        ]
    )
    X = build_features(df)
    assert pd.isna(X.loc[0, "issuer_bank"]) and pd.isna(X.loc[0, "merchant_state"])
    assert pd.isna(X.loc[1, "service_type"])
    assert X.loc[2, CATEGORICAL].tolist() == ["upi", "low", "5411", "HDFC", "Karnataka"]
    assert X[NUMERIC].drop(columns=["dev_logamt_z", "mer_logamt_z", "dev_amt_ratio"]).notna().all().all()
    # a missing payment method is its own "method": the third row's upi is new again
    assert X["dev_new_service"].tolist() == [1, 1, 0]


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_mcc_code_category_is_the_same_for_integer_dtypes(dtype):
    df = make_frame([make_txn("2026-03-10 12:00:00")]).astype({"mcc_code": dtype})
    assert build_features(df).loc[0, "mcc_code"] == "5411"


def test_mcc_code_category_is_the_same_for_float_dtype():
    df = make_frame([make_txn("2026-03-10 12:00:00")]).astype({"mcc_code": "float64"})
    assert build_features(df).loc[0, "mcc_code"] == "5411"


def test_request_time_given_as_strings_gives_the_same_features(txns, txn_features):
    head = txns.head(400).copy()
    head["request_time"] = head["request_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    pd.testing.assert_frame_equal(build_features(head), build_features(txns.head(400)))


# ------------------------------------------------------------------ edge cases
def test_single_row_frame():
    X = build_features(make_frame([make_txn("2026-03-10 12:00:00", amount=750)]))
    row = X.iloc[0]
    counts = [
        "dev_cnt_10m",
        "dev_cnt_1h",
        "dev_cnt_24h",
        "dev_cnt_7d",
        "dev_cnt_30d",
        "dev_amt_sum_1h",
        "dev_amt_sum_24h",
        "dev_fail_cnt_1h",
        "dev_fail_cnt_24h",
        "mer_cnt_1h",
        "mer_cnt_30d",
        "dev_new_merchants_24h",
        "dev_age_days",
    ]
    assert (row[counts] == 0).all()
    assert row["dev_new_merchant"] == 1 and row["dev_new_service"] == 1
    assert pd.isna(row["dev_logamt_z"]) and pd.isna(row["dev_amt_ratio"]) and pd.isna(row["mer_logamt_z"])
    assert row["mer_fraud_rate_lag"] == pytest.approx(PRIOR_RATE)
    assert row["dev_secs_since_last"] == LOOK  # "no previous payment" is the 30-day cap


def test_device_first_transaction_ignores_other_devices_history():
    """A new device at a busy merchant: device history is empty, merchant history is not."""
    rows = [make_txn(f"2026-03-10 11:{m:02d}:00", device=f"OLD{m % 3}", amount=500, status="FAILED") for m in range(30)]
    rows.append(make_txn("2026-03-10 11:30:00", device="NEW", amount=500))
    X = build_features(make_frame(rows))
    new = X.iloc[-1]
    assert (
        new[["dev_cnt_10m", "dev_cnt_1h", "dev_cnt_30d", "dev_amt_sum_24h", "dev_fail_cnt_24h", "dev_age_days"]]
        .eq(0)
        .all()
    )
    assert new["dev_new_merchant"] == 1 and new["dev_new_service"] == 1
    assert pd.isna(new["dev_logamt_z"]) and pd.isna(new["dev_amt_ratio"])
    assert new["mer_cnt_1h"] == 30 and new["mer_cnt_30d"] == 30
    assert new["mer_logamt_z"] == pytest.approx(0.0)


def test_identical_timestamps_on_one_device_are_ordered_by_request_id():
    """Rows in the same second: each one sees the ones with a smaller request_id, never itself
    or later ones. Input order is reversed to show that it is the id that breaks the tie."""
    df = make_frame(
        [
            make_txn("2026-03-10 12:00:00", amount=300, request_id="T3", merchant="M3"),
            make_txn("2026-03-10 12:00:00", amount=200, request_id="T2", merchant="M2", status="FAILED"),
            make_txn("2026-03-10 12:00:00", amount=100, request_id="T1", merchant="M1"),
        ]
    )
    X = build_features(df)
    X.index = df["request_id"]
    X = X.loc[["T1", "T2", "T3"]]
    for col in ("dev_cnt_10m", "dev_cnt_1h", "dev_cnt_24h", "dev_cnt_7d", "dev_cnt_30d"):
        assert X[col].tolist() == [0, 1, 2], col
    assert X["dev_amt_sum_1h"].tolist() == [0, 100, 300]
    assert X["dev_secs_since_last"].tolist() == [LOOK, 0, 0]
    assert X["dev_fail_cnt_1h"].tolist() == [0, 0, 1]
    assert X["dev_new_merchants_24h"].tolist() == [0, 1, 2]
    assert X["dev_amt_ratio"].tolist()[1:] == [2.0, 2.0]
    assert X["dev_age_days"].tolist() == [0, 0, 0]


def test_window_boundaries_are_closed_on_the_old_side():
    """[t - w, t): a payment exactly w seconds ago is inside, one second older is outside."""
    t = pd.Timestamp("2026-03-10 12:00:00")
    df = make_frame(
        [
            make_txn(t - pd.Timedelta(seconds=3601), amount=1),
            make_txn(t - pd.Timedelta(seconds=3600), amount=10),
            make_txn(t - pd.Timedelta(seconds=601), amount=100),
            make_txn(t - pd.Timedelta(seconds=600), amount=1000),
            make_txn(t, amount=5),
        ]
    )
    last = build_features(df).iloc[-1]
    assert last["dev_cnt_10m"] == 1
    assert last["dev_cnt_1h"] == 3
    assert last["dev_amt_sum_1h"] == 1110
    assert last["dev_cnt_24h"] == 4


def test_new_merchant_flag_resets_after_the_lookback():
    df = make_frame(
        [
            make_txn("2026-01-01 12:00:00", merchant="M1"),
            make_txn("2026-01-31 12:00:00", merchant="M1"),  # exactly 30 days later: still known
            make_txn("2026-03-02 12:00:01", merchant="M1"),  # 30 days + 1 s later: new again
            make_txn("2026-03-02 12:00:02", merchant="M2"),
        ]
    )
    X = build_features(df)
    assert X["dev_new_merchant"].tolist() == [1, 0, 1, 1]
    assert X["dev_new_merchants_24h"].tolist() == [0, 0, 0, 1]
    assert X["dev_age_days"].tolist() == [0, 30, 30, 30]  # capped at the lookback


RETENTION = pd.Timedelta(days=30 + config.LABEL_DELAY_DAYS)


def test_features_are_reproducible_from_37_days_of_history():
    """Documented contract: every feature can be rebuilt from 37 days of transactions
    (30-day lookback + label delay) plus one first-seen timestamp per device, which only
    dev_age_days needs."""
    df = make_frame(
        [
            make_txn("2026-01-01 12:00:00"),  # the device's first row, 100 days back
            make_txn("2026-02-01 12:00:00"),  # old history that may be forgotten
            make_txn("2026-04-01 12:00:00"),
            make_txn("2026-04-11 12:00:00"),  # target
        ]
    )
    full = build_features(df).loc[3]
    assert full["dev_age_days"] == 30
    within = df["request_time"] >= df.loc[3, "request_time"] - RETENTION

    retained = build_features(df[within | (df.index == 0)]).loc[3]  # 37 days + first-seen row
    pd.testing.assert_series_equal(retained, full)

    without_first_seen = build_features(df[within]).loc[3]  # 37 days only
    assert without_first_seen["dev_age_days"] == 10  # the one documented exception
    pd.testing.assert_series_equal(without_first_seen.drop("dev_age_days"), full.drop("dev_age_days"))


def test_generated_features_are_reproducible_from_37_days_plus_first_seen_rows(txns, txn_features):
    start = pd.Timestamp("2026-06-01")
    within = txns["request_time"] >= start - RETENTION
    target = txns.index[txns["request_time"] >= start]
    first_rows = ~txns["device_id"].duplicated()  # txns is in event order

    with_first = build_features(txns[within | first_rows])
    pd.testing.assert_frame_equal(
        with_first.loc[target], txn_features.loc[target], check_exact=False, rtol=1e-7, atol=1e-6
    )

    only_recent = build_features(txns[within])
    others = [f for f in FEATURES if f != "dev_age_days"]
    pd.testing.assert_frame_equal(
        only_recent.loc[target, others], txn_features.loc[target, others], check_exact=False, rtol=1e-7, atol=1e-6
    )
    # and the first-seen row is really needed: without it the device age is understated
    too_young = only_recent.loc[target, "dev_age_days"] < txn_features.loc[target, "dev_age_days"] - 1e-9
    assert too_young.sum() > 100
    assert (only_recent.loc[target, "dev_age_days"] <= txn_features.loc[target, "dev_age_days"] + 1e-9).all()


def test_nan_amount_on_one_device_does_not_corrupt_other_devices():
    df = make_frame(
        [
            make_txn("2026-03-10 12:00:00", device="A", merchant="MA", amount=100),
            make_txn("2026-03-10 12:01:00", device="B", merchant="MB", amount=200),
            make_txn("2026-03-10 12:02:00", device="B", merchant="MB", amount=300),
        ]
    )
    clean = build_features(df)
    broken = df.copy()
    broken.loc[0, "amount"] = np.nan
    X = build_features(broken)
    assert X.loc[2, "dev_amt_sum_1h"] == clean.loc[2, "dev_amt_sum_1h"] == 200
    assert X.loc[2, "dev_amt_ratio"] == clean.loc[2, "dev_amt_ratio"]
    pd.testing.assert_frame_equal(X.loc[[1, 2]], clean.loc[[1, 2]])  # device B is untouched entirely
