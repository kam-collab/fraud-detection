"""The synthetic data generator: schema, reproducibility and the properties the rest relies on."""

from __future__ import annotations

import pandas as pd
import pytest

from app.controllers.data_generation.generator import generate
from app.utils import config

SMALL = dict(n_genuine=2500, n_devices=200, n_merchants=300)


def test_same_seed_gives_identical_data():
    pd.testing.assert_frame_equal(generate(**SMALL, seed=5), generate(**SMALL, seed=5))


def test_different_seeds_give_different_data():
    a, b = generate(**SMALL, seed=5), generate(**SMALL, seed=6)
    assert len(a) != len(b) or not a["amount"].equals(b["amount"])


def test_default_seed_is_the_configured_one():
    pd.testing.assert_frame_equal(generate(**SMALL), generate(**SMALL, seed=config.SEED))


def test_columns_follow_the_assignment_schema(txns):
    assert list(txns.columns) == config.RAW_COLUMNS + ["fraud_pattern"]
    assert config.RAW_COLUMNS[0] == "request_id" and config.RAW_COLUMNS[-1] == config.TARGET
    assert pd.api.types.is_datetime64_any_dtype(txns["request_time"])
    assert pd.api.types.is_integer_dtype(txns["mcc_code"])
    assert pd.api.types.is_float_dtype(txns["amount"])


def test_time_span_is_january_to_september_2026(txns):
    t = txns["request_time"]
    assert t.min() >= pd.Timestamp("2026-01-01") and t.max() < pd.Timestamp("2026-10-01")
    assert sorted(t.dt.month.unique()) == list(range(1, 10))
    assert (t.dt.year == 2026).all()
    monthly = txns.groupby(t.dt.month).size()
    assert monthly.min() > 0.5 * monthly.mean()  # no month is nearly empty


def test_rows_are_in_time_order_with_unique_increasing_ids(txns):
    assert txns["request_time"].is_monotonic_increasing
    assert txns["request_id"].is_unique
    assert txns["request_id"].is_monotonic_increasing  # so (time, id) order is the row order
    assert txns["request_id"].str.fullmatch(r"TXN\d{7}").all()


def test_fraud_rate_is_in_a_plausible_band(txns):
    rate = txns["fraud_label"].mean()
    assert 0.005 < rate < 0.03
    assert set(txns["fraud_label"].unique()) == {0, 1}
    dev_period = txns[txns["request_time"] < pd.Timestamp(config.LIVE_START)]
    assert 0.004 < dev_period["fraud_label"].mean() < 0.025


def test_labels_follow_the_simulated_pattern_up_to_label_noise(txns):
    genuine = txns["fraud_pattern"] == "genuine"
    assert txns.loc[genuine, "fraud_label"].mean() < 0.002  # a few disputed genuine payments
    assert txns.loc[~genuine, "fraud_label"].mean() > 0.93  # a few frauds never reported
    assert set(txns["fraud_pattern"]) == {"genuine", "A_takeover", "B_mule", "C_stealth", "D_upi_scam"}


def test_new_fraud_pattern_only_appears_in_the_live_period(txns):
    scam = txns[txns["fraud_pattern"] == "D_upi_scam"]
    assert len(scam) > 0
    assert scam["request_time"].min() >= pd.Timestamp("2026-07-10")
    assert (scam["service_type"] == "upi").all()


def test_mule_devices_are_used_only_for_fraud(txns):
    mule_devices = set(txns.loc[txns["fraud_pattern"] == "B_mule", "device_id"])
    assert mule_devices
    assert (txns.loc[txns["device_id"].isin(mule_devices), "fraud_pattern"] == "B_mule").all()


def test_wallet_rows_have_no_issuer_bank(txns):
    wallet = txns["service_type"] == "wallet"
    assert wallet.sum() > 100
    assert txns.loc[wallet, "issuer_bank"].dropna().eq("NOT_APPLICABLE").all()
    assert (txns.loc[wallet, "issuer_bank"] == "NOT_APPLICABLE").mean() > 0.8
    assert not (txns.loc[~wallet, "issuer_bank"] == "NOT_APPLICABLE").any()


def test_values_are_inside_the_documented_domains(txns):
    assert set(txns["service_type"]) == {"upi", "wallet", "imps", "netbanking"}
    assert set(txns["merchant_type"]) == {"low", "medium", "high"}
    assert set(txns["request_status"]) == {"SUCCESS", "FAILED", "DECLINED"}
    assert set(txns["currency_code"]) == {"INR"}
    assert (txns["amount"] > 0).all() and (txns["amount"].round(2) == txns["amount"]).all()
    assert txns["mcc_code"].between(1, 9999).all()
    for col in ("request_id", "request_time", "device_id", "merchant_id", "amount", "mcc_code", "merchant_type"):
        assert txns[col].notna().all(), col


def test_a_merchant_has_one_category_and_one_risk_tier(txns):
    per_merchant = txns.groupby("merchant_id")[["mcc_code", "merchant_type"]].nunique()
    assert (per_merchant == 1).all().all()
    titles = txns.dropna(subset=["mcc_title"]).groupby("mcc_code")["mcc_title"].nunique()
    assert (titles == 1).all()


def test_september_issuer_bank_incident(txns):
    """The data-quality incident the monitoring section is supposed to catch."""
    missing = txns["issuer_bank"].isna()
    september = txns["request_time"] >= pd.Timestamp("2026-09-01")
    assert missing[~september].mean() < 0.03
    assert missing[september].mean() > 0.06


@pytest.mark.parametrize("column", ["merchant_city", "merchant_state", "mcc_title"])
def test_nullable_columns_are_occasionally_missing(txns, column):
    assert 0 < txns[column].isna().mean() < 0.03


@pytest.mark.parametrize(
    "n_merchants, seed",
    [
        (100, 1),
        (60, 2),
        (30, 1),
    ],
)
def test_small_merchant_populations_are_supported(n_merchants, seed):
    df = generate(n_genuine=1500, n_devices=120, n_merchants=n_merchants, seed=seed)
    assert df["merchant_id"].nunique() <= n_merchants
    assert list(df.columns) == config.RAW_COLUMNS + ["fraud_pattern"]
    assert 0 < df["fraud_label"].mean() < 0.05
