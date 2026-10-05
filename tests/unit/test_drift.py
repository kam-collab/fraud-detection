"""Drift statistics: PSI (numeric / categorical), KS and the status thresholds."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from app.controllers.monitoring.drift import feature_drift, ks_statistic, psi_categorical, psi_numeric, status
from app.utils import config


@pytest.fixture
def rng():
    return np.random.default_rng(123)


# ------------------------------------------------------------------ PSI numeric
def test_psi_is_zero_for_identical_inputs(rng):
    x = rng.normal(size=5000)
    assert psi_numeric(x, x) == pytest.approx(0.0, abs=1e-12)
    cats = rng.choice(["a", "b", "c"], 5000)
    assert psi_categorical(cats, cats) == pytest.approx(0.0, abs=1e-12)


def test_psi_is_near_zero_for_two_samples_of_one_distribution(rng):
    assert psi_numeric(rng.lognormal(size=20_000), rng.lognormal(size=20_000)) < 0.01
    p = [0.6, 0.3, 0.1]
    assert psi_categorical(rng.choice(["a", "b", "c"], 20_000, p=p), rng.choice(["a", "b", "c"], 20_000, p=p)) < 0.01


def test_psi_grows_with_the_size_of_the_shift(rng):
    ref = rng.normal(size=20_000)
    values = [psi_numeric(ref, rng.normal(loc=shift, size=20_000)) for shift in (0.0, 0.25, 0.5, 1.0, 2.0)]
    assert values == sorted(values)
    assert values[0] < config.PSI_WARN < values[2]  # half a standard deviation is at least a warning
    assert values[3] > config.PSI_ALERT  # a full standard deviation is an alert


def test_psi_matches_the_textbook_formula_on_a_two_bin_example():
    ref = ["a"] * 50 + ["b"] * 50
    cur = ["a"] * 80 + ["b"] * 20
    expected = (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert psi_categorical(ref, cur) == pytest.approx(expected)
    assert psi_categorical(cur, ref) == pytest.approx(expected)  # exactly symmetric on fixed bins


def test_psi_is_non_negative_and_roughly_symmetric(rng):
    a, b = rng.normal(0, 1, 20_000), rng.normal(0.6, 1.3, 20_000)
    forward, backward = psi_numeric(a, b), psi_numeric(b, a)
    assert forward > 0 and backward > 0
    assert forward == pytest.approx(backward, rel=0.5)  # bins come from the reference, so only roughly
    for _ in range(20):  # never negative, whatever the samples
        assert psi_numeric(rng.normal(size=300), rng.normal(size=300)) >= 0


def test_psi_does_not_depend_on_sample_sizes(rng):
    ref, cur = rng.normal(size=40_000), rng.normal(0.5, 1, 40_000)
    assert psi_numeric(ref, cur[:10_000]) == pytest.approx(psi_numeric(ref, cur), abs=0.02)


def test_psi_handles_nan_as_its_own_bin(rng):
    ref = rng.normal(size=10_000)
    same = psi_numeric(np.r_[ref, [np.nan] * 100], np.r_[ref, [np.nan] * 100])
    assert same == pytest.approx(0.0, abs=1e-12)
    more_missing = psi_numeric(ref, np.r_[ref[:8000], [np.nan] * 2000])  # 0% -> 20% missing
    assert math.isfinite(more_missing) and more_missing > config.PSI_ALERT
    fewer = psi_numeric(ref, np.r_[ref[:9900], [np.nan] * 100])  # 0% -> 1% missing
    assert 0 < fewer < more_missing


def test_psi_is_finite_when_the_current_data_leaves_the_reference_range(rng):
    value = psi_numeric(rng.uniform(0, 1, 5000), rng.uniform(10, 11, 5000))
    assert math.isfinite(value) and value > 1


def test_psi_on_a_constant_feature(rng):
    ones = np.ones(1000)
    assert psi_numeric(ones, ones) == pytest.approx(0.0, abs=1e-12)
    moved = psi_numeric(ones, np.r_[np.ones(500), np.zeros(500)])
    assert math.isfinite(moved) and moved > config.PSI_ALERT


def test_psi_on_a_binary_feature_detects_a_rate_change(rng):
    ref = (rng.random(10_000) < 0.10).astype(float)
    assert psi_numeric(ref, (rng.random(10_000) < 0.10).astype(float)) < 0.01
    assert psi_numeric(ref, (rng.random(10_000) < 0.40).astype(float)) > config.PSI_ALERT


def test_psi_with_all_missing_current_values_is_finite(rng):
    value = psi_numeric(rng.normal(size=1000), np.full(1000, np.nan))
    assert math.isfinite(value) and value > config.PSI_ALERT


# -------------------------------------------------------------- PSI categorical
def test_psi_categorical_handles_unseen_and_vanished_categories():
    ref = ["upi"] * 700 + ["wallet"] * 300
    unseen = psi_categorical(ref, ["upi"] * 600 + ["wallet"] * 300 + ["crypto"] * 100)
    vanished = psi_categorical(ref, ["upi"] * 1000)
    assert math.isfinite(unseen) and unseen > config.PSI_WARN
    assert math.isfinite(vanished) and vanished > config.PSI_ALERT


def test_psi_categorical_treats_missing_as_a_category():
    ref = pd.Series(["HDFC"] * 990 + [None] * 10)
    assert psi_categorical(ref, ref) == pytest.approx(0.0, abs=1e-12)
    incident = pd.Series(["HDFC"] * 900 + [np.nan] * 100)  # 1% -> 10% missing
    assert psi_categorical(ref, incident) > config.PSI_WARN


def test_psi_categorical_compares_integer_codes_by_value():
    ref = pd.array([5411, 5812, 5411, 5999] * 100, dtype="Int64")
    assert psi_categorical(ref, np.array([5411, 5812, 5411, 5999] * 50)) == pytest.approx(0.0, abs=1e-12)


# -------------------------------------------------------------------------- KS
def test_ks_is_zero_for_identical_inputs_and_one_for_disjoint_ones(rng):
    x = rng.normal(size=2000)
    assert ks_statistic(x, x) == 0.0
    assert ks_statistic(x, x + 100) == 1.0


def test_ks_stays_in_unit_interval_and_grows_with_shift(rng):
    ref = rng.normal(size=10_000)
    values = [ks_statistic(ref, rng.normal(loc=s, size=10_000)) for s in (0.0, 0.2, 0.5, 1.0, 3.0)]
    assert all(0.0 <= v <= 1.0 for v in values)
    assert values == sorted(values)
    assert values[0] < config.KS_WARN < config.KS_ALERT < values[2]


def test_ks_matches_a_hand_computed_value():
    # ECDFs differ most at x in [2, 3): F_ref = 2/4, F_cur = 0
    assert ks_statistic([1, 2, 3, 4], [3, 4, 5, 6]) == pytest.approx(0.5)


def test_ks_ignores_nan_and_is_nan_when_a_side_is_empty(rng):
    x = rng.normal(size=1000)
    assert ks_statistic(np.r_[x, [np.nan] * 50], x) == 0.0
    assert math.isnan(ks_statistic([np.nan, np.nan], x))
    assert math.isnan(ks_statistic(x, []))


# ---------------------------------------------------------------------- status
@pytest.mark.parametrize(
    "value, expected",
    [
        (0.0, "ok"),
        (0.0999, "ok"),
        (0.10, "warn"),
        (0.2499, "warn"),
        (0.25, "alert"),
        (3.0, "alert"),
        (float("nan"), "unknown"),
        (None, "unknown"),
    ],
)
def test_status_thresholds(value, expected):
    assert status(value, config.PSI_WARN, config.PSI_ALERT) == expected


def test_feature_drift_reports_the_worst_of_psi_and_ks(rng):
    n = 20_000
    ref = pd.DataFrame(
        {
            "stable": rng.normal(size=n),
            "shifted": rng.normal(size=n),
            "method": rng.choice(["upi", "wallet"], n, p=[0.5, 0.5]),
        }
    )
    cur = pd.DataFrame(
        {
            "stable": rng.normal(size=n),
            "shifted": rng.normal(loc=1.0, size=n),
            "method": rng.choice(["upi", "wallet"], n, p=[0.9, 0.1]),
        }
    )
    cur.loc[:999, "stable"] = np.nan
    rows = {r["feature"]: r for r in feature_drift(ref, cur, ["stable", "shifted"], ["method"])}
    assert list(rows) == ["stable", "shifted", "method"]
    assert rows["shifted"]["status"] == "alert" and rows["shifted"]["kind"] == "numeric"
    assert rows["method"]["status"] == "alert" and rows["method"]["ks"] is None
    assert rows["stable"]["missing_ref"] == 0.0 and rows["stable"]["missing_cur"] == pytest.approx(0.05)
    assert rows["stable"]["ks"] < config.KS_WARN  # KS ignores the missing values...
    assert rows["stable"]["psi"] > config.PSI_WARN  # ...the PSI missing-bin catches them
    assert rows["stable"]["status"] in ("warn", "alert")
