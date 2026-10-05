"""Input validation: every error class, every warning class, and single-request mode."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.controllers.validation.schema import NOT_NULL, REQUIRED, validate
from tests.conftest import make_frame, make_txn


@pytest.fixture
def batch() -> pd.DataFrame:
    """200 clean rows in time order with unique ids."""
    return make_frame(
        [
            make_txn(pd.Timestamp("2026-03-10") + pd.Timedelta(minutes=i), device=f"D{i % 7}", amount=100 + i)
            for i in range(200)
        ]
    )


def errors_with(df, text, **kwargs) -> list[str]:
    return [e for e in validate(df, **kwargs).errors if text in e]


def test_clean_batch_passes_without_errors_or_warnings(batch):
    result = validate(batch)
    assert result.ok
    assert result.errors == [] and result.warnings == []
    assert result.to_dict() == {"ok": True, "errors": [], "warnings": []}


def test_generated_data_is_valid(txns):
    result = validate(txns.drop(columns=["fraud_pattern"]))
    assert result.ok, result.errors


# ---------------------------------------------------------------------- errors
@pytest.mark.parametrize("column", sorted(REQUIRED))
def test_missing_column_is_an_error(batch, column):
    result = validate(batch.drop(columns=[column]))
    assert not result.ok
    assert len(result.errors) == 1 and "missing columns" in result.errors[0] and column in result.errors[0]


def test_empty_frame_is_an_error(batch):
    assert validate(batch.iloc[0:0]).errors == ["no rows"]


@pytest.mark.parametrize("column", NOT_NULL)
def test_null_in_a_required_field_is_an_error(batch, column):
    batch[column] = batch[column].astype(object)
    batch.loc[3, column] = None
    result = validate(batch)
    assert not result.ok
    assert any(e.startswith(f"{column}: 1 missing values") for e in result.errors), result.errors


def test_non_numeric_amount_is_an_error(batch):
    batch["amount"] = batch["amount"].astype(object)
    batch.loc[[5, 6], "amount"] = ["abc", "12,50"]
    assert errors_with(batch, "amount: 2 non-numeric")


def test_numeric_strings_are_accepted_as_amounts(batch):
    batch["amount"] = batch["amount"].astype(str)
    assert validate(batch).ok


@pytest.mark.parametrize("value", [0, -1, -0.01])
def test_non_positive_amount_is_an_error(batch, value):
    batch.loc[0, "amount"] = value
    assert errors_with(batch, "amount: 1 values <= 0")


def test_absurdly_large_amount_is_an_error(batch):
    batch.loc[0, "amount"] = 10_000_000.01
    assert errors_with(batch, "amount: 1 values above")
    batch.loc[0, "amount"] = 10_000_000.0
    assert validate(batch).ok


def test_unparseable_timestamp_is_an_error(batch):
    batch["request_time"] = batch["request_time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    batch.loc[10, "request_time"] = "not a time"
    batch.loc[11, "request_time"] = "2026-13-45 99:00:00"
    assert errors_with(batch, "request_time: 2 unparseable")


def test_valid_timestamps_in_mixed_iso_formats_are_accepted(batch):
    batch["request_time"] = batch["request_time"].astype(str)  # '2026-03-10 00:01:00'
    batch.loc[1, "request_time"] = "2026-03-10 00:01:00.500000"  # what str(datetime) gives with microseconds
    batch.loc[2, "request_time"] = "2026-03-10T00:02:00"
    result = validate(batch)
    assert result.ok and result.warnings == []
    batch.loc[3, "request_time"] = "not a time"  # mixed formats do not hide real garbage
    assert errors_with(batch, "request_time: 1 unparseable")


def test_features_accept_mixed_iso_timestamp_formats(batch):
    from app.controllers.features.builder import build_features

    expected = build_features(batch)
    batch["request_time"] = batch["request_time"].astype(str)
    batch.loc[2, "request_time"] = "2026-03-10T00:02:00"
    batch.loc[3, "request_time"] = "2026-03-10 00:03:00.000000"
    pd.testing.assert_frame_equal(build_features(batch), expected)


@pytest.mark.parametrize("value", [0, -5, 10_000, 123456])
def test_mcc_out_of_range_is_an_error(batch, value):
    batch.loc[0, "mcc_code"] = value
    assert errors_with(batch, "mcc_code: 1 values outside")


@pytest.mark.parametrize("value", [1, 9999])
def test_mcc_range_bounds_are_inclusive(batch, value):
    batch.loc[0, "mcc_code"] = value
    assert validate(batch).ok


def test_non_integer_mcc_is_an_error(batch):
    batch["mcc_code"] = batch["mcc_code"].astype(object)
    batch.loc[0, "mcc_code"] = 5411.5
    batch.loc[1, "mcc_code"] = "grocery"
    assert errors_with(batch, "mcc_code: 2 non-integer")


def test_duplicate_request_id_is_an_error(batch):
    batch.loc[[50, 51], "request_id"] = batch.loc[0, "request_id"]
    assert errors_with(batch, "request_id: 2 duplicates")


@pytest.mark.parametrize("value", [2, -1, 0.5, "yes"])
def test_bad_fraud_label_is_an_error(batch, value):
    batch["fraud_label"] = batch["fraud_label"].astype(object)
    batch.loc[4, "fraud_label"] = value
    assert errors_with(batch, "fraud_label: 1 values other than 0/1")


def test_fraud_label_is_optional_and_may_be_missing(batch):
    assert validate(batch.drop(columns=["fraud_label"])).ok
    batch["fraud_label"] = batch["fraud_label"].astype(float)
    batch.loc[0, "fraud_label"] = np.nan  # label not known yet
    assert validate(batch).ok


def test_several_problems_are_all_reported(batch):
    batch.loc[0, "amount"] = -5
    batch.loc[1, "mcc_code"] = 0
    batch.loc[2, "request_id"] = batch.loc[3, "request_id"]
    result = validate(batch)
    assert len(result.errors) == 3


# -------------------------------------------------------------------- warnings
@pytest.mark.parametrize(
    "column, value",
    [("service_type", "card"), ("merchant_type", "critical"), ("request_status", "PENDING"), ("currency_code", "USD")],
)
def test_unseen_category_is_a_warning_not_an_error(batch, column, value):
    batch.loc[0, column] = value
    result = validate(batch)
    assert result.ok
    assert result.warnings == [f"{column}: unseen values ['{value}']"]


@pytest.mark.parametrize(
    "column, limit", [("issuer_bank", 0.05), ("merchant_state", 0.03), ("merchant_city", 0.03), ("mcc_title", 0.03)]
)
def test_missing_rate_above_the_limit_is_a_warning(batch, column, limit):
    n_at_limit = int(limit * len(batch))
    batch.loc[: n_at_limit - 1, column] = None  # exactly at the limit: tolerated
    assert validate(batch).warnings == []
    batch.loc[n_at_limit, column] = None  # one more row: above the limit
    result = validate(batch)
    assert result.ok
    assert len(result.warnings) == 1 and result.warnings[0].startswith(f"{column}: ")
    assert "missing" in result.warnings[0]


def test_any_missing_request_status_in_a_batch_is_a_warning(batch):
    batch.loc[0, "request_status"] = None
    result = validate(batch)
    assert result.ok and len(result.warnings) == 1 and result.warnings[0].startswith("request_status")


def test_rows_out_of_time_order_are_a_warning(batch):
    result = validate(batch.iloc[::-1])
    assert result.ok and any("not in time order" in w for w in result.warnings)


# ---------------------------------------------------------- single-request mode
def test_single_request_mode_skips_rate_and_duplicate_checks(batch):
    batch.loc[[1, 2], "request_id"] = batch.loc[0, "request_id"]
    batch["issuer_bank"] = None
    batch["request_status"] = None
    batch = batch.iloc[::-1]
    as_batch = validate(batch)
    assert not as_batch.ok and len(as_batch.warnings) == 3
    single = validate(batch, batch=False)
    assert single.ok and single.warnings == []


def test_single_request_mode_still_rejects_impossible_values(batch):
    row = batch.iloc[[0]].copy()
    row["amount"] = -1.0
    assert not validate(row, batch=False).ok


def test_request_without_status_passes_in_single_request_mode():
    """The fraud decision is made before the payment outcome exists."""
    row = make_frame([make_txn("2026-03-10 12:00:00", status=None)]).drop(columns=["fraud_label"])
    result = validate(row, batch=False)
    assert result.ok and result.warnings == []
