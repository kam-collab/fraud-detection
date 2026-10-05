"""Shared fixtures and helpers for the test suite.

Unit tests never touch the 567k-row data file or the saved model: they use a small
seeded data set from the generator or hand-built frames.
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.controllers.data_generation.generator import generate
from app.controllers.features.builder import build_features


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: slower tests that use the real model / full data set or run the whole pipeline"
    )
    # third-party deprecation chatter from pandas / sklearn is not what these tests are about
    for category in ("FutureWarning", "DeprecationWarning", "UserWarning"):
        config.addinivalue_line("filterwarnings", f"ignore::{category}")


def make_txn(
    time, device="D1", merchant="M1", amount=100.0, status="SUCCESS", label=0, service="upi", request_id=None, **extra
) -> dict:
    """One raw transaction in the assignment schema, with sensible defaults."""
    row = {
        "request_id": request_id,
        "request_time": pd.Timestamp(time),
        "service_type": service,
        "device_id": device,
        "merchant_id": merchant,
        "merchant_state": "Karnataka",
        "merchant_city": "Bengaluru",
        "merchant_type": "low",
        "mcc_code": 5411,
        "mcc_title": "Grocery Stores",
        "issuer_bank": "HDFC",
        "currency_code": "INR",
        "amount": float(amount),
        "request_status": status,
        "fraud_label": label,
    }
    row.update(extra)
    return row


def make_frame(rows: list[dict]) -> pd.DataFrame:
    """Frame from `make_txn` rows; request ids default to R000, R001, ... in list order."""
    df = pd.DataFrame(rows)
    ids = [r if r is not None else f"R{i:03d}" for i, r in enumerate(df["request_id"])]
    df["request_id"] = ids
    return df


@pytest.fixture(scope="session")
def txns() -> pd.DataFrame:
    """A few thousand generated transactions (seeded, so identical on every run)."""
    return generate(n_genuine=6000, n_devices=400, n_merchants=300, seed=7)


@pytest.fixture(scope="session")
def txn_features(txns) -> pd.DataFrame:
    return build_features(txns)
