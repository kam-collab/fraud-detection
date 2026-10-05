"""Drift statistics: PSI for numeric and categorical features, KS for numeric ones.

PSI rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift.
These cut-offs are an industry convention, not a statistical test, and PSI depends on
bin count and sample size. With hundreds of thousands of rows the KS p-value is always
"significant", so alerts use the KS D statistic (an effect size), never the p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.utils import config

EPS = 1e-4


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(expected / expected.sum(), EPS, None)
    a = np.clip(actual / actual.sum(), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_numeric(reference, current, bins: int = 10) -> float:
    """PSI on bins fixed at the reference deciles. Missing values get their own bin."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").to_numpy(dtype=float)
    cur = pd.to_numeric(pd.Series(current), errors="coerce").to_numpy(dtype=float)
    edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # nearly constant feature: fall back to exact values
        return psi_categorical(ref, cur)
    edges[0], edges[-1] = -np.inf, np.inf

    def counts(x):
        c = np.histogram(x[~np.isnan(x)], bins=edges)[0].astype(float)
        return np.r_[c, np.isnan(x).sum()]

    e, a = counts(ref), counts(cur)
    keep = (e + a) > 0
    return _psi(e[keep], a[keep])


def psi_categorical(reference, current) -> float:
    ref = pd.Series(reference).astype("object").where(pd.notna(pd.Series(reference)), "__missing__").astype(str)
    cur = pd.Series(current).astype("object").where(pd.notna(pd.Series(current)), "__missing__").astype(str)
    cats = sorted(set(ref) | set(cur))
    e = ref.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    a = cur.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    return _psi(e, a)


def ks_statistic(reference, current) -> float:
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna()
    cur = pd.to_numeric(pd.Series(current), errors="coerce").dropna()
    if ref.empty or cur.empty:
        return float("nan")
    return float(ks_2samp(ref, cur).statistic)


def status(value: float, warn: float, alert: float) -> str:
    if value is None or np.isnan(value):
        return "unknown"
    return "alert" if value >= alert else "warn" if value >= warn else "ok"


def feature_drift(
    reference: pd.DataFrame, current: pd.DataFrame, numeric: list[str], categorical: list[str]
) -> list[dict]:
    rows = []
    for col in numeric:
        psi = psi_numeric(reference[col], current[col])
        ks = ks_statistic(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "kind": "numeric",
                "psi": psi,
                "ks": ks,
                "missing_ref": float(pd.isna(reference[col]).mean()),
                "missing_cur": float(pd.isna(current[col]).mean()),
                "status": max(
                    status(psi, config.PSI_WARN, config.PSI_ALERT),
                    status(ks, config.KS_WARN, config.KS_ALERT),
                    key=["unknown", "ok", "warn", "alert"].index,
                ),
            }
        )
    for col in categorical:
        psi = psi_categorical(reference[col], current[col])
        rows.append(
            {
                "feature": col,
                "kind": "categorical",
                "psi": psi,
                "ks": None,
                "missing_ref": float(pd.isna(reference[col]).mean()),
                "missing_cur": float(pd.isna(current[col]).mean()),
                "status": status(psi, config.PSI_WARN, config.PSI_ALERT),
            }
        )
    return rows
