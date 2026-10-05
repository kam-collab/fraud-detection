"""Synthetic payment-gateway transactions in the assignment schema.

The provided sample has 12 rows, which is enough to learn the schema but not to
train a model. This module simulates January-September 2026 with:

* genuine behaviour: devices with their own spending level, favourite merchants,
  preferred payment method and active hours, including "hard negatives"
  (legitimate big-ticket purchases, quick repeat purchases, retries after a failure,
  night owls, genuine use of high-risk merchants);
* four fraud patterns:
    A  account-takeover bursts on an existing device (several large payments in minutes),
    B  mule devices: brand-new devices used only for fraud at high-risk merchants,
    C  stealth fraud: single, moderately unusual payments that look almost normal,
    D  a NEW pattern that only appears from mid-July (small UPI scam payments at
       low-risk merchants) - concept drift for the monitoring section;
* drift in the live period (July-September): amount inflation, a shift towards UPI,
  an August-September "midnight sale" at electronics/travel/jewellery merchants, and
  a September data-quality incident (issuer_bank missing far more often);
* label noise: a few frauds are never reported, a few genuine payments are disputed.

Everything is driven by one seed, so the data set is reproducible.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from app.utils import config

EPOCH = pd.Timestamp("2026-01-01")
N_DAYS = 273  # 1 Jan - 30 Sep 2026
LIVE_DAY = 181  # 1 Jul
PROMO_DAY = 212  # 1 Aug
INCIDENT_DAY = 243  # 1 Sep
NEW_PATTERN_DAY = 190  # 10 Jul

# code, title, risk tier, median ticket (INR), popularity weight
MCC = [
    (5411, "Grocery Stores", "low", 800, 16),
    (5812, "Eating Places and Restaurants", "low", 600, 14),
    (5311, "Department Stores", "low", 2500, 9),
    (5541, "Service Stations", "low", 1200, 8),
    (4900, "Utilities", "low", 1800, 8),
    (5912, "Drug Stores and Pharmacies", "low", 500, 7),
    (8299, "Educational Services", "low", 12000, 3),
    (5941, "Sporting Goods Stores", "low", 5000, 3),
    (4814, "Telecommunication Services", "medium", 400, 9),
    (5999, "Miscellaneous Retail Stores", "medium", 2000, 6),
    (5732, "Electronic Stores", "medium", 15000, 4),
    (4722, "Travel Agencies", "medium", 18000, 3),
    (5944, "Jewelry Stores", "medium", 25000, 1.5),
    (6540, "Prepaid/Stored Value Card Load", "high", 5000, 2.5),
    (4829, "Money Transfer", "high", 8000, 2),
    (7995, "Betting and Gambling", "high", 3000, 1.5),
    (7273, "Dating and Escort Services", "high", 2000, 0.8),
    (6051, "Quasi Cash and Crypto", "high", 10000, 0.7),
]
PROMO_MCC = {5732, 4722, 5944}
SCAM_MCC = {5999, 4814, 4900, 5411}
CITIES = [
    ("Maharashtra", "Mumbai"),
    ("Maharashtra", "Pune"),
    ("Delhi", "New Delhi"),
    ("Karnataka", "Bengaluru"),
    ("Tamil Nadu", "Chennai"),
    ("Odisha", "Bhubaneswar"),
    ("Rajasthan", "Jaipur"),
    ("Uttar Pradesh", "Lucknow"),
    ("Uttar Pradesh", "Noida"),
    ("Telangana", "Hyderabad"),
    ("West Bengal", "Kolkata"),
    ("Gujarat", "Ahmedabad"),
    ("Gujarat", "Surat"),
    ("Kerala", "Kochi"),
    ("Punjab", "Ludhiana"),
    ("Madhya Pradesh", "Indore"),
    ("Bihar", "Patna"),
    ("Haryana", "Gurugram"),
]
BANKS = np.array(["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "BOB", "INDUSIND", "PNB", "YES", "IDFC"])
BANK_P = np.array([18, 16, 22, 10, 7, 8, 4, 8, 3, 4], dtype=float)
SERVICES = np.array(["upi", "wallet", "imps", "netbanking"])
SERVICE_P = np.array([0.55, 0.15, 0.10, 0.20])
STATUSES = np.array(["SUCCESS", "FAILED", "DECLINED"])

HOUR_GENUINE = np.array(
    [0.6, 0.4, 0.3, 0.25, 0.3, 0.6, 1.5, 3, 5, 6, 6.5, 6.5, 6.5, 6, 5.5, 5.5, 6, 6.5, 7, 7, 6.5, 5, 3, 1.5]
)
HOUR_GENUINE = HOUR_GENUINE / HOUR_GENUINE.sum()
HOUR_OWL = HOUR_GENUINE + 2.5 / 24
HOUR_OWL = HOUR_OWL / HOUR_OWL.sum()
_night = np.zeros(24)
_night[[22, 23, 0, 1, 2, 3, 4, 5]] = 1 / 8
HOUR_FRAUD = 0.5 * _night + 0.5 * HOUR_GENUINE


def _norm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return p / p.sum()


def _ramp(day: np.ndarray) -> np.ndarray:
    """0 before July, rising linearly to 1 at the end of September."""
    return np.clip((day - LIVE_DAY) / (N_DAYS - LIVE_DAY), 0, 1)


def generate(
    n_genuine: int = 500_000, n_devices: int = 30_000, n_merchants: int = 1_500, seed: int = config.SEED
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- merchants
    mcc_idx = rng.choice(len(MCC), n_merchants, p=_norm([m[4] for m in MCC]))
    m_code = np.array([MCC[i][0] for i in mcc_idx])
    m_title = np.array([MCC[i][1] for i in mcc_idx])
    m_tier_mcc = np.array([MCC[i][2] for i in mcc_idx])
    m_median = np.array([MCC[i][3] for i in mcc_idx]) * rng.lognormal(0, 0.3, n_merchants)
    # merchant_type is the gateway's own risk tier: usually, not always, the MCC tier
    tiers = np.array(["low", "medium", "high"])
    m_type = np.where(rng.random(n_merchants) < 0.85, m_tier_mcc, rng.choice(tiers, n_merchants))
    city_idx = rng.choice(len(CITIES), n_merchants)
    m_state = np.array([CITIES[i][0] for i in city_idx])
    m_city = np.array([CITIES[i][1] for i in city_idx])
    m_pop = _norm(rng.pareto(1.5, n_merchants) + 0.2)
    # a few merchants are disproportionately abused by fraudsters
    m_fraud_mult = rng.lognormal(0, 1.0, n_merchants)
    tier_w = pd.Series(m_type).map({"low": 1.0, "medium": 3.0, "high": 14.0}).to_numpy()
    p_fraud_merchant = _norm(m_pop * m_fraud_mult * tier_w)
    p_stealth_merchant = _norm(m_pop * m_fraud_mult * np.sqrt(tier_w))
    promo_merchants = np.flatnonzero(np.isin(m_code, list(PROMO_MCC)))
    if len(promo_merchants) == 0:  # tiny merchant populations: fall back to all merchants
        promo_merchants = np.arange(n_merchants)
    p_promo = _norm(m_pop[promo_merchants])
    scam_pool = np.flatnonzero(np.isin(m_code, list(SCAM_MCC)) & (m_pop < np.median(m_pop)))
    scam_merchants = rng.choice(scam_pool, min(40, len(scam_pool)), replace=False)
    low_merchants = np.flatnonzero(m_type == "low")
    p_low = _norm(m_pop[low_merchants])

    # ------------------------------------------------------------------ devices
    d_activity = _norm(rng.lognormal(0, 1.0, n_devices))
    d_offset = rng.normal(0, 0.6, n_devices)  # personal spending level (log)
    d_service = rng.choice(4, n_devices, p=SERVICE_P)
    d_bank = rng.choice(len(BANKS), n_devices, p=_norm(BANK_P))
    d_owl = rng.random(n_devices) < 0.08
    d_start = np.where(rng.random(n_devices) < 0.8, 0, rng.integers(0, N_DAYS - 5, n_devices))
    d_favs = rng.choice(n_merchants, (n_devices, 5), p=m_pop)

    # --------------------------------------------------------- genuine payments
    dev = rng.choice(n_devices, n_genuine, p=d_activity)
    day = d_start[dev] + np.floor(rng.random(n_genuine) * (N_DAYS - d_start[dev])).astype(int)
    hour = np.where(d_owl[dev], rng.choice(24, n_genuine, p=HOUR_OWL), rng.choice(24, n_genuine, p=HOUR_GENUINE))
    mer = np.where(
        rng.random(n_genuine) < 0.75,
        d_favs[dev, rng.integers(0, 5, n_genuine)],
        rng.choice(n_merchants, n_genuine, p=m_pop),
    )
    # Aug-Sep midnight sale: more high-ticket purchases, many right after midnight
    promo = (day >= PROMO_DAY) & (rng.random(n_genuine) < 0.10)
    mer = np.where(promo, rng.choice(promo_merchants, n_genuine, p=p_promo), mer)
    hour = np.where(promo & (rng.random(n_genuine) < 0.5), rng.integers(0, 2, n_genuine), hour)
    sec = day * 86400 + hour * 3600 + rng.integers(0, 3600, n_genuine)

    ramp = _ramp(day)
    svc = np.where(rng.random(n_genuine) < 0.8, d_service[dev], rng.choice(4, n_genuine, p=SERVICE_P))
    svc = np.where(rng.random(n_genuine) < 0.30 * ramp, 0, svc)  # shift towards UPI
    amount = np.exp(np.log(m_median[mer]) + 0.6 * d_offset[dev] + rng.normal(0, 0.55, n_genuine))
    big = rng.random(n_genuine) < 0.02  # genuine big tickets
    amount = np.where(big, amount * rng.uniform(4, 15, n_genuine), amount)
    amount = amount * (1 + 0.20 * ramp)  # price inflation
    status = rng.choice(3, n_genuine, p=[0.90, 0.07, 0.03])

    g = {"dev": dev, "sec": sec, "mer": mer, "svc": svc, "amount": amount, "status": status}

    # quick repeat purchases (same merchant, similar amount, a minute or two later)
    rep = np.flatnonzero(rng.random(n_genuine) < 0.06)
    rep = np.concatenate([rep, rep[rng.random(len(rep)) < 0.3]])
    g_rep = {
        "dev": dev[rep],
        "sec": sec[rep] + rng.integers(30, 240, len(rep)),
        "mer": mer[rep],
        "svc": svc[rep],
        "amount": amount[rep] * rng.uniform(0.9, 1.1, len(rep)),
        "status": rng.choice(3, len(rep), p=[0.90, 0.07, 0.03]),
    }
    # retries after a failed payment
    ret = np.flatnonzero((status > 0) & (rng.random(n_genuine) < 0.4))
    g_ret = {
        "dev": dev[ret],
        "sec": sec[ret] + rng.integers(20, 150, len(ret)),
        "mer": mer[ret],
        "svc": svc[ret],
        "amount": amount[ret],
        "status": rng.choice(3, len(ret), p=[0.70, 0.20, 0.10]),
    }

    genuine = pd.concat([pd.DataFrame(x) for x in (g, g_rep, g_ret)], ignore_index=True)
    genuine["fraud"] = 0
    genuine["pattern"] = "genuine"

    # each device's normal ticket size, used to scale fraud amounts
    typical = genuine.groupby("dev")["amount"].median().reindex(range(n_devices)).fillna(1500).to_numpy()
    n_per_dev = np.bincount(dev, minlength=n_devices)
    eligible = np.flatnonzero(n_per_dev[dev] >= 5)  # anchors: payments of devices with history

    def fraud_service(n: int, p) -> np.ndarray:
        return rng.choice(4, n, p=p)

    per_day = len(genuine) / N_DAYS
    base_fraud_txn = 0.011 * per_day * N_DAYS  # target ~1.1% before the new pattern

    # ---------------------------------------- A: account-takeover bursts
    n_a = int(0.45 * base_fraud_txn / 2.6)
    anchor = rng.choice(eligible, n_a)
    a_dev = dev[anchor]
    a_day = np.clip(day[anchor] + rng.integers(0, 6, n_a), 0, N_DAYS - 1)
    a_start = a_day * 86400 + rng.choice(24, n_a, p=HOUR_FRAUD) * 3600 + rng.integers(0, 3600, n_a)
    a_k = rng.choice([1, 2, 3, 4, 5], n_a, p=[0.25, 0.25, 0.25, 0.15, 0.10])
    burst = np.repeat(np.arange(n_a), a_k)
    pos = np.concatenate([np.arange(k) for k in a_k])
    gaps = rng.integers(20, 150, len(burst))
    a_sec = a_start[burst] + pd.Series(gaps).groupby(burst).cumsum().to_numpy() - gaps
    burst_mer = rng.choice(n_merchants, n_a, p=p_fraud_merchant)
    a_mer = np.where(
        rng.random(len(burst)) < 0.7, burst_mer[burst], rng.choice(n_merchants, len(burst), p=p_fraud_merchant)
    )
    a_amt = np.clip(typical[a_dev[burst]] * rng.lognormal(2.3, 0.8, len(burst)), 50, 200_000)
    test_txn = (pos == 0) & (a_k[burst] >= 3) & (rng.random(len(burst)) < 0.4)
    a_amt = np.where(test_txn, rng.uniform(1, 100, len(burst)), a_amt)  # small "card test" first
    fa = pd.DataFrame(
        {
            "dev": a_dev[burst],
            "sec": a_sec,
            "mer": a_mer,
            "svc": fraud_service(len(burst), [0.25, 0.30, 0.10, 0.35]),
            "amount": a_amt,
            "status": rng.choice(3, len(burst), p=[0.60, 0.25, 0.15]),
            "pattern": "A_takeover",
        }
    )

    # ---------------------------------------- B: mule devices (new, fraud only)
    n_b = int(0.30 * base_fraud_txn / 3.5)
    b_k = rng.integers(1, 7, n_b)
    b_id = np.repeat(np.arange(n_b), b_k)
    b_start = (
        rng.integers(0, N_DAYS, n_b) * 86400 + rng.choice(24, n_b, p=HOUR_FRAUD) * 3600 + rng.integers(0, 3600, n_b)
    )
    b_gap = rng.exponential(2400, len(b_id)).astype(int) + 30
    b_sec = b_start[b_id] + pd.Series(b_gap).groupby(b_id).cumsum().to_numpy() - b_gap
    fb = pd.DataFrame(
        {
            "dev": n_devices + b_id,
            "sec": b_sec,
            "mer": rng.choice(n_merchants, len(b_id), p=p_fraud_merchant),
            "svc": fraud_service(len(b_id), [0.20, 0.50, 0.10, 0.20]),
            "amount": np.clip(rng.lognormal(np.log(20_000), 0.9, len(b_id)), 100, 200_000),
            "status": rng.choice(3, len(b_id), p=[0.55, 0.30, 0.15]),
            "pattern": "B_mule",
        }
    )
    mule_bank = rng.choice(len(BANKS), n_b, p=_norm(BANK_P))

    # ---------------------------------------- C: stealth fraud (looks almost normal)
    n_c = int(0.25 * base_fraud_txn)
    anchor = rng.choice(eligible, n_c)
    c_day = np.clip(day[anchor] + rng.integers(0, 6, n_c), 0, N_DAYS - 1)
    fc = pd.DataFrame(
        {
            "dev": dev[anchor],
            "sec": c_day * 86400 + rng.choice(24, n_c, p=HOUR_GENUINE) * 3600 + rng.integers(0, 3600, n_c),
            "mer": rng.choice(n_merchants, n_c, p=p_stealth_merchant),
            "svc": fraud_service(n_c, SERVICE_P),
            "amount": np.clip(typical[dev[anchor]] * rng.lognormal(0.9, 0.6, n_c), 50, 200_000),
            "status": rng.choice(3, n_c, p=[0.82, 0.12, 0.06]),
            "pattern": "C_stealth",
        }
    )

    # ---------------------------------------- D: new UPI scam pattern, from mid-July
    n_d = int(0.011 * per_day * (N_DAYS - NEW_PATTERN_DAY) * 0.9)
    anchor = rng.choice(eligible, n_d)
    d_day = NEW_PATTERN_DAY + np.floor(np.sqrt(rng.random(n_d)) * (N_DAYS - NEW_PATTERN_DAY)).astype(int)
    d_day = np.clip(d_day, 0, N_DAYS - 1)
    price_points = np.array([499, 999, 1499, 1999, 2999, 4999, 7999])
    fd = pd.DataFrame(
        {
            "dev": dev[anchor],
            "sec": d_day * 86400 + rng.choice(24, n_d, p=HOUR_GENUINE) * 3600 + rng.integers(0, 3600, n_d),
            "mer": np.where(
                rng.random(n_d) < 0.5, rng.choice(scam_merchants, n_d), rng.choice(low_merchants, n_d, p=p_low)
            ),
            "svc": 0,
            "amount": rng.choice(price_points, n_d) * rng.uniform(0.97, 1.03, n_d),
            "status": rng.choice(3, n_d, p=[0.90, 0.07, 0.03]),
            "pattern": "D_upi_scam",
        }
    )

    fraud = pd.concat([fa, fb, fc, fd], ignore_index=True)
    fraud["fraud"] = 1
    df = pd.concat([genuine, fraud], ignore_index=True)
    df = df[(df["sec"] >= 0) & (df["sec"] < N_DAYS * 86400)].copy()

    # ------------------------------------------------------ assemble raw schema
    d, m = df["dev"].to_numpy(), df["mer"].to_numpy()
    is_mule = d >= n_devices
    bank = np.where(
        is_mule, BANKS[mule_bank[np.where(is_mule, d - n_devices, 0)]], BANKS[d_bank[np.where(is_mule, 0, d)]]
    )
    service = SERVICES[df["svc"].to_numpy()]
    bank = np.where(service == "wallet", "NOT_APPLICABLE", bank)

    out = pd.DataFrame(
        {
            "request_time": EPOCH + pd.to_timedelta(df["sec"].to_numpy(), unit="s"),
            "service_type": service,
            "device_id": ["DEV%07d" % (1000 + x) for x in d],
            "merchant_id": ["M%06d" % (100000 + x) for x in m],
            "merchant_state": m_state[m],
            "merchant_city": m_city[m],
            "merchant_type": m_type[m],
            "mcc_code": m_code[m],
            "mcc_title": m_title[m],
            "issuer_bank": bank,
            "currency_code": "INR",
            "amount": np.round(df["amount"].to_numpy(), 2),
            "request_status": STATUSES[df["status"].to_numpy()],
            "fraud_label": df["fraud"].to_numpy(),
            "fraud_pattern": df["pattern"].to_numpy(),  # simulator ground truth, never a model input
        }
    )

    # label noise: 2% of frauds are never reported, 0.03% of genuine payments are disputed
    u = rng.random(len(out))
    y = out["fraud_label"].to_numpy()
    out["fraud_label"] = np.where((y == 1) & (u < 0.02), 0, np.where((y == 0) & (u < 0.0003), 1, y))

    # missing values, including the September issuer_bank incident
    day_all = df["sec"].to_numpy() // 86400
    miss_bank = rng.random(len(out)) < np.where(day_all >= INCIDENT_DAY, 0.10, 0.01)
    out.loc[miss_bank, "issuer_bank"] = np.nan
    for col, rate in (("merchant_city", 0.005), ("merchant_state", 0.005), ("mcc_title", 0.005)):
        out.loc[rng.random(len(out)) < rate, col] = np.nan

    out = out.sort_values("request_time", kind="stable").reset_index(drop=True)
    out.insert(0, "request_id", ["TXN%07d" % (1_000_000 + i) for i in range(len(out))])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--rows", type=int, default=500_000, help="number of base genuine payments")
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()
    df = generate(n_genuine=args.rows, seed=args.seed)
    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.SYNTHETIC_CSV, index=False)
    month = df["request_time"].dt.to_period("M")
    summary = df.groupby(month)["fraud_label"].agg(rows="size", frauds="sum", fraud_rate="mean")
    print(f"Wrote {len(df):,} rows to {config.SYNTHETIC_CSV}")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
