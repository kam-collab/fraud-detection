"""Analyst-facing explanations for flagged transactions.

The numbers always come from the model (FraudModel.contributions). Two ways to word them:

  template  deterministic sentences, always available, the default
  llm       optional: Claude rewrites the same facts as a short note for the reviewer.
            Enabled only when EXPLANATIONS_LLM=true and Anthropic credentials are present.
            Any failure falls back to the template, so scoring never depends on the LLM.

Only derived features are sent to the LLM - never device, merchant or request ids.
"""

from __future__ import annotations

import json
import os

from app.utils.logger import get_logger

log = get_logger(__name__)

LLM_MODEL = "claude-opus-5-5"
SYSTEM_PROMPT = (
    "You write short notes for payment-fraud analysts who review flagged transactions. "
    "You are given the fraud model's score, its decision band, and the features that pushed the "
    "score up, each with the transaction's value and the typical value. Explain in two or three "
    "plain sentences why the transaction was flagged, then give one concrete thing the analyst "
    "should check. Use only the facts and numbers provided: never invent amounts, counts or "
    "history, and do not state that the transaction is fraud - the model only estimates risk. "
    "Plain text, no headings or bullet points."
)


def _money(v) -> str:
    return f"INR {v:,.0f}"


def _phrase(feature: str, value, raw: dict) -> str | None:
    """One human-readable sentence fragment per feature, or None when there is nothing to say."""
    v = value
    amount = raw.get("amount")
    phrases = {
        "log_amount": lambda: f"the amount ({_money(amount)}) is high" if amount else "the amount is unusual",
        "dev_logamt_z": lambda: "the amount is far above what this device normally spends",
        "dev_amt_ratio": lambda: f"the amount is {v:.1f}x this device's 30-day average" if v and v > 1.5 else None,
        "mer_logamt_z": lambda: "the amount is unusually large for this merchant",
        "amount_is_round": lambda: "the amount is a round figure" if v else None,
        "dev_cnt_10m": lambda: f"{int(v)} other payment(s) from this device in the last 10 minutes" if v else None,
        "dev_cnt_1h": lambda: f"{int(v)} other payment(s) from this device in the last hour" if v else None,
        "dev_cnt_24h": lambda: f"{int(v)} other payment(s) from this device in the last 24 hours" if v else None,
        "dev_cnt_7d": lambda: f"{int(v)} payment(s) from this device in the last 7 days",
        "dev_cnt_30d": lambda: (
            "the device has no payment history in the last 30 days"
            if not v
            else f"the device has only {int(v)} payment(s) in the last 30 days"
        ),
        "dev_amt_sum_1h": lambda: f"{_money(v)} already spent from this device in the last hour" if v else None,
        "dev_amt_sum_24h": lambda: f"{_money(v)} already spent from this device in the last 24 hours" if v else None,
        "dev_secs_since_last": lambda: (
            f"only {int(v)} seconds since the device's previous payment" if v is not None and v < 600 else None
        ),
        "dev_age_days": lambda: (
            "the device was first seen less than a day ago"
            if v is not None and v < 1
            else f"the device was first seen {v:.0f} day(s) ago" if v is not None and v < 30 else None
        ),
        "dev_new_merchant": lambda: "the device has not paid this merchant in the last 30 days" if v else None,
        "dev_new_merchants_24h": lambda: f"{int(v)} new merchant(s) tried from this device in 24 hours" if v else None,
        "dev_new_service": lambda: "the device has not used this payment method recently" if v else None,
        "dev_fail_cnt_1h": lambda: (
            f"{int(v)} failed or declined attempt(s) from this device in the last hour" if v else None
        ),
        "dev_fail_cnt_24h": lambda: (
            f"{int(v)} failed or declined attempt(s) from this device in 24 hours" if v else None
        ),
        "mer_cnt_1h": lambda: f"{int(v)} payment(s) at this merchant in the last hour",
        "mer_cnt_30d": lambda: "the merchant has little recent history" if v is not None and v < 20 else None,
        "mer_fraud_rate_lag": lambda: f"this merchant's recent confirmed fraud rate is {v:.1%}" if v else None,
        "is_night": lambda: "it happened late at night" if v else None,
        "hour": lambda: f"it happened at {int(v):02d}:00, an unusual hour",
        "hour_sin": lambda: "the time of day is unusual",
        "hour_cos": lambda: "the time of day is unusual",
        "merchant_type": lambda: f"the merchant is in the '{v}' risk tier",
        "mcc_code": lambda: f"the merchant category ({raw.get('mcc_title') or v}) carries elevated risk",
        "service_type": lambda: f"the payment method is {v}",
        "issuer_bank": lambda: f"issuer bank is {v}" if v else "issuer bank is missing",
        "merchant_state": lambda: f"the merchant is in {v}" if v else None,
    }
    fn = phrases.get(feature)
    try:
        return fn() if fn else None
    except (TypeError, ValueError):
        return None


def reason_phrases(reasons: list[dict], raw: dict) -> list[str]:
    seen, out = set(), []
    for r in reasons:
        p = _phrase(r["feature"], r["value"], raw)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def template_explanation(score: float, band: str, reasons: list[dict], raw: dict) -> str:
    phrases = reason_phrases(reasons, raw)
    head = {"block": "Blocked", "review": "Sent to manual review", "approve": "Approved"}.get(band, band)
    if not phrases:
        return f"{head} with a fraud score of {score:.0%}. No single factor stands out."
    body = "; ".join(phrases[:4])
    return f"{head} with a fraud score of {score:.0%}. Main factors: {body}."


def llm_enabled() -> bool:
    return os.getenv("EXPLANATIONS_LLM", "false").lower() == "true"


def llm_explanation(score: float, band: str, reasons: list[dict], raw: dict) -> str | None:
    """Ask Claude to word the explanation. Returns None on any failure (caller falls back)."""
    try:
        import anthropic
    except ImportError:
        return None
    facts = {
        "fraud_score": round(score, 4),
        "decision_band": band,
        "amount_inr": raw.get("amount"),
        "payment_method": raw.get("service_type"),
        "merchant_category": raw.get("mcc_title"),
        "merchant_risk_tier": raw.get("merchant_type"),
        "top_factors": [
            {
                "factor": p,
                "feature": r["feature"],
                "value": r["value"],
                "typical": r["typical"],
                "log_odds_contribution": r["log_odds_contribution"],
            }
            for r, p in zip(reasons, [_phrase(r["feature"], r["value"], raw) for r in reasons])
        ],
    }
    try:
        client = anthropic.Anthropic(timeout=30.0, max_retries=1)
        response = client.beta.messages.create(
            model=LLM_MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            output_config={"effort": "low"},
            # if a safety classifier declines, the API re-runs the request on a fallback model
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": json.dumps(facts, default=str)}],
        )
        if response.stop_reason == "refusal":
            log.warning("llm explanation refused")
            return None
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or None
    except anthropic.AuthenticationError:
        log.warning("llm explanation skipped: no valid Anthropic credentials")
    except anthropic.RateLimitError:
        log.warning("llm explanation skipped: rate limited")
    except anthropic.APIStatusError as e:
        log.warning("llm explanation failed: status %s", e.status_code)
    except anthropic.APIConnectionError:
        log.warning("llm explanation failed: connection error")
    except Exception as e:  # never let the optional LLM path break scoring
        log.warning("llm explanation failed: %s", type(e).__name__)
    return None


def explain(score: float, band: str, reasons: list[dict], raw: dict, use_llm: bool | None = None) -> dict:
    """`use_llm=False` opts a call out of the LLM. It can never opt in: only the operator's
    EXPLANATIONS_LLM setting enables LLM calls, so a caller cannot trigger spend."""
    if llm_enabled() and use_llm is not False:
        text = llm_explanation(score, band, reasons, raw)
        if text:
            return {"text": text, "source": "llm", "model": LLM_MODEL}
    return {"text": template_explanation(score, band, reasons, raw), "source": "template", "model": None}
