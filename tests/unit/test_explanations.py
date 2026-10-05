"""Analyst explanations: deterministic template, and the optional LLM path with its fallbacks.

No test here talks to the network: the Anthropic client is always replaced by a fake.
"""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import httpx
import pytest

from app.controllers.explanations import explainer

anthropic = pytest.importorskip("anthropic")

RAW = {
    "request_id": "TXN7777777",
    "request_time": "2026-09-12 01:14:00",
    "service_type": "wallet",
    "device_id": "DEV0009999",
    "merchant_id": "M123456",
    "merchant_state": "Kerala",
    "merchant_city": "Kochi",
    "merchant_type": "high",
    "mcc_code": 7995,
    "mcc_title": "Betting and Gambling",
    "issuer_bank": "NOT_APPLICABLE",
    "currency_code": "INR",
    "amount": 48250.0,
    "request_status": None,
}
REASONS = [
    {"feature": "dev_cnt_10m", "value": 4.0, "typical": 0.0, "log_odds_contribution": 2.1},
    {"feature": "log_amount", "value": 10.78, "typical": 6.9, "log_odds_contribution": 1.4},
    {"feature": "dev_amt_ratio", "value": 12.34, "typical": 1.0, "log_odds_contribution": 0.9},
    {"feature": "mcc_code", "value": "7995", "typical": "5411", "log_odds_contribution": 0.5},
    {"feature": "is_night", "value": 1.0, "typical": 0.0, "log_odds_contribution": 0.2},
]


# -------------------------------------------------------------------- template
@pytest.mark.parametrize(
    "band, opening", [("block", "Blocked"), ("review", "Sent to manual review"), ("approve", "Approved")]
)
def test_template_names_the_band_the_score_and_the_main_factors(band, opening):
    text = explainer.template_explanation(0.8731, band, REASONS, RAW)
    assert text.startswith(f"{opening} with a fraud score of 87%.")
    assert "4 other payment(s) from this device in the last 10 minutes" in text
    assert "the amount (INR 48,250) is high" in text
    assert "12.3x this device's 30-day average" in text
    assert "Betting and Gambling" in text
    assert "late at night" not in text  # only the four strongest factors are listed


@pytest.mark.parametrize("band", ["block", "review", "approve"])
def test_template_with_no_reasons(band):
    text = explainer.template_explanation(0.031, band, [], RAW)
    assert text.endswith("with a fraud score of 3%. No single factor stands out.")
    assert "Main factors" not in text


def test_template_never_mentions_identifiers():
    text = explainer.template_explanation(0.9, "block", REASONS, RAW)
    for identifier in (RAW["device_id"], RAW["merchant_id"], RAW["request_id"]):
        assert identifier not in text


def test_reason_phrases_skip_uninformative_values_and_duplicates():
    reasons = [
        {"feature": "hour_sin", "value": 0.5},
        {"feature": "hour_cos", "value": -0.2},  # same wording
        {"feature": "dev_cnt_10m", "value": 0.0},  # nothing to say about zero payments
        {"feature": "dev_amt_ratio", "value": 1.1},  # not unusual
        {"feature": "dev_secs_since_last", "value": 86400.0},  # a day ago is not "only N seconds"
        {"feature": "dev_cnt_1h", "value": None},  # missing value must not crash
        {"feature": "not_a_feature", "value": 1.0},
        {"feature": "issuer_bank", "value": None},
        {"feature": "dev_secs_since_last", "value": 42.0},
    ]
    assert explainer.reason_phrases(reasons, RAW) == [
        "the time of day is unusual",
        "issuer bank is missing",
        "only 42 seconds since the device's previous payment",
    ]


def test_every_model_feature_can_be_phrased_without_crashing():
    from app.controllers.features.builder import CATEGORICAL, NUMERIC

    for feature in NUMERIC:
        for value in (None, 0.0, 1.0, 3.7, 1e6):
            phrase = explainer._phrase(feature, value, RAW)
            assert phrase is None or (isinstance(phrase, str) and "None" not in phrase and "nan" not in phrase)
    for feature in CATEGORICAL:
        assert isinstance(explainer._phrase(feature, "x", RAW), str)


# ------------------------------------------------------- LLM switch and fallback
@pytest.mark.parametrize("value, expected", [("true", True), ("TRUE", True), ("false", False), ("1", False)])
def test_llm_enabled_reads_the_environment(monkeypatch, value, expected):
    monkeypatch.setenv("EXPLANATIONS_LLM", value)
    assert explainer.llm_enabled() is expected


def test_llm_is_disabled_by_default(monkeypatch):
    monkeypatch.delenv("EXPLANATIONS_LLM", raising=False)
    assert explainer.llm_enabled() is False


def _forbid_llm(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("the LLM must not be called")

    monkeypatch.setattr(explainer, "llm_explanation", boom)
    monkeypatch.setattr(anthropic, "Anthropic", boom)


def test_explain_uses_the_template_when_the_llm_is_disabled(monkeypatch):
    monkeypatch.setenv("EXPLANATIONS_LLM", "false")
    _forbid_llm(monkeypatch)
    out = explainer.explain(0.9, "block", REASONS, RAW)
    assert out == {
        "text": explainer.template_explanation(0.9, "block", REASONS, RAW),
        "source": "template",
        "model": None,
    }


def test_explicit_use_llm_false_overrides_the_environment(monkeypatch):
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    _forbid_llm(monkeypatch)
    assert explainer.explain(0.9, "block", REASONS, RAW, use_llm=False)["source"] == "template"


@pytest.mark.parametrize("env", ["false", None])
@pytest.mark.parametrize("use_llm", [True, None, False])
def test_a_caller_can_never_opt_in_when_the_operator_has_not_enabled_the_llm(monkeypatch, env, use_llm):
    """Only EXPLANATIONS_LLM=true enables LLM calls; `use_llm=True` must not trigger spend."""
    if env is None:
        monkeypatch.delenv("EXPLANATIONS_LLM", raising=False)
    else:
        monkeypatch.setenv("EXPLANATIONS_LLM", env)
    _forbid_llm(monkeypatch)
    assert explainer.explain(0.9, "block", REASONS, RAW, use_llm=use_llm)["source"] == "template"


@pytest.mark.parametrize("use_llm", [True, None])
def test_llm_runs_when_enabled_by_the_operator_unless_the_call_opts_out(monkeypatch, use_llm):
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    monkeypatch.setattr(explainer, "llm_explanation", lambda *a, **k: "A short note.")
    assert explainer.explain(0.9, "block", REASONS, RAW, use_llm=use_llm) == {
        "text": "A short note.",
        "source": "llm",
        "model": explainer.LLM_MODEL,
    }
    assert explainer.explain(0.9, "block", REASONS, RAW, use_llm=False)["source"] == "template"


def test_explain_falls_back_to_the_template_when_the_llm_returns_nothing(monkeypatch):
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    calls = []
    monkeypatch.setattr(explainer, "llm_explanation", lambda *a, **k: calls.append(a) or None)
    out = explainer.explain(0.42, "review", REASONS, RAW)
    assert len(calls) == 1  # the LLM path was tried...
    assert out["source"] == "template" and out["model"] is None  # ...and the template answered
    assert out["text"] == explainer.template_explanation(0.42, "review", REASONS, RAW)


# ---------------------------------------------------------- fake Anthropic client
class FakeClient:
    """Stands in for anthropic.Anthropic; records what would have been sent."""

    def __init__(self, behaviour, sent: list, **client_kwargs):
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._behaviour, self._sent = behaviour, sent

    def _create(self, **kwargs):
        self._sent.append(kwargs)
        if isinstance(self._behaviour, Exception):
            raise self._behaviour
        return self._behaviour


def reply(text="Several payments in ten minutes. Check with the customer.", stop_reason="end_turn"):
    return SimpleNamespace(stop_reason=stop_reason, content=[SimpleNamespace(type="text", text=text)])


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a fake client; returns (set_behaviour, sent_requests)."""
    sent: list = []
    state = {"behaviour": reply()}
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kw: FakeClient(state["behaviour"], sent, **kw))
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    return (lambda b: state.update(behaviour=b)), sent


def test_llm_text_is_used_when_the_call_succeeds(fake_llm):
    _, sent = fake_llm
    out = explainer.explain(0.9, "block", REASONS, RAW)
    assert out == {
        "text": "Several payments in ten minutes. Check with the customer.",
        "source": "llm",
        "model": explainer.LLM_MODEL,
    }
    assert len(sent) == 1


API_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
FAILURES = {
    "unexpected exception": RuntimeError("boom"),
    "connection error": anthropic.APIConnectionError(request=API_REQUEST),
    "timeout": anthropic.APITimeoutError(request=API_REQUEST),
    "bad credentials": anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx.Response(401, request=API_REQUEST), body=None
    ),
    "rate limited": anthropic.RateLimitError("slow down", response=httpx.Response(429, request=API_REQUEST), body=None),
    "server error": anthropic.InternalServerError(
        "overloaded", response=httpx.Response(529, request=API_REQUEST), body=None
    ),
    "model refusal": reply(stop_reason="refusal"),
    "empty answer": reply(text="   "),
    "malformed response": SimpleNamespace(stop_reason="end_turn", content=None),
}


@pytest.mark.parametrize("failure", FAILURES)
def test_explain_falls_back_to_the_template_when_the_llm_call_fails(fake_llm, failure):
    set_behaviour, sent = fake_llm
    set_behaviour(FAILURES[failure])
    out = explainer.explain(0.9, "block", REASONS, RAW)
    assert len(sent) == 1  # the call was attempted
    assert out["source"] == "template" and out["model"] is None
    assert out["text"] == explainer.template_explanation(0.9, "block", REASONS, RAW)


def test_fallback_when_the_client_cannot_even_be_constructed(monkeypatch):
    def no_credentials(**kwargs):
        raise anthropic.AnthropicError("no API key")

    monkeypatch.setattr(anthropic, "Anthropic", no_credentials)
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")  # enabled, so the client really is constructed
    assert explainer.llm_explanation(0.9, "block", REASONS, RAW) is None
    assert explainer.explain(0.9, "block", REASONS, RAW)["source"] == "template"


def test_llm_payload_contains_the_facts_but_no_identifiers(fake_llm):
    _, sent = fake_llm
    explainer.explain(0.87314, "review", REASONS, RAW)
    request = sent[0]
    wire = json.dumps(request, default=str)
    for identifier in (RAW["device_id"], RAW["merchant_id"], RAW["request_id"]):
        assert identifier not in wire
    assert "device_id" not in wire and "merchant_id" not in wire and "request_id" not in wire

    assert request["model"] == explainer.LLM_MODEL and request["system"] == explainer.SYSTEM_PROMPT
    assert [m["role"] for m in request["messages"]] == ["user"]
    facts = json.loads(request["messages"][0]["content"])
    assert facts["fraud_score"] == 0.8731 and facts["decision_band"] == "review"
    assert facts["amount_inr"] == 48250.0 and facts["merchant_category"] == "Betting and Gambling"
    assert [f["feature"] for f in facts["top_factors"]] == [r["feature"] for r in REASONS]
    assert facts["top_factors"][0] == {
        "factor": "4 other payment(s) from this device in the last 10 minutes",
        "feature": "dev_cnt_10m",
        "value": 4.0,
        "typical": 0.0,
        "log_odds_contribution": 2.1,
    }


def test_llm_request_matches_the_installed_sdk_signature(fake_llm):
    """A keyword the SDK does not know would raise TypeError, be swallowed by the broad
    except, and silently disable the LLM path for good. Bind against the real signature."""
    _, sent = fake_llm
    explainer.llm_explanation(0.9, "block", REASONS, RAW)
    from anthropic.resources.beta.messages import Messages

    inspect.signature(Messages.create).bind(None, **sent[0])
