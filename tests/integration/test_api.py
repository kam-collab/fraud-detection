"""API tests against the real model and the full stored history.

Startup loads ~567k rows and scores them, so one client is shared by the whole module.
Nothing here calls the network: the LLM path is replaced by a recorder, and the
prediction log is redirected to a temporary file.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.utils import config

pytestmark = pytest.mark.integration

MISSING = [str(p) for p in (config.MODEL_PATH, config.SYNTHETIC_CSV) if not p.exists()]
if MISSING:
    pytestmark = [
        pytest.mark.integration,
        pytest.mark.skip(reason=f"needs the trained model and data set (run `make pipeline`): {MISSING}"),
    ]

LLM_CALLS: list = []
ORIGINAL_LLM_EXPLANATION = None


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app.controllers.explanations import explainer
    from app.controllers.scoring import service as service_module
    from app.main import app

    global ORIGINAL_LLM_EXPLANATION
    ORIGINAL_LLM_EXPLANATION = explainer.llm_explanation
    patch = pytest.MonkeyPatch()
    patch.setenv("EXPLANATIONS_LLM", "false")
    patch.setattr(explainer, "llm_explanation", lambda *a, **k: LLM_CALLS.append(a) or None)
    patch.setattr(service_module, "PREDICTION_LOG", tmp_path_factory.mktemp("logs") / "predictions.jsonl")
    try:
        with TestClient(app) as c:
            assert app.state.service is not None, getattr(app.state, "load_error", "service failed to load")
            yield c
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def service(client):
    return client.app.state.service


def new_transaction(client, kind="genuine") -> dict:
    """A stored transaction turned into a brand-new request (no id, no time)."""
    txn = client.get("/api/v1/transactions/sample", params={"kind": kind}).json()
    txn.pop("request_id")
    txn.pop("request_time")
    return txn


def metric_value(text: str, name: str) -> float:
    match = re.search(rf"^{re.escape(name)} (\S+)$", text, flags=re.M)
    assert match, f"{name} not found in /metrics"
    return float(match.group(1))


# ---------------------------------------------------------------- health checks
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_readiness_reports_the_loaded_model_and_history(client, service):
    r = client.get("/readiness")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "model_version": service.model.version, "history_rows": len(service.df)}
    assert r.json()["history_rows"] > 100_000


def test_metrics_expose_the_custom_metrics_after_a_scoring_call(client):
    scored = client.post("/api/v1/score", json=new_transaction(client))
    assert scored.status_code == 200
    r = client.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    for name in (
        "fraud_score_bucket",
        "fraud_score_count",
        "fraud_decisions_total",
        "fraud_explanations_total",
        "fraud_validation_failures_total",
        "fraud_model_info",
        "fraud_review_threshold",
        "fraud_block_threshold",
        "http_requests_total",
        "http_request_duration_seconds_bucket",
    ):
        assert name in text, name
    assert metric_value(text, "fraud_score_count") >= 1
    assert f'fraud_decisions_total{{band="{scored.json()["band"]}"}}' in text
    assert 'fraud_explanations_total{source="template"}' in text
    assert re.search(r'http_requests_total\{method="POST",path="/api/v1/score",status="200"\} [1-9]', text)


def test_metrics_report_the_thresholds_of_the_loaded_model(client, service):
    text = client.get("/metrics").text
    assert metric_value(text, "fraud_review_threshold") == pytest.approx(service.model.review_threshold)
    assert metric_value(text, "fraud_block_threshold") == pytest.approx(service.model.block_threshold)
    assert f'fraud_model_info{{version="{service.model.version}"}} 1.0' in text


def test_http_metrics_use_the_route_template_not_the_raw_path(client):
    client.get("/api/v1/explain/NOPE-123")
    text = client.get("/metrics").text
    assert 'path="/api/v1/explain/{request_id}",status="404"' in text
    assert "NOPE-123" not in text


# ------------------------------------------------- training / serving consistency
SAMPLE_KINDS = ["random"] * 8 + ["fraud"] * 4 + ["genuine"] * 3 + ["review"] * 3 + ["blocked"] * 2


def test_replayed_sample_transactions_get_their_batch_score(client, service):
    """Training-serving skew check: the online path must reproduce the batch score."""
    worst, seen = 0.0, set()
    for kind in SAMPLE_KINDS:
        txn = client.get("/api/v1/transactions/sample", params={"kind": kind}).json()
        assert "fraud_label" not in txn and "request_status" not in txn
        pos = int(service._pos[txn["request_id"]])
        r = client.post("/api/v1/score", json=txn)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["request_id"] == txn["request_id"]
        assert body["score"] == pytest.approx(float(service.scores[pos]), abs=1e-6), (kind, txn["request_id"])
        assert body["band"] == service.bands[pos]
        worst = max(worst, abs(body["score"] - float(service.scores[pos])))
        seen.add(body["band"])
    assert worst <= 1e-6
    assert {"review", "block"} <= seen  # the comparison covered the flagged bands


def test_replay_matches_batch_scores_across_the_whole_history(client, service):
    """Same check on rows `sample` never returns: every month, including the warm-up weeks,
    the highest scores, and rows that share device and second with another row."""
    df = service.df
    rng = np.random.default_rng(2026)
    same_second = np.flatnonzero(df.duplicated(["device_id", "request_time"], keep=False).to_numpy())
    # rows whose history reaches back beyond the 37 days the service keeps per request:
    # long-lived devices (first seen > 37 days earlier) and devices returning after a long silence
    secs = df["request_time"].astype("int64") // 10**9
    age_days = (secs - secs.groupby(df["device_id"]).transform("first")) / 86400
    gap_days = secs.groupby(df["device_id"]).diff() / 86400
    old_device = np.flatnonzero((age_days > 37).to_numpy())
    returning = np.flatnonzero((gap_days > 37).to_numpy())
    assert len(old_device) > 1000 and len(returning) > 20
    assert (service.X["dev_age_days"].to_numpy()[old_device] == 30).all()
    positions = np.r_[
        rng.choice(len(df), 40, replace=False),
        np.argsort(-service.scores)[:10],
        same_second[:10],
        [0, 1, len(df) - 1],
        rng.choice(old_device, 25, replace=False),
        rng.choice(returning, 20, replace=False),
    ]
    for pos in positions:
        row = df.iloc[int(pos)]
        txn = {c: row[c] for c in config.RAW_COLUMNS if c != config.TARGET}
        txn = {k: (None if pd.isna(v) else v) for k, v in txn.items()}
        txn["request_time"] = row["request_time"].isoformat()
        txn["mcc_code"], txn["amount"] = int(row["mcc_code"]), float(row["amount"])
        r = client.post("/api/v1/score", json=txn)
        assert r.status_code == 200, r.text
        assert r.json()["score"] == pytest.approx(float(service.scores[int(pos)]), abs=1e-6), row["request_id"]


def test_replays_are_idempotent_and_leave_no_trace_in_the_session(client, service):
    txn = client.get("/api/v1/transactions/sample", params={"kind": "fraud"}).json()
    before = len(service.session)
    first = client.post("/api/v1/score", json=txn).json()
    second = client.post("/api/v1/score", json=txn).json()
    assert first["score"] == second["score"] and first["reasons"] == second["reasons"]
    assert len(service.session) == before


def test_score_response_shape(client, service):
    body = client.post("/api/v1/score", json=client.get("/api/v1/transactions/sample?kind=blocked").json()).json()
    assert set(body) == {
        "request_id",
        "score",
        "band",
        "thresholds",
        "model_version",
        "reasons",
        "explanation",
        "warnings",
    }
    assert body["band"] == "block" and body["score"] >= body["thresholds"]["block"]
    assert body["thresholds"] == {"review": service.model.review_threshold, "block": service.model.block_threshold}
    assert body["model_version"] == service.model.version
    assert body["explanation"]["source"] == "template" and body["explanation"]["text"].startswith("Blocked")
    assert body["reasons"] and all(r["log_odds_contribution"] > 0 for r in body["reasons"])
    assert body["warnings"] == []


# ------------------------------------------------------------- new transactions
def test_new_transaction_gets_an_id_and_repeating_it_raises_device_velocity(client, service):
    txn = new_transaction(client)
    first = client.post("/api/v1/score", json=txn)
    second = client.post("/api/v1/score", json=txn)
    assert first.status_code == 200 and second.status_code == 200
    a, b = first.json(), second.json()
    assert a["request_id"] and b["request_id"] and a["request_id"] != b["request_id"]
    assert a["request_id"] not in service._pos.index  # generated ids never collide with stored ones
    assert 0 <= a["score"] <= 1 and a["band"] in ("approve", "review", "block")

    fa = service.session_results[a["request_id"]]["features"].iloc[0]
    fb = service.session_results[b["request_id"]]["features"].iloc[0]
    for count in ("dev_cnt_10m", "dev_cnt_1h", "dev_cnt_24h", "dev_cnt_30d", "mer_cnt_1h"):
        assert fb[count] == fa[count] + 1, count
    assert fb["dev_amt_sum_1h"] == pytest.approx(fa["dev_amt_sum_1h"] + txn["amount"])
    assert fb["dev_secs_since_last"] == 1  # the service clock ticks one second per request
    assert fb["dev_new_merchant"] == 0
    assert a["score"] != b["score"]


def test_new_transaction_on_an_unknown_device_and_merchant(client):
    txn = new_transaction(client)
    txn.update(device_id="DEV-NEVER-SEEN", merchant_id="M-NEVER-SEEN")
    r = client.post("/api/v1/score", json=txn)
    assert r.status_code == 200
    reasons = {x["feature"] for x in r.json()["reasons"]}
    assert 0 <= r.json()["score"] <= 1
    assert not reasons & {"dev_cnt_10m", "dev_cnt_1h", "dev_fail_cnt_1h"}  # no history, no velocity


def test_optional_fields_may_be_omitted(client):
    r = client.post(
        "/api/v1/score",
        json={
            "service_type": "upi",
            "device_id": "DEV-MINIMAL",
            "merchant_id": "M-MINIMAL",
            "merchant_type": "low",
            "mcc_code": 5411,
            "amount": 850.0,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json()["request_id"].startswith("TXN")


def test_timezone_aware_request_time_is_scored(client):
    txn = new_transaction(client)
    txn["request_time"] = "2026-09-15T10:00:00Z"
    lenient = TestClient(client.app, raise_server_exceptions=False)  # no second startup
    r = lenient.post("/api/v1/score", json=txn)
    assert r.status_code == 200, r.text
    assert 0 <= r.json()["score"] <= 1


@pytest.mark.parametrize("zone, suffix", [("UTC", "Z"), ("Asia/Kolkata", "+05:30"), ("America/New_York", None)])
def test_timezone_aware_replay_is_the_same_instant_as_the_stored_gateway_local_time(client, service, zone, suffix):
    """Stored times are naive gateway-local (IST). The same instant written in any timezone
    must reproduce the batch score, time-of-day features included."""
    for kind in ("review", "blocked", "random"):
        txn = client.get("/api/v1/transactions/sample", params={"kind": kind}).json()
        local = pd.Timestamp(txn["request_time"]).tz_localize("Asia/Kolkata")
        aware = local.tz_convert(zone).isoformat()
        if suffix:
            aware = aware[:19] + suffix
            assert pd.Timestamp(aware) == local
        r = client.post("/api/v1/score", json={**txn, "request_time": aware})
        assert r.status_code == 200, r.text
        pos = int(service._pos[txn["request_id"]])
        assert r.json()["score"] == pytest.approx(float(service.scores[pos]), abs=1e-6), (kind, aware)


# ------------------------------------------------------------------ validation
def test_negative_amount_is_rejected_with_422(client):
    txn = new_transaction(client)
    txn["amount"] = -50
    before = metric_value(client.get("/metrics").text, "fraud_validation_failures_total")
    r = client.post("/api/v1/score", json=txn)
    assert r.status_code == 422
    assert r.json()["detail"]["errors"] == ["amount: 1 values <= 0"]
    assert metric_value(client.get("/metrics").text, "fraud_validation_failures_total") == before + 1


@pytest.mark.parametrize("field", ["amount", "device_id", "merchant_id", "service_type", "merchant_type", "mcc_code"])
def test_missing_required_field_is_rejected_with_422(client, field):
    txn = new_transaction(client)
    del txn[field]
    r = client.post("/api/v1/score", json=txn)
    assert r.status_code == 422
    assert any(field in err["loc"] for err in r.json()["detail"])


@pytest.mark.parametrize(
    "change", [{"mcc_code": 0}, {"mcc_code": 12345}, {"amount": 0}, {"amount": "lots"}, {"request_time": "yesterday"}]
)
def test_impossible_values_are_rejected_with_422(client, change):
    txn = new_transaction(client)
    txn.update(change)
    assert client.post("/api/v1/score", json=txn).status_code == 422


def test_rejected_transactions_do_not_enter_the_history(client, service):
    before = len(service.session)
    txn = new_transaction(client)
    txn["amount"] = -1
    client.post("/api/v1/score", json=txn)
    assert len(service.session) == before


def test_unknown_service_type_is_a_warning_not_an_error(client):
    txn = new_transaction(client)
    txn["service_type"] = "carrier_pigeon"
    r = client.post("/api/v1/score", json=txn)
    assert r.status_code == 200
    assert r.json()["warnings"] == ["service_type: unseen values ['carrier_pigeon']"]
    assert 0 <= r.json()["score"] <= 1


def test_sample_kind_is_validated(client):
    assert client.get("/api/v1/transactions/sample", params={"kind": "anything"}).status_code == 422


@pytest.mark.parametrize("kind, label", [("fraud", 1), ("genuine", 0)])
def test_sample_returns_a_september_transaction_of_the_requested_kind(client, service, kind, label):
    txn = client.get("/api/v1/transactions/sample", params={"kind": kind}).json()
    row = service.df.iloc[int(service._pos[txn["request_id"]])]
    assert row["fraud_label"] == label
    assert row["request_time"] >= pd.Timestamp("2026-09-01")


# ---------------------------------------------------------------- review queue
def test_review_queue_is_sorted_and_paginates_consistently(client, service):
    whole = client.get("/api/v1/review-queue", params={"limit": 10}).json()
    assert whole["total"] >= 10, "expected a non-trivial review queue in the last three days"
    assert len(whole["items"]) == 10
    scores = [i["score"] for i in whole["items"]]
    assert scores == sorted(scores, reverse=True)
    thresholds = service.model
    assert all(thresholds.review_threshold <= s < thresholds.block_threshold for s in scores)
    assert all(i["band"] == "review" for i in whole["items"])

    page1 = client.get("/api/v1/review-queue", params={"limit": 5, "offset": 0}).json()
    page2 = client.get("/api/v1/review-queue", params={"limit": 5, "offset": 5}).json()
    assert [i["request_id"] for i in page1["items"] + page2["items"]] == [i["request_id"] for i in whole["items"]]
    assert page1["total"] == page2["total"] == whole["total"]

    beyond = client.get("/api/v1/review-queue", params={"limit": 5, "offset": whole["total"]}).json()
    assert beyond["items"] == [] and beyond["total"] == whole["total"]


def test_review_queue_items_hide_identifiers_and_cover_only_recent_days(client, service):
    queue = client.get("/api/v1/review-queue", params={"limit": 200}).json()
    newest = service.df["request_time"].max()
    for item in queue["items"]:
        assert "device_id" not in item and "merchant_id" not in item and "fraud_label" not in item
        assert pd.Timestamp(item["request_time"]) >= newest - pd.Timedelta(days=3)
    assert pd.Timestamp(queue["since"]) == (newest - pd.Timedelta(days=3)).floor("s")


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
def test_review_queue_rejects_bad_paging(client, params):
    assert client.get("/api/v1/review-queue", params=params).status_code == 422


def test_actual_label_is_hidden_until_the_analyst_decides(client, service):
    before = client.get("/api/v1/review-queue", params={"limit": 50}).json()
    undecided = [i for i in before["items"] if i["decision"] is None]
    assert undecided and all(i["actual_label"] is None for i in undecided)
    item = undecided[0]
    truth = int(service.df.iloc[int(service._pos[item["request_id"]])]["fraud_label"])

    r = client.post(f"/api/v1/review-queue/{item['request_id']}/decision", json={"decision": "decline"})
    assert r.status_code == 200
    assert r.json() == {
        "request_id": item["request_id"],
        "decision": "decline",
        "actual_label": truth,
        "correct": truth == 1,
    }

    after = client.get("/api/v1/review-queue", params={"limit": 50}).json()
    decided = next(i for i in after["items"] if i["request_id"] == item["request_id"])
    assert decided["decision"] == "decline" and decided["actual_label"] == truth
    assert after["pending"] == before["pending"] - 1 and after["total"] == before["total"]
    others = [i for i in after["items"] if i["decision"] is None]
    assert all(i["actual_label"] is None for i in others)

    # an "approve" is correct exactly when the transaction was genuine
    r = client.post(f"/api/v1/review-queue/{item['request_id']}/decision", json={"decision": "approve"})
    assert r.json()["correct"] == (truth == 0)


def test_decision_on_unknown_request_is_404(client):
    r = client.post("/api/v1/review-queue/TXN-DOES-NOT-EXIST/decision", json={"decision": "approve"})
    assert r.status_code == 404


def test_decision_must_be_approve_or_decline(client, service):
    rid = service.df["request_id"].iloc[-1]
    assert client.post(f"/api/v1/review-queue/{rid}/decision", json={"decision": "maybe"}).status_code == 422
    assert client.post(f"/api/v1/review-queue/{rid}/decision", json={}).status_code == 422
    assert rid not in service.decisions


# ---------------------------------------------------------------------- explain
def test_explain_a_stored_transaction_uses_the_template_by_default(client, service):
    rid = client.get("/api/v1/review-queue", params={"limit": 1}).json()["items"][0]["request_id"]
    r = client.get(f"/api/v1/explain/{rid}")
    assert r.status_code == 200
    body = r.json()
    pos = int(service._pos[rid])
    assert body["request_id"] == rid and body["band"] == "review"
    assert body["score"] == pytest.approx(float(service.scores[pos]))
    assert body["explanation"]["source"] == "template" and body["explanation"]["model"] is None
    assert body["explanation"]["text"].startswith("Sent to manual review with a fraud score of")
    assert body["reasons"] and body["reason_phrases"]
    assert all(phrase in body["explanation"]["text"] for phrase in body["reason_phrases"][:4])
    row = service.df.iloc[pos]
    assert row["device_id"] not in r.text and row["merchant_id"] not in r.text


def test_explain_agrees_with_the_score_endpoint(client):
    txn = client.get("/api/v1/transactions/sample", params={"kind": "blocked"}).json()
    scored = client.post("/api/v1/score", json=txn).json()
    explained = client.get(f"/api/v1/explain/{txn['request_id']}").json()
    # batch and online feature values agree to ~1e-11 (floating-point summation order), not bit for bit
    assert [r["feature"] for r in explained["reasons"]] == [r["feature"] for r in scored["reasons"]]
    for batch, online in zip(explained["reasons"], scored["reasons"]):
        assert batch["log_odds_contribution"] == pytest.approx(online["log_odds_contribution"], abs=2e-3)
        assert batch["value"] == (
            pytest.approx(online["value"], rel=1e-6) if isinstance(online["value"], float) else online["value"]
        )
    assert explained["explanation"]["text"] == scored["explanation"]["text"]


def test_explain_a_transaction_scored_in_this_session(client):
    scored = client.post("/api/v1/score", json=new_transaction(client, "fraud")).json()
    r = client.get(f"/api/v1/explain/{scored['request_id']}")
    assert r.status_code == 200
    assert r.json()["score"] == scored["score"] and r.json()["reasons"] == scored["reasons"]


def test_explain_unknown_request_is_404(client):
    assert client.get("/api/v1/explain/TXN-DOES-NOT-EXIST").status_code == 404


@pytest.mark.parametrize("params", [{}, {"llm": "true"}])
def test_explain_falls_back_to_the_template_when_the_enabled_llm_fails(client, monkeypatch, params):
    rid = client.get("/api/v1/review-queue", params={"limit": 1}).json()["items"][0]["request_id"]
    template = client.get(f"/api/v1/explain/{rid}").json()["explanation"]["text"]
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")  # operator enabled it; the fake LLM returns None
    calls = len(LLM_CALLS)
    r = client.get(f"/api/v1/explain/{rid}", params=params)
    assert r.status_code == 200
    assert len(LLM_CALLS) == calls + 1  # the (faked, failing) LLM was tried
    assert r.json()["explanation"] == {"text": template, "source": "template", "model": None}


def test_explain_falls_back_when_the_anthropic_client_raises(client, monkeypatch):
    """Same, one level lower: the real llm_explanation runs and its client blows up."""
    import anthropic

    from app.controllers.explanations import explainer

    attempts = []

    def broken_client(**kwargs):
        attempts.append(kwargs)
        raise RuntimeError("no network in tests")

    monkeypatch.setattr(explainer, "llm_explanation", ORIGINAL_LLM_EXPLANATION)
    monkeypatch.setattr(anthropic, "Anthropic", broken_client)
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    rid = client.get("/api/v1/review-queue", params={"limit": 1}).json()["items"][0]["request_id"]
    r = client.get(f"/api/v1/explain/{rid}")
    assert r.status_code == 200 and len(attempts) == 1
    assert r.json()["explanation"]["source"] == "template"


def test_llm_text_is_served_when_the_operator_enabled_it_and_the_call_succeeds(client, monkeypatch):
    from app.controllers.explanations import explainer

    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    monkeypatch.setattr(explainer, "llm_explanation", lambda *a, **k: "Analyst note.")
    rid = client.get("/api/v1/review-queue", params={"limit": 1}).json()["items"][0]["request_id"]
    assert client.get(f"/api/v1/explain/{rid}").json()["explanation"] == {
        "text": "Analyst note.",
        "source": "llm",
        "model": explainer.LLM_MODEL,
    }


def test_llm_false_opts_a_request_out_when_the_operator_enabled_the_llm(client, monkeypatch):
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")
    rid = client.get("/api/v1/review-queue", params={"limit": 1}).json()["items"][0]["request_id"]
    calls = len(LLM_CALLS)
    r = client.get(f"/api/v1/explain/{rid}", params={"llm": "false"})
    assert r.status_code == 200 and r.json()["explanation"]["source"] == "template"
    assert len(LLM_CALLS) == calls


def test_scoring_never_calls_the_llm(client, monkeypatch):
    monkeypatch.setenv("EXPLANATIONS_LLM", "true")  # even when enabled: scoring is latency-critical
    calls = len(LLM_CALLS)
    client.post("/api/v1/score", json=new_transaction(client, "fraud"))
    assert len(LLM_CALLS) == calls


@pytest.mark.parametrize("env", ["false", None])
def test_llm_query_flag_cannot_enable_the_llm_when_the_operator_has_not(client, monkeypatch, env):
    if env is None:
        monkeypatch.delenv("EXPLANATIONS_LLM")
    else:
        monkeypatch.setenv("EXPLANATIONS_LLM", env)
    rid = client.get("/api/v1/review-queue", params={"limit": 1}).json()["items"][0]["request_id"]
    calls = len(LLM_CALLS)
    r = client.get(f"/api/v1/explain/{rid}", params={"llm": "true"})
    assert r.status_code == 200 and r.json()["explanation"]["source"] == "template"
    assert len(LLM_CALLS) == calls


# --------------------------------------------------------- tracing and the log
def test_trace_id_header_is_echoed(client):
    r = client.get("/health", headers={"X-Trace-Id": "trace-abc-123"})
    assert r.headers["X-Trace-Id"] == "trace-abc-123"
    r = client.get("/api/v1/explain/NOPE", headers={"X-Trace-Id": "trace-on-error"})
    assert r.status_code == 404 and r.headers["X-Trace-Id"] == "trace-on-error"


def test_trace_id_is_generated_when_absent(client):
    a, b = client.get("/health").headers["X-Trace-Id"], client.get("/health").headers["X-Trace-Id"]
    assert re.fullmatch(r"[0-9a-f]{16}", a) and a != b


def test_prediction_log_records_score_features_and_trace_id_without_identifiers(client):
    from app.controllers.scoring import service as service_module

    txn = new_transaction(client)
    body = client.post("/api/v1/score", json=txn, headers={"X-Trace-Id": "trace-for-log"}).json()
    record = json.loads(service_module.PREDICTION_LOG.read_text().splitlines()[-1])
    assert record["request_id"] == body["request_id"]
    assert record["trace_id"] == "trace-for-log"
    assert record["score"] == pytest.approx(body["score"], abs=1e-6) and record["band"] == body["band"]
    from app.controllers.features.builder import FEATURES

    assert list(record["features"]) == FEATURES
    assert txn["device_id"] not in json.dumps(record) and txn["merchant_id"] not in json.dumps(record)


def test_prediction_log_was_redirected_away_from_the_project(client):
    from app.controllers.scoring import service as service_module

    assert config.ROOT not in service_module.PREDICTION_LOG.parents


# ----------------------------------------------------------- read-only reports
def test_model_endpoint_matches_the_loaded_model(client, service):
    r = client.get("/api/v1/model")
    assert r.status_code == 200
    body = r.json()
    assert body["card"]["thresholds"]["review"] == pytest.approx(service.model.review_threshold)
    assert body["card"]["thresholds"]["block"] == pytest.approx(service.model.block_threshold)
    assert body["card"]["model_version"] == service.model.version
    assert body["card"]["features"] == service.model.features
    assert body["champion"]["name"].startswith("gbm") and body["baseline"]["name"].startswith("logreg")
    assert body["champion"]["test"]["pr_auc"] >= body["baseline"]["test"]["pr_auc"]


def test_stored_training_report_is_reproduced_by_the_loaded_model(service):
    """The numbers shown in the model card are the numbers this model gives on this data."""
    from app.controllers.evaluation import metrics as ev
    from app.controllers.training.trainer import split_masks

    report = json.loads((config.REPORTS_DIR / "training_report.json").read_text())
    test = split_masks(service.df)["test"]
    y = service.df[config.TARGET].to_numpy()[test]
    now = ev.evaluate(y, service.scores[test], service.model.review_threshold)
    then = report["champion"]["test"]
    assert then["n"] == now["n"]
    for key in ("pr_auc", "roc_auc", "precision", "recall", "tp", "fp", "fn", "tn"):
        assert now[key] == pytest.approx(then[key], rel=1e-9), key


def test_review_policy_met_its_precision_floor_on_validation(service):
    from app.controllers.evaluation import metrics as ev
    from app.controllers.training.trainer import split_masks

    valid = split_masks(service.df)["valid"]
    y = service.df[config.TARGET].to_numpy()[valid]
    review = ev.threshold_metrics(y, service.scores[valid], service.model.review_threshold)
    block = ev.threshold_metrics(y, service.scores[valid], service.model.block_threshold)
    assert review["precision"] >= config.REVIEW_MIN_PRECISION
    assert block["precision"] >= config.BLOCK_MIN_PRECISION
    assert service.model.review_threshold < service.model.block_threshold


def test_monitoring_endpoint_serves_the_three_live_months(client):
    r = client.get("/api/v1/monitoring")
    assert r.status_code == 200
    assert [m["month"] for m in r.json()["months"]] == ["2026-07", "2026-08", "2026-09"]
    assert all(m["status"] in ("ok", "warn", "alert") for m in r.json()["months"])


def test_overview_covers_the_development_period_only(client, service):
    r = client.get("/api/v1/overview")
    assert r.status_code == 200
    body = r.json()
    dev = service.df[service.df["request_time"] < pd.Timestamp(config.LIVE_START)]
    assert body["transactions"] == len(dev) and body["frauds"] == int(dev["fraud_label"].sum())
    assert body["fraud_rate"] == pytest.approx(dev["fraud_label"].mean())
    assert sum(x["transactions"] for x in body["by_service_type"]) == len(dev)
    assert sum(x["transactions"] for x in body["by_amount"]) == len(dev)
    assert [x["hour"] for x in body["by_hour"]] == list(range(24))
    assert len(body["monthly"]) == 9
