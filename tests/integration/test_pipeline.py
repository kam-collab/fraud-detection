"""End-to-end run of the pipeline on a small generated data set:
generate -> validate -> features -> train/select/calibrate -> evaluate -> monitor -> serve.

Needs neither the stored model nor the stored data, and must not write to models/ or reports/.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from app.controllers.data_generation.generator import generate
from app.controllers.features.builder import FEATURES, build_features
from app.controllers.monitoring import report as monitoring
from app.controllers.scoring import service as service_module
from app.controllers.training import trainer
from app.controllers.training.model import FraudModel
from app.controllers.validation.schema import validate
from app.utils import config

pytestmark = pytest.mark.integration


def artefact_snapshot() -> dict:
    return {
        str(p): (p.stat().st_mtime_ns, p.stat().st_size)
        for d in (config.MODELS_DIR, config.REPORTS_DIR)
        if d.exists()
        for p in d.rglob("*")
    }


@pytest.fixture(scope="module")
def run():
    before = artefact_snapshot()
    df = generate(n_genuine=40_000, n_devices=2_400, n_merchants=400, seed=11)
    result = trainer.train(df=df, save=False)
    return {"before": before, "after_train": artefact_snapshot(), "df": df, **result}


def test_generated_data_passes_validation(run):
    result = validate(run["df"].drop(columns=["fraud_pattern"]))
    assert result.ok, result.errors


def test_training_does_not_write_artefacts_when_save_is_false(run):
    assert run["after_train"] == run["before"]


def test_windows_are_populated_and_reported(run):
    report, df = run["report"], run["df"]
    assert report["data"]["rows"] == len(df)
    for name in ("train", "valid", "test", "live"):
        assert report["rows"][name]["n"] == int(run["masks"][name].sum()) > 1000
        assert report["rows"][name]["frauds"] == int(df.loc[run["masks"][name], "fraud_label"].sum()) > 0
    assert report["windows"]["label_delay_days"] == config.LABEL_DELAY_DAYS
    assert report["features"] == FEATURES


def test_feature_frame_is_aligned_with_the_data(run):
    assert run["X"].index.equals(run["df"].index)
    np.testing.assert_array_equal(run["y"], run["df"]["fraud_label"].to_numpy())
    pd.testing.assert_frame_equal(run["X"], build_features(run["df"]))


def test_selection_table_ranks_all_candidates_by_validation_pr_auc(run):
    table = run["report"]["selection_table"]
    assert len(table) == 11
    pr_aucs = [row["valid_pr_auc"] for row in table]
    assert pr_aucs == sorted(pr_aucs, reverse=True)
    assert all(0 < v <= 1 for v in pr_aucs)
    champion, baseline = run["report"]["champion"]["name"], run["report"]["baseline"]["name"]
    assert champion == next(r["model"] for r in table if r["model"].startswith("gbm"))
    assert baseline == next(r["model"] for r in table if r["model"].startswith("logreg"))


@pytest.mark.parametrize("who", ["champion", "baseline"])
def test_models_are_usable_and_clearly_better_than_chance(run, who):
    model, rep = run[who], run["report"][who]
    assert isinstance(model, FraudModel)
    assert 0 < model.review_threshold <= model.block_threshold <= 1
    assert (rep["review_threshold"], rep["block_threshold"]) == (model.review_threshold, model.block_threshold)
    test = rep["test"]
    assert test["roc_auc"] > 0.85
    assert test["pr_auc"] > 10 * test["fraud_rate"]  # far above the no-skill PR-AUC
    assert test["tp"] + test["fn"] == run["report"]["rows"]["test"]["frauds"]
    assert test["tp"] + test["fp"] + test["fn"] + test["tn"] == run["report"]["rows"]["test"]["n"]


def test_reported_test_metrics_are_reproducible_from_the_returned_model(run):
    from app.controllers.evaluation import metrics as ev

    champion, mask = run["champion"], run["masks"]["test"]
    scores = champion.score(run["X"][mask])
    again = ev.evaluate(run["y"][mask], scores, champion.review_threshold)
    for key in ("pr_auc", "roc_auc", "brier", "precision", "recall", "tp", "fp"):
        assert again[key] == pytest.approx(run["report"]["champion"]["test"][key]), key


def test_calibration_improves_the_brier_score_or_leaves_it_alone(run):
    rep = run["report"]["champion"]
    assert rep["test"]["brier"] <= rep["test_brier_uncalibrated"] * 1.10
    mask = run["masks"]["valid"]
    mean_score = run["champion"].score(run["X"][mask]).mean()
    assert mean_score == pytest.approx(run["y"][mask].mean(), rel=0.2)  # calibrated in the large


def test_report_is_json_serialisable_and_has_no_nan_headline_metrics(run):
    text = json.dumps(run["report"])
    assert json.loads(text)["model_version"] == trainer.MODEL_VERSION
    for who in ("champion", "baseline"):
        for key in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
            assert math.isfinite(run["report"][who]["test"][key])
            lo, hi = run["report"][who]["test_ci95"][key]
            assert lo <= hi
    patterns = {r["group"] for r in run["report"]["champion"]["test_recall_by_pattern"]}
    assert "D_upi_scam" not in patterns  # the new pattern does not exist before July


def test_monitoring_report_on_the_small_run(run):
    before = artefact_snapshot()
    report = monitoring.monitor(df=run["df"], model=run["champion"], save=False)
    assert artefact_snapshot() == before == run["before"]
    assert [m["month"] for m in report["months"]] == ["2026-07", "2026-08", "2026-09"]
    live_rows = sum(m["rows"] for m in report["months"])
    assert live_rows == int(run["masks"]["live"].sum())
    for month in report["months"]:
        assert month["status"] in ("ok", "warn", "alert")
        assert month["actions"] and all(a["severity"] in ("ok", "warn", "alert") for a in month["actions"])
        assert month["status"] == max((a["severity"] for a in month["actions"]), key=["ok", "warn", "alert"].index)
        drift = month["prediction_drift"]
        assert math.isfinite(drift["score_psi"])
        assert 0 <= drift["flag_rate"] <= 1 and 0 <= drift["baseline_flag_rate"] <= 1
        assert math.isfinite(drift["flag_rate_change"])
        assert all(math.isfinite(f["psi"]) for f in month["data_drift"])
        p = month["performance"]
        assert p["tp"] + p["fp"] + p["fn"] + p["tn"] == month["rows"]
    # the simulated September incident (issuer_bank missing ~10%) is caught without labels
    september = report["months"][2]
    assert any("issuer_bank" in w for w in september["data_quality"]["warnings"])
    assert not any("issuer_bank" in w for w in report["months"][0]["data_quality"]["warnings"])
    assert len(report["remediation"]["rows"]) == 3
    json.dumps(report, default=float)


def test_online_scoring_reproduces_batch_scores_for_the_freshly_trained_model(run, tmp_path, monkeypatch):
    """Training-serving consistency without the stored artefacts: replay stored rows one by
    one through ScoringService and compare with the batch scores of the same model."""
    monkeypatch.setattr(service_module, "PREDICTION_LOG", tmp_path / "predictions.jsonl")
    df = run["df"].drop(columns=["fraud_pattern"]).astype({"mcc_code": "Int64"})
    service = service_module.ScoringService(model=run["champion"], df=df)
    np.testing.assert_allclose(service.scores, run["champion"].score(run["X"]))

    # include rows whose device was first seen, or last seen, more than the 37 days of
    # history the service gathers per request (it adds the device's first row for dev_age_days)
    secs = df["request_time"].astype("int64") // 10**9
    age_days = (secs - secs.groupby(df["device_id"]).transform("first")) / 86400
    gap_days = secs.groupby(df["device_id"]).diff() / 86400
    old_device = np.flatnonzero((age_days > 37).to_numpy())
    returning = np.flatnonzero((gap_days > 37).to_numpy())
    assert len(old_device) > 100 and len(returning) > 10

    rng = np.random.default_rng(3)
    positions = np.r_[
        rng.choice(len(df), 150, replace=False),
        np.argsort(-service.scores)[:50],
        rng.choice(old_device, 60, replace=False),
        rng.choice(returning, 10, replace=False),
    ]
    for pos in positions:
        row = df.iloc[int(pos)]
        txn = {c: service_module._clean(row[c]) for c in config.RAW_COLUMNS if c != config.TARGET}
        out = service.score(txn)
        assert out["score"] == pytest.approx(float(service.scores[int(pos)]), abs=1e-6), row["request_id"]
        assert out["band"] == service.bands[int(pos)]
    assert len(service.session) == 0  # replays are not added to the history
    assert len((tmp_path / "predictions.jsonl").read_text().splitlines()) == len(positions)


def test_new_transaction_features_equal_features_built_from_the_full_history(run, tmp_path, monkeypatch):
    """The non-replay path: the service gathers 37 days of history plus the device's first
    row. The features it scores must equal those built from the complete history."""
    monkeypatch.setattr(service_module, "PREDICTION_LOG", tmp_path / "predictions.jsonl")
    df = run["df"].drop(columns=["fraud_pattern"]).astype({"mcc_code": "Int64"})
    service = service_module.ScoringService(model=run["champion"], df=df)

    last_seen = df.groupby("device_id")["request_time"].agg(["min", "max", "size"])
    end = df["request_time"].max()
    dormant = last_seen[(end - last_seen["max"] > pd.Timedelta(days=45)) & (last_seen["size"] >= 3)].index[:5]
    long_lived = last_seen[
        (end - last_seen["min"] > pd.Timedelta(days=200)) & (end - last_seen["max"] < pd.Timedelta(days=5))
    ].index[:5]
    assert len(dormant) == 5 and len(long_lived) == 5

    for device in list(dormant) + list(long_lived):
        template = df[df["device_id"] == device].iloc[-1]
        txn = {
            c: service_module._clean(template[c])
            for c in config.RAW_COLUMNS
            if c not in (config.TARGET, "request_id", "request_time", "request_status")
        }
        out = service.score(txn)
        stored = service.session_results[out["request_id"]]
        new_row = pd.DataFrame([{**stored["raw"], config.TARGET: 0}])
        new_row["request_time"] = pd.to_datetime(new_row["request_time"])
        full = pd.concat([df, service.session.iloc[:-1], new_row], ignore_index=True)
        expected = build_features(full.astype({"mcc_code": "Int64"})).iloc[[-1]]
        pd.testing.assert_frame_equal(
            stored["features"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=False,
            rtol=1e-7,
            atol=1e-6,
            check_dtype=False,
        )
        assert stored["features"]["dev_age_days"].iloc[0] == 30
        assert out["score"] == pytest.approx(float(run["champion"].score(expected)[0]), abs=1e-6)
