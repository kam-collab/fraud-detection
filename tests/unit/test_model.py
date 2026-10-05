"""The model wrapper: pipelines, calibration, decision bands and occlusion attributions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.controllers.features.builder import CATEGORICAL, FEATURES, NUMERIC
from app.controllers.training import trainer
from app.controllers.training.model import APPROVE, BLOCK, REVIEW, FraudModel, _logit, make_gbm, make_logreg


@pytest.fixture(scope="module")
def xy(txns, txn_features):
    return txn_features, txns["fraud_label"].to_numpy()


@pytest.fixture(scope="module", params=["gbm", "logreg"])
def model(request, xy) -> FraudModel:
    """A small fitted model of each kind with a non-trivial calibration and policy."""
    X, y = xy
    pipeline = make_gbm(max_iter=40) if request.param == "gbm" else make_logreg()
    pipeline.fit(X, y)
    return FraudModel(
        pipeline=pipeline,
        calib_a=0.8,
        calib_b=-0.7,
        review_threshold=0.2,
        block_threshold=0.6,
        version="test",
        reference=trainer.reference_values(X),
    )


def test_scores_are_probabilities(model, xy):
    X, _ = xy
    for scores in (model.raw_score(X), model.score(X)):
        assert scores.shape == (len(X),)
        assert np.isfinite(scores).all()
        assert scores.min() >= 0.0 and scores.max() <= 1.0


def test_model_separates_fraud_from_genuine(model, xy):
    X, y = xy
    scores = model.score(X)
    assert scores[y == 1].mean() > 5 * scores[y == 0].mean()


def test_platt_calibration_never_changes_the_ranking(model, xy):
    X, _ = xy
    raw, calibrated = model.raw_score(X), model.score(X)
    order = np.argsort(raw, kind="stable")
    assert (np.diff(calibrated[order]) >= 0).all()
    # distinct raw scores stay distinct (strictly monotone), away from the clipping at 1e-7
    inner = (raw[order] > 1e-6) & (raw[order] < 1 - 1e-6)
    strictly_up = np.diff(raw[order][inner]) > 1e-9
    assert (np.diff(calibrated[order][inner])[strictly_up] > 0).all()
    assert not np.allclose(raw, calibrated)  # the calibration does do something


def test_calibration_formula(model, xy):
    X, _ = xy
    raw = model.raw_score(X.head(50))
    expected = 1 / (
        1 + np.exp(-(0.8 * np.log(np.clip(raw, 1e-7, 1 - 1e-7) / (1 - np.clip(raw, 1e-7, 1 - 1e-7))) - 0.7))
    )
    np.testing.assert_allclose(model.score(X.head(50)), expected)


def test_fit_platt_recovers_a_known_miscalibration():
    """Scores that are systematically too high by a known logit shift are pulled back."""
    rng = np.random.default_rng(0)
    p_true = rng.beta(1, 8, 40_000)
    y = (rng.random(40_000) < p_true).astype(int)
    overconfident = 1 / (1 + np.exp(-(2.0 * _logit(p_true) + 1.0)))
    a, b = trainer.fit_platt(overconfident, y)
    assert a == pytest.approx(0.5, abs=0.05) and b == pytest.approx(-0.5, abs=0.1)


def test_decide_bands_respect_the_thresholds(model):
    scores = np.array([0.0, 0.19999, 0.2, 0.3, 0.59999, 0.6, 1.0])
    assert model.decide(scores).tolist() == [APPROVE, APPROVE, REVIEW, REVIEW, REVIEW, BLOCK, BLOCK]
    assert model.decide(0.2).item() == REVIEW


def test_decide_with_equal_thresholds_has_no_review_band(model):
    same = FraudModel(pipeline=model.pipeline, review_threshold=0.5, block_threshold=0.5)
    assert same.decide(np.array([0.49, 0.5, 0.9])).tolist() == [APPROVE, BLOCK, BLOCK]


def test_missing_numerics_are_scored(model, xy):
    X, _ = xy
    damaged = X.head(200).copy()
    damaged[NUMERIC] = damaged[NUMERIC].astype(float)
    damaged.loc[damaged.index[:100], NUMERIC] = np.nan  # every numeric missing
    damaged.loc[damaged.index[100:], ["dev_cnt_1h", "log_amount", "mer_fraud_rate_lag"]] = np.nan
    scores = model.score(damaged)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()


def test_unseen_and_missing_categories_are_scored(model, xy):
    X, _ = xy
    damaged = X.head(200).copy()
    damaged.loc[damaged.index[:100], CATEGORICAL] = ["crypto", "extreme", "9999", "NEWBANK", "Atlantis"]
    damaged.loc[damaged.index[100:], CATEGORICAL] = np.nan
    scores = model.score(damaged)
    assert np.isfinite(scores).all() and ((scores >= 0) & (scores <= 1)).all()
    # rows that were not touched keep their score when scored next to damaged rows
    mixed = pd.concat([X.iloc[200:300], damaged])
    np.testing.assert_allclose(model.score(mixed)[:100], model.score(X.iloc[200:300]))


def test_scoring_ignores_extra_columns_and_column_order(model, xy):
    X, _ = xy
    shuffled = X.head(100)[FEATURES[::-1]].assign(device_id="DEV1", junk=1.0)
    np.testing.assert_allclose(model.score(shuffled), model.score(X.head(100)))


# --------------------------------------------------------------- contributions
@pytest.fixture(scope="module")
def flagged_row(model, xy) -> pd.DataFrame:
    X, _ = xy
    return X.iloc[[int(np.argmax(model.score(X)))]]


def test_contributions_are_well_formed_sorted_and_positive(model, flagged_row):
    reasons = model.contributions(flagged_row)
    assert 1 <= len(reasons) <= 5
    for r in reasons:
        assert set(r) == {"feature", "value", "typical", "log_odds_contribution"}
        assert r["feature"] in FEATURES
        assert r["typical"] == model.reference[r["feature"]]
        assert r["log_odds_contribution"] > 0.05
        assert r["value"] is None or isinstance(r["value"], (str, float))
        actual = flagged_row.iloc[0][r["feature"]]
        assert (r["value"] is None and pd.isna(actual)) or r["value"] == actual
    contributions = [r["log_odds_contribution"] for r in reasons]
    assert contributions == sorted(contributions, reverse=True)
    assert len({r["feature"] for r in reasons}) == len(reasons)


def test_contribution_equals_the_log_odds_drop_when_the_feature_is_made_typical(model, flagged_row):
    """Independent re-computation of the occlusion value for every reported reason."""

    def log_odds(frame):
        p = model.score(frame)[0]
        return np.log(p / (1 - p))

    for r in model.contributions(flagged_row):
        occluded = flagged_row.copy()
        occluded[r["feature"]] = model.reference[r["feature"]]
        assert log_odds(flagged_row) - log_odds(occluded) == pytest.approx(r["log_odds_contribution"], abs=2e-3)


def test_contributions_respect_top_and_do_not_modify_the_input(model, flagged_row):
    before = flagged_row.copy()
    all_reasons = model.contributions(flagged_row, top=len(FEATURES))
    top2 = model.contributions(flagged_row, top=2)
    assert top2 == all_reasons[:2]
    pd.testing.assert_frame_equal(flagged_row, before)


def test_contributions_of_a_typical_transaction_are_empty(model):
    typical = pd.DataFrame([model.reference])[FEATURES]
    assert model.contributions(typical) == []


def test_contributions_report_missing_values_as_none(model, flagged_row):
    row = flagged_row.copy()
    row[NUMERIC] = row[NUMERIC].astype(float)
    row[["dev_logamt_z", "issuer_bank"]] = np.nan
    for r in model.contributions(row, top=len(FEATURES)):
        if r["feature"] in ("dev_logamt_z", "issuer_bank"):
            assert r["value"] is None


# ------------------------------------------------------- trainer helper pieces
def test_reference_values_are_medians_and_modes(xy):
    X, _ = xy
    ref = trainer.reference_values(X)
    assert set(ref) == set(FEATURES)
    assert ref["log_amount"] == pytest.approx(X["log_amount"].median())
    assert ref["dev_logamt_z"] == pytest.approx(X["dev_logamt_z"].median())  # NaN-aware
    assert ref["service_type"] == X["service_type"].mode().iloc[0]
    assert all(isinstance(ref[c], str) for c in CATEGORICAL)


def test_build_model_calibrates_and_orders_the_thresholds(xy):
    X, y = xy
    half = len(X) // 2
    pipeline = make_gbm(max_iter=40).fit(X.iloc[:half], y[:half])
    model = trainer.build_model(pipeline, X.iloc[:half], X.iloc[half:], y[half:], version="t")
    assert model.calib_a > 0  # increasing: calibration keeps the ranking
    assert 0 < model.review_threshold <= model.block_threshold <= 1
    assert model.version == "t" and set(model.reference) == set(FEATURES)
    # calibrated on the validation half: mean score there is close to the fraud rate
    assert model.score(X.iloc[half:]).mean() == pytest.approx(y[half:].mean(), rel=0.25)


def test_candidates_cover_the_baseline_and_the_tree_model():
    names = list(trainer.candidates(pos_ratio=99.0))
    assert sum(n.startswith("logreg") for n in names) == 2
    assert sum(n.startswith("gbm") for n in names) == 9
    assert len(set(names)) == len(names)


def test_split_windows_do_not_overlap_and_respect_the_label_delay(txns):
    masks = trainer.split_masks(txns)
    stacked = np.vstack([masks[k] for k in ("train", "valid", "test", "live")])
    assert stacked.sum(axis=0).max() == 1  # a row is in at most one window
    t = txns["request_time"]
    assert all(masks[k].sum() > 0 for k in masks)
    gap = pd.Timedelta(days=7)
    assert t[masks["valid"]].min() - t[masks["train"]].max() >= gap
    assert t[masks["test"]].min() - t[masks["valid"]].max() >= gap
    assert t[masks["live"]].min() > t[masks["test"]].max()
    assert t[masks["train"]].min() >= pd.Timestamp("2026-01-31")  # January is warm-up only
