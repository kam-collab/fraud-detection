"""Evaluation metrics against hand-computed values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.controllers.evaluation import metrics as ev
from app.utils import config


# ----------------------------------------------------------- threshold_metrics
def test_threshold_metrics_against_a_hand_computed_confusion_matrix():
    y = [1, 1, 0, 0, 1, 0, 0, 0]
    score = [0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.5]  # the last one sits exactly on the threshold
    m = ev.threshold_metrics(y, score, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 2, 1, 3)
    assert m["precision"] == pytest.approx(2 / 4)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(4 / 7)
    assert m["accuracy"] == pytest.approx(5 / 8)
    assert m["false_positive_rate"] == pytest.approx(2 / 5)
    assert m["flag_rate"] == pytest.approx(4 / 8)
    assert m["threshold"] == 0.5


def test_threshold_metrics_when_nothing_is_flagged():
    m = ev.threshold_metrics([1, 0, 0], [0.2, 0.1, 0.3], 0.9)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (0, 0, 1, 2)
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["flag_rate"] == 0.0


def test_evaluate_handles_a_window_without_fraud():
    m = ev.evaluate([0, 0, 0, 0], [0.1, 0.2, 0.8, 0.3], 0.5)
    assert math.isnan(m["pr_auc"]) and math.isnan(m["roc_auc"]) and math.isnan(m["brier"])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (0, 1, 0, 3)
    assert m["precision"] == 0.0 and m["recall"] == 0.0
    assert m["false_positive_rate"] == pytest.approx(0.25)
    assert m["n"] == 4 and m["fraud_rate"] == 0.0


def test_evaluate_on_a_perfect_ranking():
    m = ev.evaluate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["pr_auc"] == pytest.approx(1.0) and m["roc_auc"] == pytest.approx(1.0)
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["fraud_rate"] == 0.5


def test_ranking_metrics_are_nan_when_every_row_is_fraud():
    assert all(math.isnan(v) for v in ev.ranking_metrics([1, 1], [0.3, 0.9]).values())


# ----------------------------------------------------- threshold_for_precision
@pytest.fixture(scope="module")
def scored():
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.08).astype(int)
    score = np.clip(0.3 * y + rng.normal(0.3, 0.15, 4000), 0, 1)
    return y, score


def brute_force_threshold(y, score, floor, min_flags=30):
    """Lowest observed score t such that flagging score >= t keeps precision >= floor."""
    ok = [t for t in np.unique(score) if (score >= t).sum() >= min_flags and y[score >= t].mean() >= floor]
    return min(ok) if ok else None


@pytest.mark.parametrize("floor", [0.2, 0.5, 0.8, 0.95])
def test_threshold_for_precision_meets_the_floor_and_is_the_lowest_such_threshold(scored, floor):
    y, score = scored
    t = ev.threshold_for_precision(y, score, floor)
    flagged = score >= t
    assert flagged.sum() >= 30
    assert y[flagged].mean() >= floor
    assert ev.threshold_metrics(y, score, t)["precision"] >= floor
    assert t == pytest.approx(brute_force_threshold(y, score, floor))


def test_higher_precision_floor_never_lowers_the_threshold(scored):
    y, score = scored
    thresholds = [ev.threshold_for_precision(y, score, f) for f in (0.2, 0.5, 0.8, 0.95)]
    assert thresholds == sorted(thresholds)
    recalls = [ev.threshold_metrics(y, score, t)["recall"] for t in thresholds]
    assert recalls == sorted(recalls, reverse=True)


def test_threshold_for_precision_ignores_thresholds_with_fewer_than_30_flags():
    """The top 10 scores are all fraud (precision 1.0), but 10 flags are too few to trust:
    the rule must go down to the first threshold with at least 30 flags that meets the floor."""
    y = np.r_[np.ones(10), np.zeros(5), np.ones(25), np.zeros(160)].astype(int)
    score = np.linspace(1.0, 0.0, len(y))
    t = ev.threshold_for_precision(y, score, 0.86)
    flagged = score >= t
    assert flagged.sum() == 40  # 35 of 40 = 0.875 qualifies, 35 of 41 does not
    assert y[flagged].mean() == pytest.approx(35 / 40)


def test_threshold_for_precision_falls_back_to_a_top_quantile_when_the_floor_is_unreachable(scored):
    y, score = scored
    useless = np.random.default_rng(1).random(len(y))  # no signal: 95% precision is impossible
    t = ev.threshold_for_precision(y, useless, 0.95)
    assert t == pytest.approx(np.quantile(useless, 0.999))
    assert (useless >= t).mean() <= 0.002  # flags almost nothing rather than everything


def test_threshold_for_max_f1_is_the_best_f1_among_observed_scores(scored):
    y, score = scored
    t = ev.threshold_for_max_f1(y, score)
    best = max(ev.threshold_metrics(y, score, u)["f1"] for u in np.unique(score)[::20])
    assert ev.threshold_metrics(y, score, t)["f1"] >= best - 1e-12


# ------------------------------------------------------------ business_metrics
def test_business_metrics_arithmetic_on_a_tiny_example():
    #            blocked        reviewed       approved
    y = [1, 0, 1, 0, 1, 0, 0]
    score = [0.95, 0.90, 0.60, 0.50, 0.10, 0.2, 0.49]
    amount = [1000, 5000, 2000, 300, 400, 50, 70]
    m = ev.business_metrics(y, score, amount, review_threshold=0.5, block_threshold=0.9)

    assert m["transactions"] == 7
    assert m["blocked"] == 2 and m["review_queue"] == 2
    assert m["approval_rate"] == pytest.approx(3 / 7) and m["review_rate"] == pytest.approx(2 / 7)
    assert m["false_declines"] == 1 and m["false_decline_amount"] == 5000
    assert m["genuine_sent_to_review"] == 1

    # approved fraud is lost in full; reviewed fraud is lost when the analyst misses it
    fraud_loss = 400 + (1 - config.REVIEW_CATCH_RATE) * 2000
    assert config.REVIEW_CATCH_RATE == 0.90 and fraud_loss == pytest.approx(600)
    assert m["fraud_amount_total"] == 3400
    assert m["fraud_loss_amount"] == pytest.approx(fraud_loss)
    assert m["fraud_amount_prevented"] == pytest.approx(3400 - fraud_loss)
    assert m["fraud_value_recall"] == pytest.approx(1 - fraud_loss / 3400)

    false_decline_cost = config.FALSE_DECLINE_MARGIN * 5000 + config.FALSE_DECLINE_FIXED * 1
    review_cost = config.REVIEW_COST * 2
    assert m["total_cost"] == pytest.approx(fraud_loss + false_decline_cost + review_cost)
    assert m["cost_without_model"] == 3400


def test_business_metrics_bands_partition_the_traffic():
    rng = np.random.default_rng(5)
    y, score, amount = (rng.random(500) < 0.1).astype(int), rng.random(500), rng.lognormal(6, 1, 500)
    m = ev.business_metrics(y, score, amount, 0.6, 0.85)
    assert m["blocked"] + m["review_queue"] + round(m["approval_rate"] * 500) == 500
    assert m["fraud_amount_prevented"] + m["fraud_loss_amount"] == pytest.approx(m["fraud_amount_total"])
    assert 0 <= m["fraud_value_recall"] <= 1


def test_business_metrics_with_equal_thresholds_has_no_review_band():
    m = ev.business_metrics([1, 0, 1], [0.9, 0.9, 0.1], [100, 200, 300], 0.5, 0.5)
    assert m["review_queue"] == 0 and m["blocked"] == 2 and m["false_declines"] == 1
    assert m["fraud_loss_amount"] == 300


def test_business_metrics_without_fraud():
    m = ev.business_metrics([0, 0], [0.95, 0.1], [1000, 10], 0.5, 0.9)
    assert m["fraud_amount_total"] == 0 and m["fraud_loss_amount"] == 0
    assert math.isnan(m["fraud_value_recall"])
    assert m["total_cost"] == pytest.approx(config.FALSE_DECLINE_MARGIN * 1000 + config.FALSE_DECLINE_FIXED)


# ------------------------------------------------------------------- the rest
def test_recall_by_group():
    y = [1, 1, 1, 1, 0, 1]
    score = [0.9, 0.1, 0.8, 0.7, 0.9, 0.2]
    group = ["A", "A", "B", "B", "A", "C"]
    out = ev.recall_by_group(y, score, group, 0.5).set_index("group")
    assert out["frauds"].to_dict() == {"A": 2, "B": 2, "C": 1}
    assert out["recall"].to_dict() == {"A": 0.5, "B": 1.0, "C": 0.0}


def test_bootstrap_ci_brackets_the_point_estimate_and_is_reproducible(scored):
    y, score = scored
    point = ev.evaluate(y, score, 0.5)
    ci = ev.bootstrap_ci(y, score, 0.5, n_boot=100)
    for name in ("pr_auc", "roc_auc", "precision", "recall", "f1"):
        lo, hi = ci[name]
        assert lo < point[name] < hi, name
        assert hi - lo < 0.25, name
    assert ci == ev.bootstrap_ci(y, score, 0.5, n_boot=100)


def test_pr_curve_points_are_valid_and_ordered(scored):
    y, score = scored
    points = ev.pr_curve_points(y, score, n=50)
    assert 2 <= len(points) <= 50
    assert all(0 <= p["precision"] <= 1 and 0 <= p["recall"] <= 1 for p in points)
    assert [p["threshold"] for p in points] == sorted(p["threshold"] for p in points)
    assert points[0]["recall"] == 1.0
