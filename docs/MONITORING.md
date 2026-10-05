# Drift detection and monitoring

Scenario from the assignment: the model was developed on January-June and receives live
traffic in July-September. `make monitor` produces `reports/monitoring_report.json`, the
notebook walks through it, and the console's Monitoring page displays it.

## What is monitored

| Layer | Needs labels? | Method | Reference | Alert rule |
|---|---|---|---|---|
| Data quality | No | Schema, type, range and missing-rate validator | Fixed limits | Any error; missing rate above its limit |
| Data drift | No | PSI (numeric: reference deciles; categorical: category shares) and the KS D statistic | Training window | PSI >= 0.10 warn, >= 0.25 alert; KS D >= 0.05 warn, >= 0.10 alert |
| Prediction drift | No | PSI of the score distribution; share of payments flagged | June hold-out | PSI as above; flagged share changes by 25% warn, 50% alert |
| Performance drift | Yes (7-day delay) | PR-AUC, ROC-AUC, precision, recall, F1 at the frozen thresholds | June hold-out with bootstrap 95% CI | Below the CI lower bound warn; below 85% of baseline alert |
| Business metrics | Partly | Fraud loss amount, false declines, approval rate, review queue per day | June hold-out | False-decline rate doubles |

Notes on the choices:

- **Reference windows differ on purpose.** Inputs are compared with the data the model was
  fitted on. Scores and metrics are compared with the out-of-sample June window, because
  scores on training rows are optimistic.
- **KS alerts use the D statistic, not the p-value.** With tens of thousands of rows per
  month every p-value is "significant"; D is an effect size.
- **PSI cut-offs are conventions, not tests.** PSI depends on bin count and sample size
  (Yurdakul 2018). They are used here as triage levels.
- **Score PSI alone is not enough.** About 97% of scores sit in the lowest bin, so PSI on
  the whole distribution barely moves when the risky tail grows. The share of payments at
  or above the review threshold is the sensitive label-free signal, and is what sizes the
  analyst queue.
- **Label-free layers are leading indicators.** Performance can only be measured after the
  label delay, so data and prediction drift are what you see first.

## What happened in July-September

| | June (baseline) | July | August | September |
|---|---|---|---|---|
| Overall status | | warn | alert | alert |
| `amount` PSI / KS D | | 0.001 / 0.010 | 0.062 / 0.091 | 0.081 / 0.107 |
| `mcc_code` PSI | | 0.001 | 0.088 | 0.089 |
| `service_type` PSI | | 0.002 | 0.024 | 0.062 |
| `issuer_bank` PSI (missing share) | | 0.001 (1.0%) | 0.007 (1.0%) | 0.233 (9.9%) |
| Score PSI | | 0.000 | 0.027 | 0.055 |
| Share flagged (change) | 1.59% | 1.64% (+3%) | 1.96% (+23%) | 2.47% (+56%) |
| Fraud rate | 1.11% | 1.22% | 1.76% | 2.18% |
| PR-AUC | 0.752 | 0.700 | 0.474 | 0.417 |
| Precision | 0.533 | 0.534 | 0.424 | 0.383 |
| Recall | 0.762 | 0.719 | 0.472 | 0.435 |
| Review queue per day | 21 | 22 | 35 | 53 |
| False declines | 11 | 30 | 23 | 80 |
| Fraud prevented, by value | 90.2% | 91.2% | 80.5% | 80.0% |

Reading it:

- **July** looks stable on every label-free signal. The only finding is a small drop in
  recall and PR-AUC once labels arrive.
- **August** shows a moderate shift in `amount` (the KS statistic warns before PSI does)
  and in the merchant-category mix. The flagged share is up 23%, just under the 25% warning
  level, so the label-free signals understate a month in which precision and recall both
  fall sharply. That gap is the argument for monitoring labelled performance as soon as
  labels mature, and for treating the thresholds here as starting points to tune.
- **September** adds a data-quality incident (`issuer_bank` missing in 9.9% of rows against
  1% in training), an alert on `amount`, and a 56% rise in the flagged share. PR-AUC is at
  55% of its baseline.

Recall by fraud pattern (available only because the data is simulated) explains the
performance drop: the three patterns the model was trained on are still caught at similar
rates, while the UPI scam pattern that begins in mid-July is caught 10-13% of the time and
grows to 981 frauds in September. That is concept drift: the relationship between inputs
and fraud changed, which input-drift statistics alone would not have revealed.

## Actions

| Issue | First action | Follow-up |
|---|---|---|
| Validator error, or missing rate above limit | **Investigate data quality** with the upstream owner; check for a schema or pipeline change | Exclude affected rows from retraining until fixed |
| Input drift, performance unknown or stable | Confirm it is a real business change (pricing, promotion, product mix), not a fault | Watch prediction drift; schedule a challenger |
| Prediction drift (flagged share up or down) | **Adjust the decision thresholds** to keep the review queue within analyst capacity | Confirm with labels when they arrive |
| Precision down (more false positives) | **Re-tune thresholds** on the latest labelled window to restore the precision floors | Raise the block threshold first: send borderline cases to review instead of declining |
| Recall or PR-AUC down (fraud missed) | **Train a challenger** on recent data and run it in **shadow** against the champion | Promote if it wins at the same alert rate; then **retrain** on a schedule |
| Severe collapse right after a release | **Roll back** to the previous model version | Root-cause before re-releasing |
| Severe collapse that built up gradually | Add a temporary rule for the new pattern | Fast-track the challenger |

### Demonstrated on September

Labels up to the end of August are available by 8 September, so three options are compared
on 8-30 September traffic:

| Option | PR-AUC | Precision | Recall | False declines | Review queue | Fraud loss (INR) |
|---|---|---|---|---|---|---|
| Champion v1.0.0 unchanged | 0.404 | 0.371 | 0.428 | 59 | 1,321 | 3.47M |
| Champion, thresholds re-tuned on August | 0.404 | 0.433 | 0.396 | 43 | 1,012 | 3.99M |
| Challenger v1.1.0, trained to end of July, in shadow | 0.443 | 0.456 | 0.433 | 41 | 1,032 | 3.57M |

Each row uses its own thresholds, so only PR-AUC is strictly like-for-like.

- Re-tuning thresholds is cheap and immediate. It cuts false declines and the review queue
  but cannot improve ranking, so more fraud gets through.
- The challenger ranks better (PR-AUC 0.443 against 0.404) and, at its own thresholds, has
  fewer false declines and a smaller queue for about the same recall. Its fraud loss by
  value is slightly higher than the unchanged champion's, so it is an improvement in
  ranking and customer impact, not a clear win on every measure.
- None of the three gets close to the June baseline. The new pattern looks like ordinary small UPI
  payments, so the existing features do not separate it well. The real fix is new signals
  (payee or beneficiary history, collect-request flags, customer-reported scam data), not
  only retraining.

## Rollback plan

- Each release is a versioned artefact (`models/fraud_model.joblib`, `model_card.json` with
  version, thresholds, evaluation and intervals). The previous artefact is kept.
- Rolling back means restoring the previous artefact and restarting the service; the API
  reports the loaded version at `/readiness` and `/metrics`, and every prediction-log line
  carries it.
- Thresholds live in the artefact, so a threshold-only change is also a versioned release.

## In production this would also need

- Monitoring on a rolling window (daily or weekly), not calendar months.
- A small random share of blocked transactions let through, or reviewed, to obtain
  unbiased labels. Otherwise blocked payments never receive a label and the model is
  evaluated only on what it approved.
- Calibration monitoring, and alert routing to an on-call owner.
