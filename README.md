# Payment Gateway Fraud Detection

A classical-ML fraud model for payment-gateway transactions, built for a take-home
assignment, with a scoring API, an analyst console and drift monitoring.

- **Assignment answers, in order:** [`notebooks/fraud_detection.ipynb`](notebooks/fraud_detection.ipynb) (executed, outputs included)
- **Design decisions:** [`docs/ARCH.md`](docs/ARCH.md)
- **Drift monitoring and runbook:** [`docs/MONITORING.md`](docs/MONITORING.md)
- **Assignment brief:** [`docs/Fraud_Detection_Assignment.pdf`](docs/Fraud_Detection_Assignment.pdf)

## Data used

The provided `data/fraud_detection_sample_transactions.csv` has 12 rows. It defines the
schema but cannot train a model, so, as the brief allows, this repository uses **synthetic
data in the same schema**: 566,754 transactions for January-September 2026 from a seeded
generator (`app/controllers/data_generation/generator.py`, seed 42). The generated file is
committed as `data/synthetic_transactions.csv.gz` and can be regenerated with `make data`.

The generator simulates genuine device behaviour, four fraud patterns, and deliberate drift
in July-September. **All results below measure how well the model recovers simulated
patterns. They are not evidence of performance on real transactions.**

## Quick start

```bash
make setup        # Python 3.10+; creates .venv, installs app/requirements.txt
make pipeline     # generate data -> train -> monitor (about 2 minutes)
make test         # unit tests       (make test-all adds integration tests)
make api          # scoring API on http://localhost:8000  (docs at /docs)

make frontend-setup && make frontend    # analyst console on http://localhost:3000
```

Details: [`docs/SETUP.md`](docs/SETUP.md), [`docs/USAGE.md`](docs/USAGE.md).

## Approach

| Step | Choice |
|---|---|
| Split | By time. January is feature warm-up, fit on February-April, select and calibrate on May, report on June, monitor July-September. 7-day gaps mirror the label delay. |
| Features | 28 numeric and 5 categorical, all computed from strictly earlier transactions with 30-day lookbacks: amount against device and merchant history, device velocity, new merchant or payment method, earlier failed attempts, delayed merchant fraud rate. |
| Not used directly | `request_id`, `device_id`, `merchant_id` (identifiers; used only to build history), `merchant_city`, `mcc_title`, `currency_code`, and the current payment's `request_status` (leakage: not known at decision time). |
| Baseline | Logistic Regression (imputation, scaling, one-hot). |
| Tree model | scikit-learn `HistGradientBoostingClassifier` (native missing values and categoricals). |
| Imbalance | No resampling. Class weights compared on validation; the operating point is set with thresholds. Scores are Platt-calibrated. |
| Decision | Three bands: **block** (precision >= 95% on validation), **manual review** (precision >= 50%), approve. |

## Results on the June hold-out

47,584 transactions, 529 frauds (1.11%). Thresholds were fixed on May before June was scored.

| | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.643 | 0.960 | 0.581 | 0.637 | 0.608 |
| **Gradient boosting** | **0.752** | **0.980** | 0.533 | **0.762** | **0.627** |
| 95% interval (gradient boosting) | 0.712-0.792 | 0.972-0.986 | 0.497-0.569 | 0.719-0.799 | 0.594-0.658 |

Precision, recall and F1 are at the review threshold (flagged = review or block).

Confusion matrix, gradient boosting, flagged vs not flagged:

| | Predicted fraud | Predicted genuine |
|---|---|---|
| **Actual fraud** | 403 | 126 |
| **Actual genuine** | 353 | 46,702 |

### False positives

The 353 flagged genuine payments are not all declined. The policy splits them:

| Band | Transactions | Precision | Share of fraud caught | Genuine customers affected |
|---|---|---|---|---|
| Block (auto-decline) | 266 | 95.9% | 48.2% | **11 false declines** (0.02% of traffic) |
| Manual review | 490 (about 21 a day) | 30.2% | a further 28.0% | 342 reviewed, not declined |
| Approve | 46,828 | | 23.8% missed | |

Measured by value, 90.2% of fraud amount is stopped (this assumes analysts catch 90% of the
fraud they review; the cost figures in `config.py` are illustrative assumptions).

For comparison, a single cut-off at the default 0.5 gives precision 0.923 and recall 0.565
with 25 false declines: fewer reviews, but 104 more frauds missed.

### Which metric matters

**PR-AUC** for ranking, and **recall at a fixed precision** for the operating point. With
1% fraud, accuracy is uninformative (approving everything scores 98.9%, the model 99.0%)
and ROC-AUC is flattered by the huge number of easy genuine payments. The business decision
is how much fraud can be stopped for a given number of customers inconvenienced, which is
exactly recall at a precision floor.

### Where the model is weak

Recall by simulated pattern: account takeover 90%, mule devices 92%, **stealth fraud 43%**.
Fraud that looks like normal spending is mostly missed, as expected.

### Class imbalance finding

On validation PR-AUC the three weightings are within noise (0.722 none, 0.732 square-root,
0.719 balanced), but balanced weights make the uncalibrated Brier score 3.4 times worse.
Square-root weights were selected; the choice over no weighting is a tie-break, not a
demonstrated gain.

## Drift monitoring (July-September)

| | June | July | August | September |
|---|---|---|---|---|
| Status | baseline | warn | alert | alert |
| `amount` PSI / KS D | | 0.001 / 0.010 | 0.062 / 0.091 | 0.081 / 0.107 |
| `issuer_bank` missing | 1.0% | 1.0% | 1.0% | 9.9% |
| Share of payments flagged | 1.59% | 1.64% | 1.96% | 2.47% |
| PR-AUC | 0.752 | 0.700 | 0.474 | 0.417 |
| Precision / recall | 0.533 / 0.762 | 0.534 / 0.719 | 0.424 / 0.472 | 0.383 / 0.435 |
| False declines | 11 | 30 | 23 | 80 |

A fraud pattern that starts in mid-July (small UPI scam payments) is caught only 10-13% of
the time, which drives the collapse. A challenger trained on data to the end of July
improves September PR-AUC from 0.404 to 0.443 with fewer false declines, but its fraud loss
by value is slightly higher and it remains far from the June baseline. Closing that gap needs new signals, not only retraining. The full
table, alert rules, actions and rollback plan are in [`docs/MONITORING.md`](docs/MONITORING.md).

## Production checks

| Check | Where |
|---|---|
| Input schema and data types | `app/controllers/validation/schema.py`; run before training and on every API request |
| Missing-value and range checks | Same validator: required fields, amount and MCC ranges, per-column missing-rate limits |
| Training-serving consistency | One `build_features` implementation for both; an integration test replays stored transactions through the API and requires the batch score |
| Version, evaluation, rollback | `models/model_card.json`; rollback steps in `docs/MONITORING.md` |
| Tests | `tests/`: features checked against an independent brute-force implementation, no look-ahead, no label leakage, validator, drift statistics, metrics, API |

## Application

- **API** (`app/main.py`, FastAPI): scoring, review queue, explanations, model card, monitoring report.
- **Console** (`frontend/`, Next.js): overview, score a transaction, review queue, monitoring.
- **Observability**: JSON logs with a trace id per request, Prometheus metrics at `/metrics`,
  and an append-only prediction log.
- **Explanations**: each score comes with its main contributing features. By default they
  are worded by templates. Optionally Claude words them (`EXPLANATIONS_LLM=true` plus
  Anthropic credentials); the numbers still come from the model. Only the template path was
  verified for this submission; the LLM call was not run against the live API.

## Repository layout

```
app/
  controllers/   data_generation, features, training, evaluation, validation,
                 monitoring, explanations, scoring, health_check
  utils/         config, logger, metrics
  main.py        FastAPI app
frontend/        Next.js analyst console
notebooks/       executed assignment notebook
tests/           unit and integration tests
scripts/         pipeline runner, notebook builder
data/  models/  reports/  docs/
```

## Limitations

- **Synthetic data.** The fraud patterns were written by the same person who built the
  features, so the model has an unrealistic advantage. Real fraud is noisier and adversarial.
- **Selective labels.** In production, blocked payments never get a fraud label. The
  monitoring here assumes every transaction is eventually labelled.
- **Label delay** is a fixed 7 days. Real chargebacks arrive over weeks.
- **Small search.** Nine boosting configurations and no cross-validated tuning; selection
  used one validation month, and differences between the top candidates are within noise.
- **Explanations** use one-feature-at-a-time occlusion, which ignores interactions.
- **Serving** keeps history in memory and the API has no authentication. It is a
  demonstration, not a deployable service.
- **Illustrative costs.** The review cost, false-decline cost and analyst catch rate are
  assumptions.

## Next steps

1. Validate on a real or public labelled data set, and replace the single validation month
   with prequential (rolling-origin) validation.
2. Add signals for the pattern the model misses: payee and beneficiary history,
   collect-request flags, device-to-account graph features.
3. Choose thresholds from real costs per transaction amount rather than precision floors.
4. Handle selective labels with a small randomised hold-out of blocked traffic.
5. Move feature history to an online feature store; schedule retraining with champion and
   challenger in shadow.

## Use of AI tools

The brief allows AI tools. This submission was built with Claude Code (planning, code, tests,
notebook and documentation), with the approach informed by the references in `docs/ARCH.md`.
