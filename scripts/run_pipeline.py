"""Run the whole pipeline: generate data -> train and evaluate -> monitor the live period.

python scripts/run_pipeline.py                # all stages
python scripts/run_pipeline.py --skip-data    # reuse data/synthetic_transactions.csv.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.controllers.data_generation.generator import generate  # noqa: E402
from app.controllers.features.builder import load_transactions  # noqa: E402
from app.controllers.monitoring.report import monitor  # noqa: E402
from app.controllers.training.trainer import train  # noqa: E402
from app.controllers.validation.schema import validate  # noqa: E402
from app.utils import config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-data", action="store_true", help="reuse the existing synthetic data file")
    args = parser.parse_args()

    if not args.skip_data or not config.SYNTHETIC_CSV.exists():
        config.DATA_DIR.mkdir(exist_ok=True)
        generate().to_csv(config.SYNTHETIC_CSV, index=False)
        print(f"[1/3] data written to {config.SYNTHETIC_CSV.relative_to(config.ROOT)}")
    df = load_transactions()

    dev = df[df["request_time"] < config.LIVE_START].drop(columns=["fraud_pattern"], errors="ignore")
    check = validate(dev)
    if not check.ok:
        print("input validation failed:", *check.errors, sep="\n  ")
        return 1

    out = train(df)
    r = out["report"]
    print("[2/3] trained. June hold-out:")
    for who in ("baseline", "champion"):
        t = r[who]["test"]
        print(
            f"   {who:9s} {r[who]['name']:42s} PR-AUC {t['pr_auc']:.3f}  ROC-AUC {t['roc_auc']:.3f}  "
            f"precision {t['precision']:.3f}  recall {t['recall']:.3f}  F1 {t['f1']:.3f}"
        )
    b = r["champion"]["test_block_band"]
    print(f"   block band: precision {b['precision']:.3f}, recall {b['recall']:.3f}, false declines {b['fp']}")

    m = monitor(df, out["champion"])
    print("[3/3] monitoring, live period:")
    for month in m["months"]:
        p = month["performance"]
        print(
            f"   {month['month']} [{month['status']:5s}] score PSI {month['prediction_drift']['score_psi']:.3f}  "
            f"PR-AUC {p['pr_auc']:.3f}  precision {p['precision']:.3f}  recall {p['recall']:.3f}"
        )
    print("reports: reports/training_report.json, reports/monitoring_report.json, models/model_card.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
