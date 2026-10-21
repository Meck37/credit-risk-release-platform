"""Release gate: may this candidate model be released?

The candidate is scored on the frozen hold-out set and compared with the
current champion (the model in production). Every check is written to a JSON
report that is kept as release evidence, and the result is stamped on the
model version in the registry.

Exit code 0 = passed, 1 = failed (so the CI job stops the release).

Usage:
    python release/validate_model.py --candidate 3 --holdout data/prepared/holdout.parquet
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from scoring.config import (
    GATE_MAX_AUC_DROP_VS_CHAMPION,
    GATE_MAX_P95_LATENCY_MS,
    GATE_MIN_AUC,
    GATE_MIN_SLICE_AUC,
    MODEL_NAME,
)
from scoring.evaluation import evaluate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_version(version: str):
    return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{version}")


def champion_version(client: MlflowClient) -> str | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, "champion").version
    except MlflowException:
        return None  # first ever release: nothing to compare against


def run_checks(candidate: dict, champion: dict | None, holdout_hash_ok: bool) -> list[dict]:
    checks = [
        {"check": "holdout_file_matches_training_record", "passed": holdout_hash_ok},
        {"check": "no_nan_predictions", "passed": candidate["nan_scores"] == 0,
         "value": candidate["nan_scores"]},
        {"check": "min_auc", "passed": candidate["auc"] >= GATE_MIN_AUC,
         "value": round(candidate["auc"], 4), "threshold": GATE_MIN_AUC},
        {"check": "p95_single_row_latency_ms",
         "passed": candidate["p95_latency_ms"] <= GATE_MAX_P95_LATENCY_MS,
         "value": round(candidate["p95_latency_ms"], 2), "threshold": GATE_MAX_P95_LATENCY_MS},
    ]
    for name, result in candidate["slices"].items():
        checks.append({"check": f"slice_auc[{name}]",
                       "passed": result["auc"] >= GATE_MIN_SLICE_AUC,
                       "value": round(result["auc"], 4), "rows": result["rows"],
                       "threshold": GATE_MIN_SLICE_AUC})
    if champion is not None:
        drop = champion["auc"] - candidate["auc"]
        checks.append({"check": "not_worse_than_champion",
                       "passed": drop <= GATE_MAX_AUC_DROP_VS_CHAMPION,
                       "value": round(-drop, 4),
                       "threshold": -GATE_MAX_AUC_DROP_VS_CHAMPION})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--holdout", default="data/prepared/holdout.parquet")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    client = MlflowClient()
    holdout_path = Path(args.holdout)
    holdout = pd.read_parquet(holdout_path)
    cand_mv = client.get_model_version(MODEL_NAME, args.candidate)
    holdout_hash_ok = cand_mv.tags.get("holdout_sha256") == sha256(holdout_path)

    candidate_metrics = evaluate(load_version(args.candidate), holdout)
    champ = champion_version(client)
    champion_metrics = None
    if champ is not None and champ != str(args.candidate):
        champion_metrics = evaluate(load_version(champ), holdout)

    checks = run_checks(candidate_metrics, champion_metrics, holdout_hash_ok)
    passed = all(c["passed"] for c in checks)

    report = {
        "model_name": MODEL_NAME,
        "candidate_version": str(args.candidate),
        "champion_version": champ,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "holdout_sha256": sha256(holdout_path),
        "result": "passed" if passed else "failed",
        "checks": checks,
        "candidate_metrics": {k: v for k, v in candidate_metrics.items() if k != "slices"},
    }
    out = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / f"validation_v{args.candidate}.json"
    report_path.write_text(json.dumps(report, indent=2))

    client.set_model_version_tag(MODEL_NAME, args.candidate, "validation_status", report["result"])
    client.set_model_version_tag(MODEL_NAME, args.candidate, "validated_at",
                                 report["evaluated_at"])

    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']} {c.get('value', '')}")
    print(f"Gate result: {report['result'].upper()}  (report: {report_path})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())