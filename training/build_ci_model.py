"""Build a small model from synthetic data into build/model (no MLflow server needed).

Used by CI on GitHub-hosted runners, which have no access to the real data or
to your registry. It lets CI build and test the real Docker image end to end.

Usage:
    python training/build_ci_model.py
"""

import json
import shutil
from pathlib import Path

import mlflow

from scoring.config import MODEL_NAME, RAW_INPUTS, SKOPS_TRUSTED_TYPES, TARGET
from scoring.model_factory import make_pipeline
from training.make_fixture import make_fixture

OUT = Path("build/model")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    data = make_fixture(rows=5000, seed=11)
    model = make_pipeline(max_iter=80)
    model.fit(data[RAW_INPUTS], data[TARGET])
    mlflow.sklearn.save_model(model, str(OUT), skops_trusted_types=SKOPS_TRUSTED_TYPES)
    meta = {"model_name": MODEL_NAME, "model_version": "ci", "run_id": "ci", "git_sha": "ci",
            "holdout_sha256": "ci", "holdout_auc": "n/a",
            "baseline_mean_score": float(model.predict_proba(data[RAW_INPUTS])[:, 1].mean()),
            "exported_at": "ci"}
    (OUT / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"CI model written to {OUT}")


if __name__ == "__main__":
    main()
