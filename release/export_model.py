"""Download one approved registry version into the Docker build context.

Why bake the model into the image (instead of downloading it at start-up):
  - the image becomes the single, immutable "deployment artefact"
  - rollback is just "run the previous image tag"
  - the running API never needs credentials for the registry or storage

Usage:
    python release/export_model.py --version 3 --out build/model
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from scoring.config import MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", default="build/model")
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    client = MlflowClient()
    mv = client.get_model_version(MODEL_NAME, args.version)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{MODEL_NAME}/{args.version}", dst_path=str(out)
    )

    meta = {
        "model_name": MODEL_NAME,
        "model_version": str(mv.version),
        "run_id": mv.run_id,
        "git_sha": mv.tags.get("git_sha", "unknown"),
        "holdout_sha256": mv.tags.get("holdout_sha256", "unknown"),
        "holdout_auc": mv.tags.get("holdout_auc", "unknown"),
        "baseline_mean_score": float(mv.tags.get("baseline_mean_score", "nan")),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()