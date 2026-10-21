"""Train a candidate model, log everything to MLflow and register it.

What gets recorded (this is the model's audit trail):
  - parameters, metrics, slice metrics
  - the data manifest (hashes of the exact train and hold-out files)
  - the git commit the code came from
  - the model itself, with an input signature, as a new registry version

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    python training/train.py --data-dir data/prepared
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from scoring.config import MODEL_NAME, RANDOM_SEED, RAW_INPUTS, SKOPS_TRUSTED_TYPES, TARGET
from scoring.evaluation import evaluate
from scoring.model_factory import make_pipeline


def git_commit() -> str:
    if os.getenv("GIT_SHA"):
        return os.environ["GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/prepared")
    parser.add_argument("--experiment", default="hc-credit-risk")
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = pd.read_parquet(data_dir / "train.parquet")
    holdout = pd.read_parquet(data_dir / "holdout.parquet")
    manifest = json.loads((data_dir / "manifest.json").read_text())

    params = {
        "learning_rate": args.learning_rate,
        "max_iter": args.max_iter,
        "max_leaf_nodes": args.max_leaf_nodes,
    }

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        model = make_pipeline(**params)
        model.fit(train[RAW_INPUTS], train[TARGET])
        metrics = evaluate(model, holdout)

        mlflow.log_params({**params, "random_seed": RANDOM_SEED, "train_rows": len(train)})
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
        mlflow.log_dict(metrics["slices"], "evaluation/slices.json")
        mlflow.log_dict(manifest, "data/manifest.json")
        mlflow.set_tags({"git_sha": git_commit(), "holdout_sha256": manifest["holdout_sha256"]})

        example = holdout[RAW_INPUTS].head(5)
        signature = infer_signature(example, model.predict_proba(example)[:, 1])
        info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=example,
            registered_model_name=MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )

        client = MlflowClient()
        version = info.registered_model_version
        tags = {
            "status": "candidate",
            "git_sha": git_commit(),
            "holdout_sha256": manifest["holdout_sha256"],
            "holdout_auc": f"{metrics['auc']:.4f}",
            "baseline_mean_score": f"{metrics['mean_score']:.5f}",
        }
        for key, value in tags.items():
            client.set_model_version_tag(MODEL_NAME, version, key, value)

        print(json.dumps({"run_id": run.info.run_id, "model": MODEL_NAME,
                          "version": version, "auc": round(metrics["auc"], 4)}, indent=2))


if __name__ == "__main__":
    main()