"""Create the MLflow service accounts and grant least-privilege permissions.

Run once, as the MLflow admin, after the platform is up.

  svc-trainer     can create experiments, runs and new model versions
  svc-release     can READ the training experiment and EDIT the registered
                  model (set tags / aliases) - but cannot train or log runs

Passwords are read from files in deploy/platform/secrets/ (never typed into
code, never committed).

Usage:
    export MLFLOW_TRACKING_URI=http://localhost:5000
    export MLFLOW_TRACKING_USERNAME=admin
    export MLFLOW_TRACKING_PASSWORD="$(cat deploy/platform/secrets/mlflow_admin_password.txt)"
    python scripts/bootstrap_mlflow_access.py --step users
    # ... train the first model as svc-trainer (it then owns the experiment and model) ...
    python scripts/bootstrap_mlflow_access.py --step grants
"""

import argparse
import os
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.server import get_app_client

from scoring.config import MODEL_NAME

SECRETS = Path("deploy/platform/secrets")
EXPERIMENT = "hc-credit-risk"
USERS = ["svc-trainer", "svc-release"]


def read_secret(name: str) -> str:
    return (SECRETS / f"{name}_password.txt").read_text().strip()


def create_users(auth) -> None:
    for user in USERS:
        try:
            auth.create_user(username=user, password=read_secret(user))
            print(f"created user {user}")
        except MlflowException as exc:
            if "already exists" in str(exc).lower() or "RESOURCE_ALREADY_EXISTS" in str(exc):
                print(f"user {user} already exists")
            else:
                raise


def grant(auth) -> None:
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(f"Experiment '{EXPERIMENT}' not found: train the first model first.")
    grants = [
        ("experiment", experiment.experiment_id, "READ"),
        ("registered_model", MODEL_NAME, "EDIT"),
    ]
    for resource_type, resource_id, permission in grants:
        auth.grant_user_permission("svc-release", resource_type, resource_id, permission)
        print(f"granted {permission} on {resource_type} {resource_id} to svc-release")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["users", "grants"], required=True)
    args = parser.parse_args()
    auth = get_app_client("basic-auth", tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    create_users(auth) if args.step == "users" else grant(auth)


if __name__ == "__main__":
    main()