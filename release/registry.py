"""Small command-line tool for registry actions used by the release scripts.

Aliases are named pointers to a model version:
  staging   -> version currently in the staging environment
  champion  -> version currently serving production
  previous  -> the champion before the current one (the rollback target)

Usage:
    python release/registry.py promote --version 4 --approved-by meck --change-id CHG-0007
    python release/registry.py rollback
    python release/registry.py set-alias --alias staging --version 4
    python release/registry.py show
"""

import argparse
import json
from datetime import datetime, timezone

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from scoring.config import MODEL_NAME


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def alias_version(client: MlflowClient, alias: str) -> str | None:
    try:
        return client.get_model_version_by_alias(MODEL_NAME, alias).version
    except MlflowException:
        return None


def promote(client: MlflowClient, version: str, approved_by: str, change_id: str) -> dict:
    mv = client.get_model_version(MODEL_NAME, version)
    if mv.tags.get("validation_status") != "passed":
        raise SystemExit(f"Refusing to promote v{version}: release gate has not passed.")
    old = alias_version(client, "champion")
    if old and old != str(version):
        client.set_registered_model_alias(MODEL_NAME, "previous", old)
        client.set_model_version_tag(MODEL_NAME, old, "status", "superseded")
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    for key, value in {"status": "production", "approved_by": approved_by,
                       "change_id": change_id, "promoted_at": now()}.items():
        client.set_model_version_tag(MODEL_NAME, version, key, value)
    return {"action": "promote", "champion": version, "previous": old}


def rollback(client: MlflowClient) -> dict:
    current, previous = alias_version(client, "champion"), alias_version(client, "previous")
    if previous is None:
        raise SystemExit("No 'previous' alias: nothing to roll back to.")
    client.set_registered_model_alias(MODEL_NAME, "champion", previous)
    client.delete_registered_model_alias(MODEL_NAME, "previous")
    client.set_model_version_tag(MODEL_NAME, current, "status", "rolled_back")
    client.set_model_version_tag(MODEL_NAME, current, "rolled_back_at", now())
    client.set_model_version_tag(MODEL_NAME, previous, "status", "production")
    return {"action": "rollback", "champion": previous, "rolled_back": current}


def show(client: MlflowClient) -> dict:
    return {alias: alias_version(client, alias) for alias in ("staging", "champion", "previous")}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("promote")
    p.add_argument("--version", required=True)
    p.add_argument("--approved-by", required=True)
    p.add_argument("--change-id", required=True)
    sub.add_parser("rollback")
    s = sub.add_parser("set-alias")
    s.add_argument("--alias", required=True)
    s.add_argument("--version", required=True)
    sub.add_parser("show")
    args = parser.parse_args()

    client = MlflowClient()
    if args.command == "promote":
        result = promote(client, args.version, args.approved_by, args.change_id)
    elif args.command == "rollback":
        result = rollback(client)
    elif args.command == "set-alias":
        client.set_registered_model_alias(MODEL_NAME, args.alias, args.version)
        result = {"action": "set-alias", args.alias: args.version}
    else:
        result = show(client)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
