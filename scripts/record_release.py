"""Append one release event to the release log ($OPS_HOME/records/releases.jsonl).

One JSON object per line = an append-only audit trail of every deploy, canary
decision and rollback: who, what, when, where, which approval, which result.
GitHub Actions context (run URL, actor, commit) is added automatically.

Usage:
    python scripts/record_release.py env=prod action=deploy image=credit-scoring:v4-abc123 result=success
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

OPS_HOME = Path(os.environ.get("OPS_HOME", Path.home() / "mlops-ops"))


def main() -> None:
    event = {"ts": datetime.now(timezone.utc).isoformat()}
    for pair in sys.argv[1:]:
        key, _, value = pair.partition("=")
        event[key] = value
    if os.getenv("GITHUB_RUN_ID"):
        event.setdefault("actor", os.getenv("GITHUB_ACTOR", ""))
        event.setdefault("git_sha", os.getenv("GITHUB_SHA", ""))
        event.setdefault("run_url", f"{os.getenv('GITHUB_SERVER_URL')}/"
                                    f"{os.getenv('GITHUB_REPOSITORY')}/actions/runs/"
                                    f"{os.getenv('GITHUB_RUN_ID')}")
    log = OPS_HOME / "records" / "releases.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as handle:
        handle.write(json.dumps(event) + "\n")
    print(f"recorded: {event.get('action')} {event.get('env')} {event.get('result')}")


if __name__ == "__main__":
    main()