"""Tell Prometheus what to monitor, based on the containers that are running now.

Writes two files that Prometheus watches (file-based service discovery):
  scoring.json - every scoring API container, labelled with env and slot
  probes.json  - URLs that blackbox probes for availability (gateways + registry)

Files are written atomically (write to temp, then rename) so Prometheus never
reads a half-written file.
"""

import json
import os
import subprocess
from pathlib import Path

OPS_HOME = Path(os.environ.get("OPS_HOME", Path.home() / "mlops-ops"))
TARGET_DIR = OPS_HOME / "prometheus-targets"


def running(label: str) -> list[dict]:
    fmt = '{{.Names}}|{{.Label "com.scoring.env"}}|{{.Label "com.scoring.slot"}}'
    out = subprocess.run(
        ["docker", "ps", "--filter", f"label={label}=true", "--format", fmt],
        check=True, capture_output=True, text=True,
    ).stdout
    rows = []
    for line in out.splitlines():
        name, env, slot = line.split("|")
        rows.append({"name": name, "env": env, "slot": slot})
    return rows


def write_atomic(path: Path, payload) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    apis = [{"targets": [f"{c['name']}:8000"], "labels": {"env": c["env"], "slot": c["slot"]}}
            for c in running("com.scoring.app")]
    probes = [{"targets": [f"http://{g['name']}:8080/health/ready"],
               "labels": {"env": g["env"], "component": "gateway"}}
              for g in running("com.scoring.gateway")]
    probes.append({"targets": ["http://mlflow:5000/health"],
                   "labels": {"env": "platform", "component": "registry"}})
    write_atomic(TARGET_DIR / "scoring.json", apis)
    write_atomic(TARGET_DIR / "probes.json", probes)
    print(f"Prometheus targets: {len(apis)} API containers, {len(probes)} probes")


if __name__ == "__main__":
    main()