"""Compare the canary with the stable version using live Prometheus metrics.

While the canary receives ~10% of production traffic, this script asks
Prometheus three questions for each slot (stable vs canary):
  1. error rate      - share of /v1/score calls that returned 5xx
  2. p95 latency     - 95th percentile response time of /v1/score
  3. mean prediction - average predicted PD (a big jump hints at a model problem)

The canary passes only if it is not meaningfully worse than stable.
Too little canary traffic is treated as a FAIL (we never promote blind).

Exit code 0 = promote, 1 = roll back.

Usage:
    python release/canary_analysis.py --prometheus http://localhost:9090 --env prod --window 5m
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import requests

MIN_CANARY_REQUESTS = 50
MAX_ERROR_RATE_ABSOLUTE = 0.01      # canary may have at most 1% errors ...
MAX_ERROR_RATE_INCREASE = 0.005     # ... and at most +0.5 points above stable
MAX_P95_RATIO = 1.5                 # canary p95 may be at most 1.5x stable ...
P95_ALLOWANCE_SECONDS = 0.02        # ... plus 20 ms (avoids failing on tiny numbers)
MAX_MEAN_SCORE_SHIFT = 0.5          # canary mean PD within +/-50% of stable


def prometheus_query(base_url: str) -> Callable[[str], dict[str, float]]:
    """Return a function that runs a PromQL query and maps slot -> value."""

    def query(promql: str) -> dict[str, float]:
        response = requests.get(f"{base_url}/api/v1/query", params={"query": promql}, timeout=10)
        response.raise_for_status()
        result = response.json()["data"]["result"]
        return {item["metric"].get("slot", "?"): float(item["value"][1]) for item in result}

    return query


def gather(query: Callable[[str], dict[str, float]], env: str, window: str) -> dict:
    sel = f'env="{env}",handler="/v1/score"'
    return {
        "requests": query(f"sum by (slot) (increase(http_requests_total{{{sel}}}[{window}]))"),
        "errors": query(
            f'sum by (slot) (increase(http_requests_total{{{sel},status="5xx"}}[{window}]))'
        ),
        "p95": query(
            "histogram_quantile(0.95, sum by (slot, le) "
            f"(rate(http_request_duration_seconds_bucket{{{sel}}}[{window}])))"
        ),
        "mean_score": query(
            f'sum by (slot) (rate(scoring_prediction_score_sum{{env="{env}"}}[{window}])) / '
            f'sum by (slot) (rate(scoring_prediction_score_count{{env="{env}"}}[{window}]))'
        ),
    }


def decide(data: dict) -> tuple[bool, list[dict]]:
    def get(metric: str, slot: str) -> float:
        return data[metric].get(slot, 0.0)

    canary_requests = get("requests", "canary")
    stable_requests = get("requests", "stable")
    canary_err = get("errors", "canary") / canary_requests if canary_requests else 1.0
    stable_err = get("errors", "stable") / stable_requests if stable_requests else 0.0
    canary_p95, stable_p95 = get("p95", "canary"), get("p95", "stable")
    canary_mean, stable_mean = get("mean_score", "canary"), get("mean_score", "stable")
    shift = abs(canary_mean - stable_mean) / stable_mean if stable_mean else 0.0

    checks = [
        {"check": "enough_canary_traffic", "value": canary_requests,
         "passed": canary_requests >= MIN_CANARY_REQUESTS},
        {"check": "canary_error_rate", "value": round(canary_err, 5),
         "stable": round(stable_err, 5),
         "passed": canary_err <= MAX_ERROR_RATE_ABSOLUTE
         and canary_err <= stable_err + MAX_ERROR_RATE_INCREASE},
        {"check": "canary_p95_latency_seconds", "value": round(canary_p95, 4),
         "stable": round(stable_p95, 4),
         "passed": canary_p95 <= stable_p95 * MAX_P95_RATIO + P95_ALLOWANCE_SECONDS},
        {"check": "mean_score_shift", "value": round(shift, 4),
         "passed": shift <= MAX_MEAN_SCORE_SHIFT},
    ]
    return all(c["passed"] for c in checks), checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--env", default="prod")
    parser.add_argument("--window", default="5m")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args()

    data = gather(prometheus_query(args.prometheus), args.env, args.window)
    passed, checks = decide(data)
    report = {"evaluated_at": datetime.now(timezone.utc).isoformat(), "env": args.env,
              "window": args.window, "result": "promote" if passed else "rollback",
              "checks": checks, "raw": data}
    Path(args.report_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.report_dir) / "canary_analysis.json").write_text(json.dumps(report, indent=2))
    for c in checks:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['check']}: {c['value']}")
    print(f"Canary decision: {report['result'].upper()}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
