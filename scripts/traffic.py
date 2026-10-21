"""Send realistic scoring requests to a gateway (canary observation and load tests).

Payloads are generated from the synthetic fixture generator, so no real
customer data is needed. A small share of deliberately invalid requests
(5%) checks that validation keeps working under load.

Usage:
    python scripts/traffic.py --url http://127.0.0.1:8082 --duration 60 --rps 5
    python scripts/traffic.py --url http://127.0.0.1:8082 --duration 60 --rps 10 \
        --max-p95-ms 250 --max-error-rate 0.01      # load test with pass/fail limits
"""

import argparse
import math
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from scoring.config import RAW_INPUTS
from training.make_fixture import make_fixture


def build_payloads(n: int) -> list[dict]:
    df = make_fixture(rows=n, seed=int(time.time()) % 10_000)
    payloads = []
    for i, row in enumerate(df[RAW_INPUTS].to_dict(orient="records")):
        clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
        clean["DAYS_BIRTH"] = int(clean["DAYS_BIRTH"])
        clean["DAYS_EMPLOYED"] = int(clean["DAYS_EMPLOYED"])
        clean["DAYS_ID_PUBLISH"] = int(clean["DAYS_ID_PUBLISH"])
        clean["CNT_CHILDREN"] = int(clean["CNT_CHILDREN"])
        clean["REGION_RATING_CLIENT"] = int(clean["REGION_RATING_CLIENT"])
        clean["application_id"] = f"LOAD-{i:06d}"
        payloads.append(clean)
    return payloads


def call(url: str, payload: dict) -> tuple[int, float]:
    start = time.perf_counter()
    try:
        status = requests.post(f"{url}/v1/score", json=payload, timeout=10).status_code
    except requests.RequestException:
        status = 599
    return status, (time.perf_counter() - start) * 1000


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--rps", type=float, default=5)
    parser.add_argument("--max-p95-ms", type=float)
    parser.add_argument("--max-error-rate", type=float)
    args = parser.parse_args()

    total = int(args.duration * args.rps)
    payloads = build_payloads(total)
    for p in random.sample(payloads, k=max(1, total // 20)):
        p["EXT_SOURCE_2"] = 5.0   # invalid on purpose: expect 422

    results = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = []
        for payload in payloads:
            futures.append(pool.submit(call, args.url, payload))
            time.sleep(1 / args.rps)
        results = [f.result() for f in futures]

    statuses = [s for s, _ in results]
    latencies = sorted(ms for s, ms in results if s == 200)
    server_errors = sum(1 for s in statuses if s >= 500)
    error_rate = server_errors / len(statuses)
    p95 = latencies[int(0.95 * (len(latencies) - 1))] if latencies else float("inf")
    print(f"requests={len(statuses)} ok={statuses.count(200)} rejected_422={statuses.count(422)} "
          f"server_errors={server_errors} error_rate={error_rate:.4f} p95_ms={p95:.1f}")

    failed = (args.max_p95_ms is not None and p95 > args.max_p95_ms) or (
        args.max_error_rate is not None and error_rate > args.max_error_rate)
    if failed:
        print("LOAD TEST FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())