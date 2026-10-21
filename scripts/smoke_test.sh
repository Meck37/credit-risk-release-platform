#!/usr/bin/env bash
# Quick checks through the environment's gateway, exactly as a client would call it:
#   1. the service reports ready
#   2. a valid application is scored with a sane answer
#   3. an invalid application is rejected (input validation is switched on)
source "$(dirname "$0")/lib.sh"
ENV_NAME="${1:?usage: smoke_test.sh <env>}"
load_env "$ENV_NAME"
BASE="http://127.0.0.1:${GATEWAY_PORT}"
SAMPLE="$REPO_ROOT/tests/fixtures/sample_application.json"

for _ in $(seq 1 20); do   # nginx + API may need a few seconds after a reload
  curl -fsS "$BASE/health/ready" >/dev/null 2>&1 && break
  sleep 2
done

ready="$(curl -fsS "$BASE/health/ready")" || { log "FAIL ready"; exit 1; }
log "ready: $ready"

response="$(curl -fsS -X POST "$BASE/v1/score" -H 'content-type: application/json' -d @"$SAMPLE")" \
  || { log "FAIL score request"; exit 1; }
"$PYTHON" - "$response" <<'PY' || { log "FAIL response check"; exit 1; }
import json, sys
body = json.loads(sys.argv[1])
assert 0.0 <= body["probability_of_default"] <= 1.0, body
assert body["risk_band"] in "ABCDE" and body["decision"] in {"approve", "review", "decline"}, body
print(f"scored: PD={body['probability_of_default']} band={body['risk_band']} model=v{body['model_version']}")
PY

code="$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/v1/score" \
  -H 'content-type: application/json' -d '{"application_id":"bad","AMT_INCOME_TOTAL":-1}')"
[[ "$code" == "422" ]] || { log "FAIL invalid input returned $code, expected 422"; exit 1; }
log "smoke test passed for $ENV_NAME"