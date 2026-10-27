#!/usr/bin/env bash
# Production canary release with automatic promotion or rollback.
#
#   scripts/canary.sh <image> <model_version>
#
# 1. start the new version next to the current one (slot "canary")
# 2. send CANARY_WEIGHT% of production traffic to it (nginx weights)
# 3. generate traffic for CANARY_SECONDS while Prometheus records both slots
# 4. release/canary_analysis.py compares canary vs stable
# 5a. PASS -> shift all traffic to canary, replace stable with the new image, remove canary
# 5b. FAIL -> send all traffic back to stable, remove canary, exit 1
# Users never see downtime: at every moment at least one healthy slot serves traffic.
source "$(dirname "$0")/lib.sh"
IMAGE="${1:?usage: canary.sh <image> <model_version>}"
MODEL_VERSION="${2:?usage: canary.sh <image> <model_version>}"
load_env prod
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found locally"
[[ -n "$(state_get prod current_image)" ]] || die "prod has no stable version yet: use deploy.sh for the very first release"
START=$(date +%s)

abort_canary() {
  log "Rolling back canary: all traffic to stable"
  write_nginx prod stable
  reload_gateway prod
  compose prod --profile canary rm -sf canary
  refresh_targets
  record env=prod action=canary image="$IMAGE" model_version="$MODEL_VERSION" \
    result=rolled_back duration_s=$(( $(date +%s) - START ))
}

log "Starting canary $IMAGE"
CANARY_IMAGE="$IMAGE" STABLE_IMAGE="$(state_get prod current_image)" \
  compose prod --profile canary up -d --wait --wait-timeout 120 canary || { abort_canary; die "canary unhealthy"; }
write_nginx prod split
reload_gateway prod
refresh_targets

log "Observing canary for ${CANARY_SECONDS}s at ${CANARY_WEIGHT}% traffic"
"$PYTHON" "$REPO_ROOT/scripts/traffic.py" --url "http://127.0.0.1:${GATEWAY_PORT}" \
  --duration "$CANARY_SECONDS" --rps 5 || true
sleep 20   # let Prometheus scrape the final numbers

if "$PYTHON" "$REPO_ROOT/release/canary_analysis.py" --env prod --window "$CANARY_WINDOW" \
     --report-dir "$OPS_HOME/records/canary-$(date -u +%Y%m%dT%H%M%S)"; then
  log "Canary passed - promoting"
  PREVIOUS_IMAGE="$(state_get prod current_image)"
  PREVIOUS_VERSION="$(state_get prod current_model_version)"
  write_nginx prod canary-only; reload_gateway prod             # all traffic on the new version
  STABLE_IMAGE="$IMAGE" compose prod up -d --wait --wait-timeout 120 stable   # replace stable
  write_nginx prod stable; reload_gateway prod                  # traffic back to (new) stable
  compose prod --profile canary rm -sf canary
  refresh_targets
  "$REPO_ROOT/scripts/smoke_test.sh" prod
  state_set prod current_image="$IMAGE" previous_image="$PREVIOUS_IMAGE" \
    current_model_version="$MODEL_VERSION" previous_model_version="$PREVIOUS_VERSION"
  record env=prod action=canary image="$IMAGE" model_version="$MODEL_VERSION" \
    result=promoted duration_s=$(( $(date +%s) - START ))
else
  abort_canary
  die "canary failed analysis - production unchanged"
fi
