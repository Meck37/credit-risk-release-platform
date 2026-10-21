#!/usr/bin/env bash
# Deploy an image to one environment (all traffic goes to it after a passing smoke test).
# Used for dev and staging, and by canary.sh to finish a production promotion.
#
#   scripts/deploy.sh <dev|staging|prod> <image> [model_version]
#
# If the smoke test fails, the previous image is restored automatically.
source "$(dirname "$0")/lib.sh"

ENV_NAME="${1:?usage: deploy.sh <env> <image> [model_version]}"
IMAGE="${2:?usage: deploy.sh <env> <image> [model_version]}"
MODEL_VERSION="${3:-unknown}"
load_env "$ENV_NAME"

docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image $IMAGE not found locally"
PREVIOUS_IMAGE="$(state_get "$ENV_NAME" current_image)"
PREVIOUS_VERSION="$(state_get "$ENV_NAME" current_model_version)"
START=$(date +%s)

log "Deploying $IMAGE to $ENV_NAME (previous: ${PREVIOUS_IMAGE:-none})"
write_nginx "$ENV_NAME" stable
STABLE_IMAGE="$IMAGE" compose "$ENV_NAME" up -d --wait --wait-timeout 120 stable gateway \
  || die "containers did not become healthy"
reload_gateway "$ENV_NAME"
refresh_targets

if "$REPO_ROOT/scripts/smoke_test.sh" "$ENV_NAME"; then
  state_set "$ENV_NAME" current_image="$IMAGE" previous_image="$PREVIOUS_IMAGE" \
    current_model_version="$MODEL_VERSION" previous_model_version="$PREVIOUS_VERSION"
  record env="$ENV_NAME" action=deploy image="$IMAGE" model_version="$MODEL_VERSION" \
    result=success duration_s=$(( $(date +%s) - START ))
  log "Deploy to $ENV_NAME succeeded"
else
  record env="$ENV_NAME" action=deploy image="$IMAGE" model_version="$MODEL_VERSION" \
    result=failed_smoke_test duration_s=$(( $(date +%s) - START ))
  if [[ -n "$PREVIOUS_IMAGE" ]]; then
    log "Smoke test failed - restoring $PREVIOUS_IMAGE"
    STABLE_IMAGE="$PREVIOUS_IMAGE" compose "$ENV_NAME" up -d --wait --wait-timeout 120 stable
    reload_gateway "$ENV_NAME"
    record env="$ENV_NAME" action=auto_rollback image="$PREVIOUS_IMAGE" result=success
  fi
  die "deploy to $ENV_NAME failed its smoke test"
fi