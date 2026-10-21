#!/usr/bin/env bash
# Roll an environment back to the image it ran before the last deploy.
#
#   scripts/rollback.sh <env> [--reason "text"]
#
# For prod it also moves the registry's "champion" alias back, so the registry
# always tells the truth about what is serving.
source "$(dirname "$0")/lib.sh"
ENV_NAME="${1:?usage: rollback.sh <env> [--reason text]}"
REASON="${3:-manual rollback}"
load_env "$ENV_NAME"

CURRENT="$(state_get "$ENV_NAME" current_image)"
TARGET="$(state_get "$ENV_NAME" previous_image)"
TARGET_VERSION="$(state_get "$ENV_NAME" previous_model_version)"
[[ -n "$TARGET" ]] || die "no previous image recorded for $ENV_NAME"
docker image inspect "$TARGET" >/dev/null 2>&1 || die "previous image $TARGET is no longer on this machine"
START=$(date +%s)

log "Rolling back $ENV_NAME: $CURRENT -> $TARGET ($REASON)"
write_nginx "$ENV_NAME" stable
STABLE_IMAGE="$TARGET" compose "$ENV_NAME" up -d --wait --wait-timeout 120 stable gateway
reload_gateway "$ENV_NAME"
refresh_targets
"$REPO_ROOT/scripts/smoke_test.sh" "$ENV_NAME"

if [[ "$ENV_NAME" == "prod" ]]; then
  "$PYTHON" "$REPO_ROOT/release/registry.py" rollback
fi
state_set "$ENV_NAME" current_image="$TARGET" previous_image="$CURRENT" \
  current_model_version="$TARGET_VERSION" \
  previous_model_version="$(state_get "$ENV_NAME" current_model_version)"
record env="$ENV_NAME" action=rollback image="$TARGET" from_image="$CURRENT" \
  reason="$REASON" result=success duration_s=$(( $(date +%s) - START ))
log "Rollback of $ENV_NAME complete in $(( $(date +%s) - START ))s"