#!/usr/bin/env bash
# One-time setup of platform secrets. Generates strong random passwords into
# deploy/platform/secrets/ (git-ignored, readable only by you) and renders the
# Alertmanager config with your Telegram chat IDs.
#
#   scripts/init_secrets.sh
#
# Re-running never overwrites an existing secret (rotate one by deleting its file first).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIR="$REPO_ROOT/deploy/platform/secrets"
mkdir -p "$DIR"
chmod 700 "$DIR"

new_secret() {  # new_secret <name>
  local f="$DIR/$1.txt"
  if [[ -f "$f" ]]; then echo "exists: $1"; return; fi
  openssl rand -base64 32 | tr -d '/+=\n' | cut -c1-32 > "$f"
  chmod 600 "$f"
  echo "created: $1"
}

for name in db_password mlflow_admin_password mlflow_flask_secret grafana_admin_password \
            svc-trainer_password svc-release_password; do
  new_secret "$name"
done

# Containers run as non-root (uid/gid 10001). Compose file secrets keep host ownership,
# so grant read access via group 10001 (no human user is in it) instead of making
# the files world-readable. Only the secrets the MLflow container receives.
for name in db_password mlflow_admin_password mlflow_flask_secret; do
  f="$DIR/$name.txt"
  if [[ "$(stat -c %g "$f")" != "10001" ]]; then
    sudo chgrp 10001 "$f"
  fi
  chmod 640 "$f"
done

if [[ ! -f "$DIR/telegram_bot_token.txt" ]]; then
  read -r -s -p "Telegram bot token (from @BotFather): " token; echo
  printf '%s' "$token" > "$DIR/telegram_bot_token.txt"
  chmod 600 "$DIR/telegram_bot_token.txt"
fi

TEMPLATE="$REPO_ROOT/deploy/platform/alertmanager/alertmanager.yml.tmpl"
RENDERED="$REPO_ROOT/deploy/platform/alertmanager/alertmanager.yml"
if [[ ! -f "$RENDERED" ]]; then
  read -r -p "Telegram chat ID for PLATFORM on-call alerts: " platform_id
  read -r -p "Telegram chat ID for MODEL-OWNER alerts: " model_id
  sed -e "s/__PLATFORM_CHAT_ID__/${platform_id}/" -e "s/__MODEL_OWNER_CHAT_ID__/${model_id}/" \
    "$TEMPLATE" > "$RENDERED"
  echo "rendered alertmanager.yml"
fi

if [[ ! -f "$REPO_ROOT/deploy/platform/.env" ]]; then
  echo "OPS_HOME=${OPS_HOME:-$HOME/mlops-ops}" > "$REPO_ROOT/deploy/platform/.env"
fi
mkdir -p "${OPS_HOME:-$HOME/mlops-ops}/prometheus-targets"
for f in scoring.json probes.json; do
  t="${OPS_HOME:-$HOME/mlops-ops}/prometheus-targets/$f"
  [[ -f "$t" ]] || echo '[]' > "$t"
done
# Grafana (uid 472, gid 0) and Alertmanager (nobody, gid 65534) read these secrets
# inside their containers. Compose file secrets keep the host owner and mode, so each
# file gets the container's group and mode 640 (verified with `id`, Step 11).
for pair in grafana_admin_password:0 telegram_bot_token:65534; do
  name="${pair%%:*}"; gid="${pair##*:}"
  f="$DIR/$name.txt"
  [[ -f "$f" ]] || continue
  if [[ "$(stat -c %g "$f")" != "$gid" ]]; then
    sudo chgrp "$gid" "$f"
  fi
  chmod 640 "$f"
done

echo "Secrets ready in $DIR"
