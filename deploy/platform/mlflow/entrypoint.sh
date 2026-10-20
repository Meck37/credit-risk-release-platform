#!/bin/sh
# Builds the MLflow configuration from Docker secrets at start-up, so no password is
# ever stored in an image layer, a container environment variable, a command-line
# argument or a committed file.
set -eu

DB_PASSWORD="$(cat /run/secrets/db_password)"
ADMIN_PASSWORD="$(cat /run/secrets/mlflow_admin_password)"
MLFLOW_FLASK_SERVER_SECRET_KEY="$(cat /run/secrets/mlflow_flask_secret)"
export MLFLOW_FLASK_SERVER_SECRET_KEY

# The database URI contains the password, so it is passed as an environment variable
# set only inside this process - NOT as a --backend-store-uri argument, because
# command-line arguments are visible to anyone who can list processes (ps, docker top).
export MLFLOW_BACKEND_STORE_URI="postgresql+psycopg2://mlflow:${DB_PASSWORD}@mlflow-db:5432/mlflow"

# Auth settings: new users get NO permissions until explicitly granted (deny by default).
cat > /mlflow/auth/basic_auth.ini <<EOF2
[mlflow]
default_permission = NO_PERMISSIONS
database_uri = sqlite:////mlflow/auth/basic_auth.db
admin_username = admin
admin_password = ${ADMIN_PASSWORD}
authorization_function = mlflow.server.auth:authenticate_request_basic_auth
EOF2
chmod 600 /mlflow/auth/basic_auth.ini
export MLFLOW_AUTH_CONFIG_PATH=/mlflow/auth/basic_auth.ini

exec mlflow server \
  --app-name basic-auth \
  --artifacts-destination /mlflow/artifacts \
  --host 0.0.0.0 --port 5000 --workers 2 \
  --allowed-hosts "localhost,localhost:5000,127.0.0.1,127.0.0.1:5000,mlflow,mlflow:5000"
