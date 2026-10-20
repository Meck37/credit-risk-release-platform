# MLflow tracking server + model registry with built-in user authentication.
# Built locally (instead of a public image) so every installed package is pinned and known.
FROM python:3.12-slim

# MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false: MLflow 3 starts a background job system
# (huey workers, ~1 GB RAM) by default. This platform does not use server-side jobs,
# so it is switched off (least functionality: don't run what you don't need).
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false

RUN pip install "mlflow[auth]==3.17.1" "psycopg2-binary==2.9.13" \
 && useradd --uid 10001 --create-home mlflow \
 && mkdir -p /mlflow/artifacts /mlflow/auth \
 && chown -R mlflow:mlflow /mlflow

COPY --chmod=755 entrypoint.sh /usr/local/bin/entrypoint.sh
USER mlflow
WORKDIR /mlflow
EXPOSE 5000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
