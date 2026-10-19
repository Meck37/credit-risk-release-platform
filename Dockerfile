# Scoring API image. The approved model is copied in at build time, so the image is
# the complete, immutable deployment artefact: code + model + dependencies.
FROM python:3.14-slim

ARG GIT_SHA=unknown
ARG MODEL_VERSION=unknown
LABEL org.opencontainers.image.title="credit-scoring-api" \
      org.opencontainers.image.revision="${GIT_SHA}" \
      com.scoring.model_version="${MODEL_VERSION}"

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src MODEL_DIR=/app/model HOME=/tmp \
    MLFLOW_DISABLE_TELEMETRY=true MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /app
COPY requirements-serving.txt .
RUN pip install -r requirements-serving.txt && useradd --uid 10001 --no-create-home appuser

COPY src/ src/
COPY build/model/ model/

USER appuser
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2)"
# One worker per container: Prometheus metrics live in process memory, so scale by
# adding containers, not workers.
CMD ["uvicorn", "scoring.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
