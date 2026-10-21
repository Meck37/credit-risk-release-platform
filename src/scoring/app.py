"""Credit scoring API.

Endpoints
  POST /v1/score      score one application
  GET  /health/live   is the process up?
  GET  /health/ready  is the model loaded and able to predict?
  GET  /metrics       Prometheus metrics
"""

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

from scoring.config import RAW_INPUTS, RISK_BANDS
from scoring.schema import Application, ScoreResponse

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/model"))
ENVIRONMENT = os.getenv("APP_ENV", "local")
SLOT = os.getenv("DEPLOYMENT_SLOT", "stable")

# ---- structured JSON logs: one line per event, easy to search during an incident
logger = logging.getLogger("scoring")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_event(event: str, **fields) -> None:
    logger.info(json.dumps({"ts": time.time(), "event": event, "env": ENVIRONMENT,
                            "slot": SLOT, **fields}))


# ---- model metrics (HTTP metrics come from the Instrumentator below)
PREDICTION_SCORE = Histogram(
    "scoring_prediction_score", "Predicted probability of default",
    buckets=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0],
)
MISSING_INPUTS = Counter(
    "scoring_input_missing_total", "Requests where an optional field was missing", ["field"]
)
# env and slot labels are added by Prometheus from the scrape target, not here.
MODEL_INFO = Gauge(
    "scoring_model_info", "Model currently served (value is always 1)",
    ["model_name", "model_version"],
)
BASELINE_MEAN = Gauge(
    "scoring_baseline_mean_score", "Mean predicted PD on the hold-out set at training time"
)
PREDICTION_ERRORS = Counter("scoring_prediction_errors_total", "Model failures while scoring")

STATE: dict = {}


def load_model() -> None:
    meta = json.loads((MODEL_DIR / "model_meta.json").read_text())
    STATE["model"] = mlflow.sklearn.load_model(str(MODEL_DIR))
    STATE["meta"] = meta
    MODEL_INFO.labels(meta["model_name"], meta["model_version"]).set(1)
    BASELINE_MEAN.set(meta["baseline_mean_score"])
    log_event("model_loaded", model_version=meta["model_version"], run_id=meta["run_id"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Credit Risk Scoring API", version="1.0.0", lifespan=lifespan)
# Fine-grained latency buckets: the default (0.1s, 0.5s, 1s) is too coarse to see
# a scoring call that normally takes ~20 ms getting slower.
Instrumentator(excluded_handlers=["/metrics", "/health/live"]).instrument(
    app,
    latency_lowr_buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0),
).expose(app, endpoint="/metrics", include_in_schema=False)


def to_band(pd_value: float) -> tuple[str, str]:
    for upper, band, decision in RISK_BANDS:
        if pd_value < upper:
            return band, decision
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


@app.get("/health/live")
def live() -> dict:
    return {"status": "alive"}


@app.get("/health/ready")
def ready() -> dict:
    model = STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": STATE["meta"]["model_version"],
            "env": ENVIRONMENT, "slot": SLOT}


@app.post("/v1/score", response_model=ScoreResponse)
def score(application: Application, request: Request) -> ScoreResponse:
    request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
    payload = application.model_dump()
    for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "AMT_ANNUITY"):
        if payload[field] is None:
            MISSING_INPUTS.labels(field).inc()

    frame = pd.DataFrame([{k: payload[k] for k in RAW_INPUTS}])
    try:
        probability = float(STATE["model"].predict_proba(frame)[0, 1])
    except Exception as exc:  # a model failure must be visible, not silent
        PREDICTION_ERRORS.inc()
        log_event("prediction_error", request_id=request_id, error=str(exc))
        raise HTTPException(status_code=500, detail="prediction failed") from exc

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        PREDICTION_ERRORS.inc()
        log_event("invalid_prediction", request_id=request_id, value=str(probability))
        raise HTTPException(status_code=500, detail="invalid prediction")

    PREDICTION_SCORE.observe(probability)
    band, decision = to_band(probability)
    meta = STATE["meta"]
    log_event("scored", request_id=request_id, application_id=application.application_id,
              pd=round(probability, 5), band=band, model_version=meta["model_version"])
    return ScoreResponse(
        application_id=application.application_id,
        probability_of_default=round(probability, 5),
        risk_band=band,
        decision=decision,
        model_name=meta["model_name"],
        model_version=meta["model_version"],
        request_id=request_id,
    )