"""Prediction service for the Bhopal Clean Air Cell.

POST /predict  -> {"prediction": <pm25_next_day>}
GET  /health   -> {"status": "ok"}
"""
import json
import logging
import os
import sys
import time
from typing import Literal

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
FEATURES = ["pm25_today", "wind_kmph", "humidity_pct", "temp_c",
            "traffic_index", "stubble_fires", "day_type", "station_zone"]


class JsonFormatter(logging.Formatter):
    """One JSON object per log line."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}))
        return json.dumps(entry)


class _StdoutHandler(logging.StreamHandler):
    """Writes to whatever sys.stdout is *now* (screen only, never a file)."""

    def __init__(self):
        super().__init__(sys.stdout)

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


def _build_logger() -> logging.Logger:
    log = logging.getLogger("mp097")
    if not log.handlers:
        handler = _StdoutHandler()  # screen only, never a file
        handler.setFormatter(JsonFormatter())
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    return log


logger = _build_logger()


class PredictRequest(BaseModel):
    pm25_today: float = Field(..., description="Today's PM2.5 (µg/m³)")
    # client rule 2: below 0 is a typing mistake
    wind_kmph: float = Field(..., ge=0, description="Wind speed in km/h (must be ≥ 0)")
    humidity_pct: float = Field(..., description="Relative humidity, %")
    # client rule 1: above 75 is impossible
    temp_c: float = Field(..., le=75, description="Temperature in °C (must be ≤ 75)")
    traffic_index: float = Field(..., description="Traffic index")
    stubble_fires: float = Field(..., description="Stubble fire count")
    day_type: Literal["weekday", "weekend", "festival"] = Field(..., description="Type of day")
    station_zone: Literal["Central", "Residential", "Industrial"] = Field(
        ..., description="Monitoring station zone")

    model_config = {"json_schema_extra": {"example": {
        "pm25_today": 150, "wind_kmph": 12, "humidity_pct": 55, "temp_c": 30,
        "traffic_index": 60, "stubble_fires": 120,
        "day_type": "weekday", "station_zone": "Central"}}}


class PredictResponse(BaseModel):
    prediction: float = Field(..., description="Predicted PM2.5 for tomorrow (µg/m³)")


DESCRIPTION = """
Predicts **next-day PM2.5** for Bhopal monitoring stations so schools and hospitals
get a one-day warning.

* Model: Ridge regression, registered in MLflow as `mp097-pm25-model` (`@champion`)
* Metric: MAE (µg/m³)
* Input rules: `temp_c` ≤ 75, `wind_kmph` ≥ 0
"""

app = FastAPI(
    title="Tomorrow's Air Quality - Bhopal",
    version="1.0.0",
    description=DESCRIPTION,
    contact={"name": "Group G142 - MP097"},
    openapi_tags=[
        {"name": "Monitoring", "description": "Service health"},
        {"name": "Prediction", "description": "Next-day PM2.5 forecast"},
    ],
)
_model = None


def get_model():
    """Load the champion model once, on first use."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def frontend():
    """The web page for the Clean Air Cell (calls POST /predict)."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", tags=["Monitoring"], summary="Service health check")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"],
          summary="Predict next-day PM2.5",
          responses={503: {"description": "Model not available"}})
def predict(req: PredictRequest):
    try:
        model = get_model()
    except Exception:
        logger.exception("model could not be loaded")
        raise HTTPException(status_code=503, detail="model not available")
    row = pd.DataFrame([req.model_dump()])[FEATURES]
    start = time.perf_counter()
    value = float(model.predict(row)[0])
    logger.info("prediction served", extra={"fields": {
        "event": "predict",
        "station_zone": req.station_zone,
        "day_type": req.day_type,
        "prediction": round(value, 3),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }})
    return PredictResponse(prediction=value)
