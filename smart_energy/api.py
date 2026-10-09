"""REST API over the collected data and the ML models.

    uvicorn smart_energy.api:app --reload      ->  http://localhost:8000/docs
"""
import json
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from . import config, db
from .health import device_health
from .ml.features import hourly_energy

FORECAST_PATH = config.MODELS_DIR / "forecast.joblib"
METRICS_PATH = config.MODELS_DIR / "metrics.json"

app = FastAPI(title="Smart Energy AI", version="1.0",
              description="IoT energy monitoring: readings, anomalies and consumption forecasts")
_forecaster = {"mtime": None, "model": None}


def _records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records"))


def _get_forecaster():
    if not FORECAST_PATH.exists():
        raise HTTPException(503, "Forecast model not trained yet: run python -m smart_energy.ml.train")
    mtime = FORECAST_PATH.stat().st_mtime
    if mtime != _forecaster["mtime"]:
        _forecaster.update(model=joblib.load(FORECAST_PATH), mtime=mtime)
    return _forecaster["model"]


@app.get("/health")
def health():
    with db.session() as conn:
        last = conn.execute("SELECT MAX(ts) FROM readings").fetchone()[0]
    return {"status": "ok", "last_reading_ts": last, "lag_s": int(time.time() - last) if last else None}


@app.get("/devices")
def devices():
    with db.session() as conn:
        df = db.list_devices(conn)
    df["online"] = (time.time() - df["ts"]) <= config.OFFLINE_AFTER_S
    return _records(df)


@app.get("/health/devices")
def devices_health(hours: float = Query(24, gt=0, le=24 * 7)):
    """Per-device sensor health: online status, data completeness, missing values, sensor-fault events."""
    now = int(time.time())
    with db.session() as conn:
        ids = [r[0] for r in conn.execute("SELECT DISTINCT device_id FROM readings ORDER BY device_id")]
        return [device_health(conn, d, now, int(hours * 3600)) for d in ids]


@app.get("/readings/{device_id}")
def readings(device_id: str, hours: float = Query(6, gt=0, le=24 * 90),
             max_points: int = Query(2000, ge=10, le=20000)):
    with db.session() as conn:
        df = db.load_readings(conn, device_id, since=int(time.time() - hours * 3600))
    if len(df) > max_points:  # downsample for charts
        df = df.iloc[:: len(df) // max_points + 1]
    return _records(df)


@app.get("/consumption/{device_id}")
def consumption(device_id: str, days: int = Query(7, ge=1, le=90)):
    """Hourly consumption (kWh)."""
    with db.session() as conn:
        df = db.load_readings(conn, device_id, since=int(time.time() - days * 86400))
    if df.empty:
        raise HTTPException(404, f"No data for {device_id}")
    h = hourly_energy(df).dropna()
    return [{"hour_utc": t.isoformat(), "kwh": round(v, 4)} for t, v in h.items()]


@app.get("/anomalies")
def anomalies(device_id: str | None = None, hours: float = Query(24 * 7, gt=0), limit: int = 200):
    with db.session() as conn:
        df = db.load_anomalies(conn, device_id, since=int(time.time() - hours * 3600), limit=limit)
    df["duration_s"] = df["end_ts"] - df["ts"]
    return _records(df)


@app.get("/forecast/{device_id}")
def forecast(device_id: str, horizon: int = Query(24, ge=1, le=168)):
    model = _get_forecaster()
    if device_id not in model.models:
        raise HTTPException(404, f"No forecast model for {device_id} (needs >= 3 weeks of data, then retrain)")
    with db.session() as conn:  # 3 weeks of history covers all lag features (max lag = 14 days)
        df = db.load_readings(conn, device_id, since=int(time.time() - 21 * 86400))
    fc = model.predict(device_id, df, horizon)
    return {
        "device_id": device_id,
        "total_kwh": round(float(fc["kwh"].sum()), 3),
        "hours": [{"hour_utc": t.isoformat(), **row} for t, row in fc.to_dict("index").items()],
    }


@app.get("/metrics")
def metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(503, "Models not trained yet")
    return json.loads(METRICS_PATH.read_text())
