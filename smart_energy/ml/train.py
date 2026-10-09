"""Train both models on the database history, evaluate them and save the artifacts.

    python -m smart_energy.ml.train
"""
import json
import time

import joblib
import numpy as np
import pandas as pd

from .. import config, db
from .anomaly import AnomalyDetector, to_events
from .forecast import Forecaster

ANOMALY_PATH = config.MODELS_DIR / "anomaly.joblib"
FORECAST_PATH = config.MODELS_DIR / "forecast.joblib"
METRICS_PATH = config.MODELS_DIR / "metrics.json"


def evaluate_anomalies(detections: pd.DataFrame, labels_path) -> dict:
    """Compare detections with the simulator's ground truth (point-level and event-level)."""
    labels = pd.read_csv(labels_path)
    truth = detections[["device_id", "ts"]].merge(labels, on=["device_id", "ts"], how="left")
    y_true = truth["label"].notna().to_numpy()
    y_pred = (detections["kind"] != "").to_numpy()
    tp, fp, fn = int((y_true & y_pred).sum()), int((~y_true & y_pred).sum()), int((y_true & ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    # Event-level recall: a labelled event counts as detected if any of its readings was flagged
    per_kind = {}
    lab = truth.assign(pred=y_pred).dropna(subset=["label"]).sort_values(["device_id", "ts"])
    lab["event"] = ((lab["label"] != lab["label"].shift()) | (lab["device_id"] != lab["device_id"].shift())
                    | (lab["ts"].diff() > 300)).cumsum()
    ev = lab.groupby("event").agg(label=("label", "first"), hit=("pred", "any"))
    for kind, g in ev.groupby("label"):
        per_kind[kind] = {"events": int(len(g)), "detected": int(g["hit"].sum())}
    return {
        "point_precision": round(precision, 4),
        "point_recall": round(recall, 4),
        "point_f1": round(2 * precision * recall / max(precision + recall, 1e-9), 4),
        "event_recall": round(float(ev["hit"].mean()), 4) if len(ev) else None,
        "events_by_kind": per_kind,
    }


def main():
    t0 = time.time()
    with db.session() as conn:
        readings = db.load_readings(conn)
    if readings.empty:
        raise SystemExit("No readings in the database. Run: python -m smart_energy.simulator backfill")
    print(f"Loaded {len(readings):,} readings for {readings['device_id'].nunique()} devices")

    # ---- anomaly detection
    detector = AnomalyDetector().fit(readings)
    detections, events = [], []
    for device, g in readings.groupby("device_id"):
        g = g.sort_values("ts")
        det = detector.detect(g)
        detections.append(g[["device_id", "ts"]].join(det))
        events += to_events(device, g, det)
    detections = pd.concat(detections)
    with db.session() as conn:  # historical anomalies are recomputed with the new model
        conn.execute("DELETE FROM anomalies")
        db.insert_events(conn, events)
    print(f"[anomaly] {len(events):,} anomaly events written to the database")
    print("          " + detections.loc[detections['kind'] != '', 'kind'].value_counts().to_string()
          .replace("\n", "\n          "))

    metrics: dict = {"trained_at": int(time.time()), "n_readings": int(len(readings))}
    labels_path = config.DATA_DIR / "labels.csv"
    if labels_path.exists():
        metrics["anomaly"] = evaluate_anomalies(detections, labels_path)
        print(f"[anomaly] evaluation vs ground truth: {json.dumps(metrics['anomaly'], indent=2)}")

    # ---- forecasting
    forecaster = Forecaster().fit(readings)
    metrics["forecast"] = forecaster.metrics
    for device, m in forecaster.metrics.items():
        print(f"[forecast] {device}: model MAE={m['model']['MAE_kWh']} kWh MAPE={m['model']['MAPE_%']}% "
              f"| yesterday MAE={m['baseline_yesterday']['MAE_kWh']} "
              f"| last-week MAE={m['baseline_last_week']['MAE_kWh']} "
              f"| P10-P90 coverage={m['interval_coverage_p10_p90']}")

    joblib.dump(detector, ANOMALY_PATH)
    joblib.dump(forecaster, FORECAST_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=lambda o: o.item()
                                       if isinstance(o, np.generic) else str(o)))
    print(f"Saved models to {config.MODELS_DIR} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
