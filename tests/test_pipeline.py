import json
import time

import numpy as np
import pandas as pd
import pytest

from smart_energy import alerts, db, health, simulator
from smart_energy.ml.anomaly import AnomalyDetector, to_events
from smart_energy.ml.features import forecast_features, hour_of_week
from smart_energy.ml.forecast import Forecaster


@pytest.fixture(autouse=True)
def sent_alerts(monkeypatch):
    """Capture notifications instead of sending them, even when Telegram/SMTP are configured."""
    sent = []
    monkeypatch.setattr(alerts, "notify", sent.append)
    return sent


@pytest.fixture(scope="module")
def history():
    """42 days of clean 10-minute data for two devices, ending now."""
    rng = np.random.default_rng(0)
    end = int(time.time()) // 600 * 600
    ts = np.arange(end - 42 * 86400, end, 600)
    return pd.concat([simulator.simulate(d, ts, rng) for d in ["lab_a", "office_b"]], ignore_index=True)


@pytest.fixture(scope="module")
def detector(history):
    return AnomalyDetector().fit(history)


def _device(history, name):
    return history[history["device_id"] == name].sort_values("ts").reset_index(drop=True)


def test_clean_data_has_few_false_alarms(history, detector):
    det = detector.detect(_device(history, "lab_a"))
    assert (det["kind"] != "").mean() < 0.01


def _outage(df, i):
    df.loc[i, ["voltage", "current", "power", "pf", "frequency"]] = 0.0


def _sag(df, i):
    df.loc[i, "voltage"] = 180.0


def _spike(df, i):
    df.loc[i, "power"] *= 5
    df.loc[i, "current"] = df.loc[i, "power"] / (df.loc[i, "voltage"] * df.loc[i, "pf"])


@pytest.mark.parametrize("mutate, expected", [
    (_outage, "outage"), (_sag, "voltage_sag"), (_spike, "high_consumption"),
])
def test_injected_anomalies_are_detected(history, detector, mutate, expected):
    df = _device(history, "lab_a").copy()
    i = df.index[-50:-45]
    mutate(df, i)
    det = detector.detect(df)
    assert (det.loc[i, "kind"] == expected).all(), det.loc[i, "kind"].tolist()


def test_frozen_sensor_detected(history, detector):
    df = _device(history, "office_b").copy()
    i = df.index[-40:]
    for col in ["voltage", "current", "power", "pf", "frequency", "temperature", "humidity"]:
        df.loc[i, col] = df.loc[i[0], col]
    det = detector.detect(df)
    assert (det.loc[i[-20:], "kind"] == "stuck_sensor").all()


def test_unknown_device_uses_rules_only(detector):
    df = pd.DataFrame({"device_id": "new_node", "ts": [1_759_000_000 + 5 * k for k in range(3)],
                       "voltage": [230.0, 230.0, 0.0], "current": [1.0, 1.0, 0.0], "power": [200.0, 210, 0],
                       "pf": [0.87, 0.87, 0], "frequency": [50.0, 50, 0]})
    assert detector.detect(df)["kind"].tolist() == ["", "", "outage"]


def test_events_are_merged():
    df = pd.DataFrame({"ts": [0, 60, 120, 1000, 1060], "power": [1, 2, 3, 4, 5]})
    det = pd.DataFrame({"kind": ["spike", "spike", "", "spike", "outage"], "score": [0.1] * 5, "z": [1.0] * 5})
    events = to_events("d", df, det)
    assert [(e["kind"], e["ts"], e["end_ts"], e["n_points"]) for e in events] == \
        [("spike", 0, 60, 2), ("spike", 1000, 1000, 1), ("outage", 1060, 1060, 1)]


def test_forecast_features_never_use_last_24h():
    idx = pd.date_range("2026-01-01", periods=24 * 20, freq="1h")
    hourly = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    target = idx[-24:]
    X = forecast_features(hourly[hourly.index < target[0]], target)
    assert (X["lag_24"].to_numpy() == hourly[target - pd.Timedelta(hours=24)].to_numpy()).all()


def test_forecaster_beats_naive_baseline(history):
    fc = Forecaster().fit(history, test_days=5)
    for device, m in fc.metrics.items():
        assert m["model"]["MAE_kWh"] < m["baseline_last_week"]["MAE_kWh"], device
    pred = fc.predict("lab_a", _device(history, "lab_a"), horizon=36)
    assert len(pred) == 36 and (pred["p10"] <= pred["kwh"]).all() and (pred["kwh"] <= pred["p90"]).all()


def test_ingestor_end_to_end(tmp_path, history, detector, monkeypatch):
    from smart_energy import ingestor as ing

    monkeypatch.setattr(ing, "MODEL_PATH", tmp_path / "missing.joblib")
    conn = db.connect(tmp_path / "t.db")
    it = ing.Ingestor(conn=conn)
    it.detector = detector
    now = int(time.time())
    # a normal reading = the typical power for lab_a at this hour of the week
    power = float(detector.profiles["lab_a"]["median"].iloc[hour_of_week(pd.Series([now]))[0]])
    base = {"device_id": "lab_a", "voltage": 231.0, "current": power / (231 * 0.87), "power": power, "pf": 0.87,
            "frequency": 50.0, "energy": 1.0, "temperature": 24.0, "humidity": 50.0}
    assert it.handle("smartenergy/lab_a/telemetry", json.dumps({**base, "ts": now - 10}).encode()) is None
    event = it.handle("smartenergy/lab_a/telemetry",
                      json.dumps({**base, "ts": now - 5, "voltage": 0, "current": 0, "power": 0, "pf": 0}).encode())
    assert event and event["kind"] == "outage"
    it.handle("smartenergy/lab_a/telemetry",
              json.dumps({**base, "ts": now, "voltage": 0, "current": 0, "power": 0, "pf": 0}).encode())
    it.handle("smartenergy/lab_a/telemetry", b"not json")
    it.handle("smartenergy/lab_a/telemetry", json.dumps({**base, "voltage": 999}).encode())

    assert conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 3
    ev = db.load_anomalies(conn)
    assert len(ev) == 1 and ev.loc[0, "n_points"] == 2  # two outage readings merged into one event
    assert it.stats["invalid"] == 2


def _ingestor(tmp_path, monkeypatch, rows=()):
    from smart_energy import ingestor as ing

    monkeypatch.setattr(ing, "MODEL_PATH", tmp_path / "missing.joblib")
    conn = db.connect(tmp_path / "t.db")
    db.insert_readings(conn, rows)
    return ing.Ingestor(conn=conn)


def test_completeness_counts_long_gaps():
    ts = np.arange(0, 3600, 5)
    assert health.completeness(ts, 0, 3600) == (1.0, 5.0)
    gapped = ts[(ts < 1000) | (ts >= 1900)]  # 15 of 60 minutes missing
    assert health.completeness(gapped, 0, 3600)[0] == pytest.approx(0.75)
    # a node switching from one reading per minute to one every 5 s has no missing data
    rate_change = np.concatenate([np.arange(0, 1800, 60), np.arange(1800, 3600, 5)])
    assert health.completeness(rate_change, 0, 3600) == (1.0, 5.0)


def test_watchdog_flags_silent_and_incomplete_devices(tmp_path, monkeypatch, sent_alerts):
    now = 1_790_000_000
    ts = np.arange(now - 86400, now, 60)
    keep = {"healthy": ts, "silent": ts[ts < now - 600],
            "gappy": ts[(ts < now - 12 * 3600) | (ts >= now - 6 * 3600)]}  # 6 h missing
    it = _ingestor(tmp_path, monkeypatch,
                   [{"device_id": d, "ts": int(t), "power": 100.0} for d, tt in keep.items() for t in tt])

    events = it.check_health(now)
    assert {e["device_id"]: e["kind"] for e in events} == {"silent": "device_offline", "gappy": "data_gap_24h"}
    assert it.check_health(now + 60) == []  # same problems a minute later: events extended, not re-opened
    ev = db.load_anomalies(it.conn)
    assert len(ev) == 2 and (ev["n_points"] == 2).all()
    assert len(sent_alerts) == 2  # one notification per problem


def test_missing_sensor_values_raise_an_alert(tmp_path, monkeypatch):
    it = _ingestor(tmp_path, monkeypatch)
    msg = {"device_id": "lab_a", "voltage": 231.0, "current": 1.0, "power": 200.0, "pf": 0.87, "frequency": 50.0,
           "energy": 1.0, "temperature": None, "humidity": None}
    event = it.handle("smartenergy/lab_a/telemetry", json.dumps(msg).encode())
    assert event["kind"] == "missing_field" and event["detail"] == "temperature, humidity missing"
    # During an outage the PZEM cannot read its energy counter: reported as an outage only
    outage = {**msg, "temperature": 24.0, "humidity": 50.0, "voltage": 0, "current": 0, "power": 0, "pf": 0,
              "energy": None}
    assert it.handle("smartenergy/lab_a/telemetry", json.dumps(outage).encode())["kind"] == "outage"
    assert sorted(db.load_anomalies(it.conn)["kind"]) == ["missing_field", "outage"]


def test_last_will_opens_offline_event(tmp_path, monkeypatch):
    it = _ingestor(tmp_path, monkeypatch)
    it.handle_status("lab_a", "offline")
    it.handle_status("lab_a", "online")
    assert db.load_anomalies(it.conn)["kind"].tolist() == ["device_offline"]
