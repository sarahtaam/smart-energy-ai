"""MQTT -> validation -> SQLite, with real-time anomaly detection on every reading
and a watchdog for sensor nodes that go silent or send incomplete data.

    python -m smart_energy.ingestor
"""
import json
import logging
import threading
import time
from collections import defaultdict, deque

import joblib
import pandas as pd
from pydantic import ValidationError

from . import alerts, config, db
from .health import device_health
from .ml.anomaly import AnomalyDetector
from .schemas import FIELDS, Telemetry

log = logging.getLogger("ingestor")
MODEL_PATH = config.MODELS_DIR / "anomaly.joblib"


class Ingestor:
    def __init__(self, conn=None, window: int = 30):
        self.conn = conn or db.connect()
        self.buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.detector = AnomalyDetector()  # rules only until a trained model exists
        self._model_mtime = None
        self._last_model_check = 0.0
        self.stats = {"ok": 0, "invalid": 0, "anomalies": 0}
        # device -> ts of its latest reading; loaded from the DB so nodes that went silent while the
        # ingestor was stopped are still watched
        self.last_seen: dict[str, int] = dict(
            self.conn.execute("SELECT device_id, MAX(ts) FROM readings GROUP BY device_id").fetchall())
        self.maybe_reload_model(force=True)

    def maybe_reload_model(self, force=False) -> None:
        """Hot-reload the model when `train` writes a new one (checked every 60 s)."""
        if not force and time.time() - self._last_model_check < 60:
            return
        self._last_model_check = time.time()
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_mtime != self._model_mtime:
            self.detector = joblib.load(MODEL_PATH)
            self._model_mtime = MODEL_PATH.stat().st_mtime
            log.info("Loaded anomaly model for devices: %s", ", ".join(self.detector.models) or "-")
        elif force and not MODEL_PATH.exists():
            log.warning("No trained anomaly model yet: rule-based detection only")

    def _record(self, device_id: str, ts: int, kind: str, score: float, power: float, detail: str) -> dict | None:
        """Store one anomalous observation. Returns the event (and notifies it) only when it opens a new one."""
        if not db.upsert_event(self.conn, device_id, ts, kind, score, power, detail):
            return None
        event = {"device_id": device_id, "ts": ts, "kind": kind, "power": power, "detail": detail}
        log.warning("ANOMALY %-20s device=%s power=%sW %s", kind, device_id, power, detail)
        alerts.notify(event)
        return event

    def handle(self, topic: str, payload: bytes) -> dict | None:
        """Process one telemetry message. Returns the first new event it opened, if any."""
        try:
            data = json.loads(payload)
            data.setdefault("device_id", topic.split("/")[1])
            reading = Telemetry.model_validate(data).with_ts()
        except (ValueError, ValidationError, IndexError) as e:
            self.stats["invalid"] += 1
            log.warning("Rejected message on %s: %s", topic, str(e).splitlines()[0])
            return None

        row = reading.model_dump()
        db.insert_readings(self.conn, [row])
        self.stats["ok"] += 1
        self.last_seen[reading.device_id] = max(reading.ts, self.last_seen.get(reading.device_id, 0))

        buf = self.buffers[reading.device_id]
        buf.append(row)
        self.maybe_reload_model()
        df = pd.DataFrame(buf).astype({f: float for f in FIELDS})
        last = self.detector.detect(df).iloc[-1]
        events = []
        if last["kind"]:
            self.stats["anomalies"] += 1
            detail = f"z={last['z']:+.1f}" if pd.notna(last["z"]) else ""
            events.append(self._record(reading.device_id, reading.ts, last["kind"], float(last["score"]),
                                       row["power"] or 0.0, detail))
        # A null field means that sensor did not answer (e.g. DHT22 unplugged). During an outage the PZEM
        # cannot read its energy counter: that is expected and already reported as an outage.
        missing = [f for f in FIELDS if row[f] is None and not (f == "energy" and row["voltage"] == 0)]
        if missing:
            events.append(self._record(reading.device_id, reading.ts, "missing_field", 0.0, row["power"] or 0.0,
                                       ", ".join(missing) + " missing"))
        self.conn.commit()
        return next((e for e in events if e), None)

    def handle_status(self, device_id: str, status: str) -> None:
        log.info("Device %s is %s", device_id, status)
        if status == "offline":  # MQTT last-will: the broker lost the connection to the node
            self._record(device_id, int(time.time()), "device_offline", 0.0, 0.0, "connection lost (MQTT last-will)")
            self.conn.commit()

    def check_health(self, now: int | None = None) -> list[dict]:
        """Watchdog: flag nodes that stopped publishing, and nodes with incomplete data over 24 h.

        Runs every minute; while a problem lasts its event is extended, so it is notified only once.
        Returns the new events.
        """
        now = int(now or time.time())
        new = []
        for device, last_ts in self.last_seen.items():
            age = now - last_ts
            if age > config.OFFLINE_AFTER_S:
                new.append(self._record(device, now, "device_offline", 0.0, 0.0, f"no data for {age // 60} min"))
                continue
            comp = device_health(self.conn, device, now)["completeness"]
            if comp is not None and comp < config.MIN_COMPLETENESS:
                new.append(self._record(device, now, "data_gap_24h", 0.0, 0.0,
                                        f"only {comp:.0%} of the last 24 h received"))
        self.conn.commit()
        return [e for e in new if e]


def main():
    import paho.mqtt.client as mqtt

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ingestor = Ingestor()
    lock = threading.Lock()  # the MQTT thread and the watchdog share the SQLite connection
    telemetry_topic = f"{config.TOPIC_PREFIX}/+/telemetry"
    status_topic = f"{config.TOPIC_PREFIX}/+/status"

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            log.error("MQTT connection refused: %s", reason_code)
            return
        log.info("Connected to MQTT %s:%s, subscribing to %s", config.MQTT_HOST, config.MQTT_PORT, telemetry_topic)
        client.subscribe([(telemetry_topic, 1), (status_topic, 1)])

    def on_message(client, userdata, msg):
        try:
            with lock:
                if msg.topic.endswith("/status"):
                    ingestor.handle_status(msg.topic.split("/")[1], msg.payload.decode(errors="replace"))
                else:
                    ingestor.handle(msg.topic, msg.payload)
        except Exception:  # never let one bad message kill the subscriber loop
            log.exception("Error while processing message on %s", msg.topic)

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="smart-energy-ingestor")
    if config.MQTT_USER:
        client.username_pw_set(config.MQTT_USER, config.MQTT_PASSWORD)
    client.on_connect = on_connect
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    while True:
        try:
            client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=60)
            break
        except OSError as e:
            log.warning("MQTT broker %s:%s unreachable (%s), retrying in 5s", config.MQTT_HOST, config.MQTT_PORT, e)
            time.sleep(5)
    client.loop_start()  # MQTT network loop in a background thread; the watchdog runs in this one
    try:
        while True:
            time.sleep(config.WATCHDOG_INTERVAL_S)
            with lock:
                try:
                    ingestor.check_health()
                except Exception:
                    log.exception("Watchdog error")
    except KeyboardInterrupt:
        log.info("Stopped. Stats: %s", ingestor.stats)
    finally:
        client.loop_stop()


if __name__ == "__main__":
    main()
