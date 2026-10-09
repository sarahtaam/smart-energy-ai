"""Sensor health: is each node still publishing, and is its data complete over the last 24 h?"""
import time

import numpy as np

from . import config, db
from .schemas import FIELDS

# Event kinds that point to the sensor node itself rather than to the monitored consumption
SENSOR_FAULT_KINDS = ("device_offline", "missing_field", "data_gap_24h", "stuck_sensor", "meter_inconsistency")


def completeness(ts: np.ndarray, window_start: int, now: int,
                 max_gap_s: int = config.OFFLINE_AFTER_S) -> tuple[float, float | None]:
    """Share of [window_start, now] covered by readings, and the typical publish interval.

    A gap (including before the first and after the last reading) counts as missing time when it is
    longer than both 3 typical intervals and `max_gap_s`, i.e. when the node would have been reported
    offline. The second condition keeps a change of publish rate (e.g. 60 s -> 5 s) from looking like
    missing data.
    """
    if len(ts) < 2:
        return 0.0, None
    interval = max(float(np.median(np.diff(ts))), 1.0)
    steps = np.diff(np.concatenate([[window_start], ts, [now]]))
    missing = (steps - interval)[steps > max(3 * interval, max_gap_s)].sum()
    return float(max(0.0, 1 - missing / max(now - window_start, 1))), interval


def device_health(conn, device_id: str, now: int | None = None, window_s: int = 86400) -> dict:
    now = int(now or time.time())
    since = now - window_s
    first_ts = conn.execute("SELECT MIN(ts) FROM readings WHERE device_id = ?", (device_id,)).fetchone()[0]
    df = db.load_readings(conn, device_id, since=since)
    ts = df["ts"].to_numpy()

    window_start = max(since, first_ts) if first_ts is not None else now
    comp, interval = completeness(ts, window_start, now)
    if now - window_start < 3600:  # node installed less than an hour ago: too early to judge
        comp = None
    last_ts = int(ts[-1]) if len(ts) else conn.execute(
        "SELECT MAX(ts) FROM readings WHERE device_id = ?", (device_id,)).fetchone()[0]
    age = now - last_ts if last_ts is not None else None
    missing = {f: int(n) for f, n in df[FIELDS].isna().sum().items() if n}
    faults = conn.execute(
        f"SELECT COUNT(*) FROM anomalies WHERE device_id = ? AND end_ts >= ? "
        f"AND kind IN ({','.join('?' * len(SENSOR_FAULT_KINDS))})",
        (device_id, since, *SENSOR_FAULT_KINDS),
    ).fetchone()[0]

    online = age is not None and age <= config.OFFLINE_AFTER_S
    degraded = (comp is not None and comp < config.MIN_COMPLETENESS) or bool(missing) or faults > 0
    return {
        "device_id": device_id,
        "status": "offline" if not online else "degraded" if degraded else "ok",
        "online": online,
        "last_ts": last_ts,
        "age_s": age,
        "n_readings": int(len(ts)),
        "interval_s": round(interval, 1) if interval else None,
        "completeness": round(comp, 4) if comp is not None else None,
        "missing_values": missing,
        "fault_events": int(faults),
    }
