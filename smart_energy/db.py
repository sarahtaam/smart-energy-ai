"""SQLite storage for readings and detected anomalies."""
import sqlite3
from contextlib import contextmanager
from typing import Iterable, Optional

import pandas as pd

from .config import DB_PATH
from .schemas import FIELDS

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    ts INTEGER NOT NULL,
    voltage REAL, current REAL, power REAL, energy REAL,
    frequency REAL, pf REAL, temperature REAL, humidity REAL
);
CREATE INDEX IF NOT EXISTS idx_readings_device_ts ON readings(device_id, ts);
CREATE INDEX IF NOT EXISTS idx_readings_ts ON readings(ts);

CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id TEXT NOT NULL,
    ts INTEGER NOT NULL,        -- event start
    end_ts INTEGER NOT NULL,    -- event end (last anomalous reading)
    kind TEXT NOT NULL,
    score REAL,                 -- max Isolation Forest score during the event
    power REAL,                 -- peak power during the event
    n_points INTEGER NOT NULL DEFAULT 1,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_anomalies_device_ts ON anomalies(device_id, ts);
"""


def connect(path=DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(SCHEMA)
    return conn


@contextmanager
def session(path=DB_PATH):
    conn = connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def insert_readings(conn: sqlite3.Connection, rows: Iterable[dict]) -> None:
    cols = ["device_id", "ts"] + FIELDS
    conn.executemany(
        f"INSERT INTO readings ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        ([r.get(c) for c in cols] for r in rows),
    )


def insert_events(conn, events: Iterable[dict]) -> None:
    cols = ["device_id", "ts", "end_ts", "kind", "score", "power", "n_points", "detail"]
    conn.executemany(
        f"INSERT INTO anomalies ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
        ([e[c] for c in cols] for e in events),
    )


def upsert_event(conn, device_id, ts, kind, score, power, detail="", max_gap_s=300) -> bool:
    """Extend the device's latest event of this kind if it is recent, else open a new one.

    Matching on kind lets several problems run at once (e.g. a frozen sensor and a missing
    temperature) without each one splitting the other into many events.
    Returns True when a NEW event was created (useful for alerting).
    """
    row = conn.execute(
        "SELECT id, end_ts FROM anomalies WHERE device_id = ? AND kind = ? ORDER BY end_ts DESC LIMIT 1",
        (device_id, kind),
    ).fetchone()
    if row and 0 <= ts - row[1] <= max_gap_s:
        conn.execute(
            """UPDATE anomalies SET end_ts = ?, n_points = n_points + 1,
               score = MAX(COALESCE(score, 0), ?), power = MAX(COALESCE(power, 0), ?) WHERE id = ?""",
            (ts, score, power, row[0]),
        )
        return False
    insert_events(conn, [{"device_id": device_id, "ts": ts, "end_ts": ts, "kind": kind, "score": score,
                          "power": power, "n_points": 1, "detail": detail}])
    return True


def load_readings(conn, device_id: Optional[str] = None, since: Optional[int] = None,
                  until: Optional[int] = None) -> pd.DataFrame:
    q = "SELECT * FROM readings WHERE 1=1"
    params: list = []
    if device_id:
        q += " AND device_id = ?"
        params.append(device_id)
    if since is not None:
        q += " AND ts >= ?"
        params.append(since)
    if until is not None:
        q += " AND ts < ?"
        params.append(until)
    q += " ORDER BY device_id, ts"
    return pd.read_sql_query(q, conn, params=params)


def load_anomalies(conn, device_id: Optional[str] = None, since: Optional[int] = None,
                   limit: int = 500) -> pd.DataFrame:
    q = "SELECT * FROM anomalies WHERE 1=1"
    params: list = []
    if device_id:
        q += " AND device_id = ?"
        params.append(device_id)
    if since is not None:
        q += " AND end_ts >= ?"
        params.append(since)
    q += " ORDER BY ts DESC LIMIT ?"
    params.append(limit)
    return pd.read_sql_query(q, conn, params=params)


def list_devices(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        """SELECT r.* , c.n_readings, c.first_ts FROM readings r
           JOIN (SELECT device_id, MAX(id) AS max_id, COUNT(*) AS n_readings, MIN(ts) AS first_ts
                 FROM readings GROUP BY device_id) c ON r.id = c.max_id
           ORDER BY r.device_id""",
        conn,
    )
