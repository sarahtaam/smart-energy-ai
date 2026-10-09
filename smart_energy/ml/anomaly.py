"""Hybrid anomaly detection: physical rules + profile z-score + Isolation Forest (per device)."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .. import config
from .features import anomaly_features, build_profile

IF_FEATURES = ["z", "log_power", "voltage", "pf", "consistency"]
V_LOW = config.NOMINAL_VOLTAGE * (1 - config.VOLTAGE_TOLERANCE)
V_HIGH = config.NOMINAL_VOLTAGE * (1 + config.VOLTAGE_TOLERANCE)


def rule_kinds(F: pd.DataFrame) -> np.ndarray:
    """Deterministic checks, evaluated in priority order."""
    kinds = np.full(len(F), "", dtype=object)
    conds = [
        ("outage", F["voltage"] < 50),
        ("stuck_sensor", F["frozen"]),
        ("voltage_sag", F["voltage"] < V_LOW),
        ("voltage_surge", F["voltage"] > V_HIGH),
        ("meter_inconsistency", (F["consistency"] > 0.25) & (F["voltage"] >= 50)),
    ]
    for kind, cond in reversed(conds):  # reversed so the highest priority wins
        kinds[cond.to_numpy()] = kind
    return kinds


@dataclass
class AnomalyDetector:
    z_threshold: float = 6.0
    if_quantile: float = 0.995  # Isolation Forest flags the 0.5 % most isolated points
    profiles: dict = field(default_factory=dict)
    models: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)

    def fit(self, readings: pd.DataFrame) -> "AnomalyDetector":
        for device, g in readings.groupby("device_id"):
            g = g.sort_values("ts")
            self.profiles[device] = build_profile(g)
            F = anomaly_features(g, self.profiles[device])
            # Train the forest on readings that pass the rule checks (cleaner "normal" set)
            X = F.loc[rule_kinds(F) == "", IF_FEATURES].dropna()
            model = IsolationForest(n_estimators=200, max_samples=1024, random_state=0, n_jobs=-1).fit(X)
            self.models[device] = model
            self.thresholds[device] = float(np.quantile(-model.score_samples(X), self.if_quantile))
        return self

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score readings of ONE device (sorted by ts). Returns kind ('' = normal) and score."""
        device = df["device_id"].iloc[0]
        F = anomaly_features(df, self.profiles.get(device))
        kinds = rule_kinds(F)
        score = np.zeros(len(F))
        z = F["z"].to_numpy()

        if device in self.models:
            X = F[IF_FEATURES].fillna(0)
            score = -self.models[device].score_samples(X)
            ml_flag = score > self.thresholds[device]
            z_flag = np.abs(z) > self.z_threshold
            # ML-only flags need a minimum deviation from the profile to avoid noise alerts
            flag = (kinds == "") & (z_flag | (ml_flag & (np.abs(z) > 3)))
            kinds[flag & (z > 0)] = "high_consumption"
            kinds[flag & (z <= 0)] = "low_consumption"

        return pd.DataFrame({"kind": kinds, "score": np.round(score, 4), "z": np.round(z, 2),
                             "expected": F["expected"].round(1)}, index=df.index)


def to_events(device_id: str, df: pd.DataFrame, det: pd.DataFrame, max_gap_s: int = 300) -> list[dict]:
    """Merge consecutive anomalous readings of the same kind into events."""
    flagged = df.loc[det["kind"] != "", ["ts", "power"]].join(det[["kind", "score", "z"]])
    events: list[dict] = []
    for row in flagged.itertuples(index=False):
        last = events[-1] if events else None
        if last and last["kind"] == row.kind and row.ts - last["end_ts"] <= max_gap_s:
            last["end_ts"] = int(row.ts)
            last["n_points"] += 1
            last["score"] = max(last["score"], float(row.score))
            last["power"] = max(last["power"], float(row.power))
        else:
            events.append({"device_id": device_id, "ts": int(row.ts), "end_ts": int(row.ts),
                           "kind": row.kind, "score": float(row.score), "power": float(row.power),
                           "n_points": 1, "detail": f"z={row.z:+.1f}" if not np.isnan(row.z) else ""})
    return events
