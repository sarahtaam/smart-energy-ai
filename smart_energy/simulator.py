"""ESP32 node simulator.

Generates realistic consumption for several monitored circuits (office hours, weekends,
cooling load driven by outdoor temperature) and injects labelled anomalies.

  backfill : write N days of 1-minute history straight into the database (+ ground-truth labels)
  live     : publish readings over MQTT exactly like a real ESP32 node would

    python -m smart_energy.simulator backfill --days 60
    python -m smart_energy.simulator live --interval 5
"""
import argparse
import json
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import config, db

TZ_OFFSET_H = 1  # Tunisia: UTC+1, no DST

DEVICES = {
    # base/active in W, hvac in W per degC above 24 degC
    "lab_a": dict(base=350, active=1600, hvac=140, noise=0.05, pf=(0.78, 0.95), always_cooled=False),
    "office_b": dict(base=120, active=850, hvac=90, noise=0.06, pf=(0.82, 0.96), always_cooled=False),
    "server_room": dict(base=2200, active=250, hvac=110, noise=0.02, pf=(0.90, 0.97), always_cooled=True),
}

ANOMALY_KINDS = ["spike", "night_load", "stuck", "outage", "voltage_sag"]


def _local_hour_dow(ts: np.ndarray):
    local = ts + TZ_OFFSET_H * 3600
    hour = (local % 86400) / 3600.0
    dow = ((local // 86400) + 3) % 7  # 1970-01-01 was a Thursday -> Monday = 0
    return hour, dow.astype(int)


def outdoor_temperature(ts: np.ndarray) -> np.ndarray:
    """Sfax-like autumn climate: ~24 degC mean, +/-6 degC daily swing, multi-day weather fronts."""
    hour, _ = _local_hour_dow(ts)
    day = ts / 86400.0
    return 24 + 3.0 * np.sin(2 * np.pi * day / 9.3) + 1.5 * np.sin(2 * np.pi * day / 3.7) \
        + 6.0 * np.sin(2 * np.pi * (hour - 9) / 24)


def occupancy(ts: np.ndarray) -> np.ndarray:
    hour, dow = _local_hour_dow(ts)

    def window(start, end):
        return 1 / (1 + np.exp(-(hour - start) * 3)) * 1 / (1 + np.exp((hour - end) * 3))

    weekday = window(8, 17.5) * (1 - 0.35 * np.exp(-((hour - 13) ** 2) / (2 * 0.4 ** 2)))
    saturday = 0.5 * window(8, 13)
    occ = np.where(dow < 5, weekday, np.where(dow == 5, saturday, 0.0))
    return occ + 0.03  # standby/security


def simulate(device: str, ts: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Normal (anomaly-free) readings for the given timestamps."""
    p = DEVICES[device]
    n = len(ts)
    t_out = outdoor_temperature(ts)
    occ = occupancy(ts)
    cooling = np.maximum(0, t_out - 24) * p["hvac"]
    if not p["always_cooled"]:
        cooling = cooling * (0.15 + 0.85 * np.clip(occ, 0, 1))
    power = (p["base"] + p["active"] * occ + cooling) * (1 + p["noise"] * rng.standard_normal(n))
    power = np.maximum(power, 0)

    hour, _ = _local_hour_dow(ts)
    voltage = 230 - 3.0 * np.sin(2 * np.pi * (hour - 13) / 24) + 1.2 * rng.standard_normal(n)
    load_ratio = np.clip(power / (p["base"] + p["active"]), 0, 1)
    pf = np.clip(p["pf"][0] + (p["pf"][1] - p["pf"][0]) * load_ratio + 0.01 * rng.standard_normal(n), 0.5, 1)
    current = power / (voltage * pf)
    frequency = 50 + 0.02 * rng.standard_normal(n)
    base_indoor = 20.0 if p["always_cooled"] else 22.0
    temperature = base_indoor + 0.35 * (t_out - 22) + 1.5 * occ + 0.2 * rng.standard_normal(n)
    humidity = np.clip(55 + 12 * np.sin(2 * np.pi * (hour - 4) / 24) + 2 * rng.standard_normal(n), 10, 95)

    return pd.DataFrame({
        "device_id": device, "ts": ts.astype(np.int64),
        "voltage": voltage, "current": current, "power": power,
        "frequency": frequency, "pf": pf, "temperature": temperature, "humidity": humidity,
    })


def _recompute_current(df: pd.DataFrame, idx) -> None:
    vpf = df.loc[idx, "voltage"] * df.loc[idx, "pf"]
    df.loc[idx, "current"] = np.where(vpf > 0, df.loc[idx, "power"] / vpf.where(vpf > 0, 1), 0.0)


def inject_anomalies(df: pd.DataFrame, device: str, step_s: int, rng: np.random.Generator):
    """Inject anomalies in place. Returns an array of labels ('' = normal)."""
    n = len(df)
    labels = np.array([""] * n, dtype=object)
    days = n * step_s / 86400
    per_min = 60 / step_s
    active = DEVICES[device]["active"]

    def n_events(every_days):
        return rng.poisson(days / every_days)

    def segment(min_minutes, max_minutes):
        length = max(1, int(rng.uniform(min_minutes, max_minutes) * per_min))
        start = int(rng.integers(0, max(1, n - length)))
        return np.arange(start, min(n, start + length))

    for _ in range(n_events(2)):  # short power spikes (inrush, short-circuit-like)
        idx = segment(1, 5)
        df.loc[idx, "power"] *= rng.uniform(2.5, 4.0)
        _recompute_current(df, idx)
        labels[idx] = "spike"

    hour, _ = _local_hour_dow(df["ts"].to_numpy())
    for _ in range(n_events(10)):  # equipment left running overnight
        start = int(rng.integers(0, n))
        night = np.where((hour >= 20) | (hour < 6))[0]
        night = night[(night >= start) & (night < start + int(14 * 60 * per_min))]
        if len(night) == 0:
            continue
        # keep the first contiguous night block
        breaks = np.where(np.diff(night) > 1)[0]
        idx = night[: breaks[0] + 1] if len(breaks) else night
        df.loc[idx, "power"] += active * rng.uniform(0.6, 0.9)
        _recompute_current(df, idx)
        labels[idx] = "night_load"

    for _ in range(n_events(7)):  # voltage sag on the grid
        idx = segment(5, 20)
        factor = rng.uniform(0.78, 0.86)
        df.loc[idx, "voltage"] *= factor
        df.loc[idx, "power"] *= factor ** 2 * 0.3 + 0.7
        _recompute_current(df, idx)
        labels[idx] = "voltage_sag"

    for _ in range(n_events(15)):  # power outage
        idx = segment(10, 60)
        df.loc[idx, ["voltage", "current", "power", "frequency", "pf"]] = 0.0
        labels[idx] = "outage"

    for _ in range(n_events(10)):  # sensor frozen (I2C/UART hang): same values repeated
        idx = segment(30, 120)
        frozen = df.loc[idx[0], ["voltage", "current", "power", "frequency", "pf", "temperature", "humidity"]]
        for col, val in frozen.items():
            df.loc[idx, col] = val
        labels[idx] = "stuck"

    return labels


def add_energy(df: pd.DataFrame, step_s: float, start_kwh: float = 0.0) -> pd.DataFrame:
    df["energy"] = start_kwh + np.cumsum(df["power"].to_numpy() * step_s / 3.6e6)
    return df


def backfill(days: int, step_s: int, seed: int, reset: bool) -> None:
    rng = np.random.default_rng(seed)
    end = int(time.time()) // step_s * step_s
    ts = np.arange(end - days * 86400, end, step_s)
    all_labels = []
    with db.session() as conn:
        if reset:
            conn.execute("DELETE FROM readings")
            conn.execute("DELETE FROM anomalies")
        for device in DEVICES:
            df = simulate(device, ts, rng)
            labels = inject_anomalies(df, device, step_s, rng)
            add_energy(df, step_s, start_kwh=float(rng.uniform(100, 5000)))
            df = df.round({"voltage": 1, "current": 3, "power": 1, "energy": 3, "frequency": 2,
                           "pf": 2, "temperature": 1, "humidity": 1})
            db.insert_readings(conn, df.to_dict("records"))
            all_labels.append(pd.DataFrame({"device_id": device, "ts": df["ts"], "label": labels}))
            n_anom = int((labels != "").sum())
            print(f"{device}: {len(df):,} readings, {n_anom:,} anomalous ({n_anom / len(df):.2%})")
    labels_df = pd.concat(all_labels)
    labels_path = config.DATA_DIR / "labels.csv"
    labels_df[labels_df["label"] != ""].to_csv(labels_path, index=False)
    print(f"Ground-truth anomaly labels -> {labels_path}")


@dataclass
class LiveDevice:
    name: str
    energy: float
    event: str = ""
    remaining: int = 0
    frozen: dict = field(default_factory=dict)


def live(interval: float, anomaly_prob: float, seed: int) -> None:
    import paho.mqtt.client as mqtt

    rng = np.random.default_rng(seed)
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"simulator-{rng.integers(1e6)}")
    if config.MQTT_USER:
        client.username_pw_set(config.MQTT_USER, config.MQTT_PASSWORD)
    client.connect(config.MQTT_HOST, config.MQTT_PORT, keepalive=60)
    client.loop_start()
    devices = {d: LiveDevice(d, energy=float(rng.uniform(100, 5000))) for d in DEVICES}
    durations = {"spike": (1, 3), "outage": (6, 30), "stuck": (15, 40), "voltage_sag": (4, 12),
                 "night_load": (60, 180)}
    print(f"Publishing {len(devices)} devices every {interval}s to {config.MQTT_HOST}:{config.MQTT_PORT} "
          f"(anomaly probability {anomaly_prob}/tick). Ctrl+C to stop.", flush=True)
    try:
        while True:
            now = int(time.time())
            for dev in devices.values():
                row = simulate(dev.name, np.array([now]), rng).iloc[0].to_dict()
                if dev.remaining <= 0 and rng.random() < anomaly_prob:
                    dev.event = str(rng.choice(list(durations)))
                    dev.remaining = int(rng.integers(*durations[dev.event]))
                    dev.frozen = dict(row)
                    print(f"  >> injecting {dev.event} on {dev.name} for {dev.remaining} ticks", flush=True)
                if dev.remaining > 0:
                    if dev.event == "spike":
                        row["power"] *= 3.2
                    elif dev.event == "night_load":
                        row["power"] += DEVICES[dev.name]["active"] * 0.75
                    elif dev.event == "voltage_sag":
                        row["voltage"] *= 0.82
                    elif dev.event == "outage":
                        row.update(voltage=0.0, current=0.0, power=0.0, frequency=0.0, pf=0.0)
                    elif dev.event == "stuck":
                        row.update({k: dev.frozen[k] for k in ["voltage", "current", "power", "frequency",
                                                                "pf", "temperature", "humidity"]})
                    if dev.event in ("spike", "night_load", "voltage_sag"):
                        row["current"] = row["power"] / (row["voltage"] * row["pf"])
                    dev.remaining -= 1
                dev.energy += row["power"] * interval / 3.6e6
                payload = {
                    "device_id": dev.name, "ts": now,
                    "voltage": round(row["voltage"], 1), "current": round(row["current"], 3),
                    "power": round(row["power"], 1), "energy": round(dev.energy, 3),
                    "frequency": round(row["frequency"], 2), "pf": round(row["pf"], 2),
                    "temperature": round(row["temperature"], 1), "humidity": round(row["humidity"], 1),
                }
                client.publish(f"{config.TOPIC_PREFIX}/{dev.name}/telemetry", json.dumps(payload), qos=0)
            time.sleep(max(0.0, interval - (time.time() - now)))
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="generate historical data into the database")
    b.add_argument("--days", type=int, default=60)
    b.add_argument("--step", type=int, default=60, help="seconds between readings")
    b.add_argument("--seed", type=int, default=42)
    b.add_argument("--no-reset", action="store_true", help="keep existing rows")
    lv = sub.add_parser("live", help="publish live readings over MQTT")
    lv.add_argument("--interval", type=float, default=5.0)
    lv.add_argument("--anomaly-prob", type=float, default=0.01)
    lv.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.cmd == "backfill":
        backfill(args.days, args.step, args.seed, reset=not args.no_reset)
    else:
        live(args.interval, args.anomaly_prob, args.seed)


if __name__ == "__main__":
    main()
