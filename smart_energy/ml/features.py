"""Feature engineering shared by training, the live ingestor and the API."""
import numpy as np
import pandas as pd

TZ_OFFSET_H = 1  # Tunisia: UTC+1, no DST


def local_time(ts: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(ts, unit="s") + pd.Timedelta(hours=TZ_OFFSET_H)


def hour_of_week(ts: pd.Series) -> np.ndarray:
    t = local_time(ts)
    return (t.dt.dayofweek * 24 + t.dt.hour).to_numpy()


# anomaly features

def build_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Typical consumption per hour-of-week (median + robust spread) for one device."""
    how = hour_of_week(df["ts"])
    g = pd.DataFrame({"how": how, "power": df["power"].to_numpy()}).groupby("how")["power"]
    med = g.median()
    mad = g.apply(lambda s: (s - s.median()).abs().median())
    prof = pd.DataFrame({"median": med, "spread": 1.4826 * mad}).reindex(range(168))
    prof = prof.interpolate(limit_direction="both")
    # floor the spread so near-constant hours don't produce huge z-scores
    prof["spread"] = np.maximum(prof["spread"], 0.03 * prof["median"].clip(lower=50))
    return prof


def anomaly_features(df: pd.DataFrame, profile: pd.DataFrame | None, stuck_window: int = 10) -> pd.DataFrame:
    """Per-reading features. `df` must contain a single device sorted by ts.

    Without a profile (device never seen in training) z/expected are NaN and only rules apply.
    """
    if profile is not None:
        how = hour_of_week(df["ts"])
        expected = profile["median"].to_numpy()[how]
        spread = profile["spread"].to_numpy()[how]
    else:
        expected = spread = np.full(len(df), np.nan)
    power = df["power"].to_numpy(dtype=float)
    voltage = df["voltage"].to_numpy(dtype=float)
    pf = df["pf"].to_numpy(dtype=float)
    current = df["current"].to_numpy(dtype=float)

    # Power, current AND voltage all exactly constant over the window means the sensor is frozen
    # (max - min is exact; rolling std leaves floating-point residue on constant series)
    roll = df[["power", "current", "voltage"]].rolling(stuck_window, min_periods=stuck_window)
    frozen = ((roll.max() - roll.min()) == 0).all(axis=1).to_numpy()

    return pd.DataFrame({
        "z": (power - expected) / spread,
        "log_power": np.log1p(np.maximum(power, 0)),
        "expected": expected,
        "voltage": voltage,
        "pf": pf,
        # P must equal V * I * PF on a healthy meter
        "consistency": np.abs(power - voltage * current * pf) / (power + 10),
        "frozen": frozen,
    }, index=df.index)


# forecasting features

FORECAST_LAGS = [24, 48, 72, 168, 336]  # hours; all >= 24 -> direct day-ahead forecasting


def hourly_energy(df: pd.DataFrame) -> pd.Series:
    """Hourly consumption in kWh for one device (mean power over the hour / 1000).

    Robust to the sampling interval and to the cumulative counter being reset.
    """
    s = pd.Series(df["power"].to_numpy(dtype=float), index=pd.to_datetime(df["ts"], unit="s"))
    hourly = s.resample("1h").mean() / 1000.0
    return hourly.asfreq("1h")


def forecast_features(hourly: pd.Series, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Features for target hours `index`, using only values at least 24 h older."""
    full = hourly.reindex(hourly.index.union(index))
    local = index + pd.Timedelta(hours=TZ_OFFSET_H)
    X = pd.DataFrame(index=index)
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["is_weekend"] = (local.dayofweek >= 5).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * local.hour / 24)
    X["hour_cos"] = np.cos(2 * np.pi * local.hour / 24)
    for lag in FORECAST_LAGS:
        X[f"lag_{lag}"] = full.shift(lag).reindex(index).to_numpy()
    same_hour = np.column_stack([full.shift(24 * k).reindex(index).to_numpy() for k in range(1, 8)])
    X["same_hour_mean_7d"] = pd.DataFrame(same_hour, index=index).mean(axis=1)
    X["mean_24h_before"] = full.shift(24).rolling(24, min_periods=12).mean().reindex(index).to_numpy()
    X["max_24h_before"] = full.shift(24).rolling(24, min_periods=12).max().reindex(index).to_numpy()
    return X
