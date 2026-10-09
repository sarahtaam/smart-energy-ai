"""Hourly energy consumption forecasting (per device) with prediction intervals."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import forecast_features, hourly_energy

QUANTILES = {"p10": 0.1, "p90": 0.9}


def _model(loss="squared_error", quantile=None) -> HistGradientBoostingRegressor:
    kw = dict(max_iter=250, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, random_state=0)
    if loss == "quantile":
        kw.update(loss="quantile", quantile=quantile)
    return HistGradientBoostingRegressor(**kw)


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ok = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[ok], y_pred[ok]
    err = y_pred - y_true
    nz = y_true > 0.05  # MAPE is meaningless on (near-)zero hours, e.g. outages
    return {
        "MAE_kWh": round(float(np.mean(np.abs(err))), 4),
        "RMSE_kWh": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "MAPE_%": round(float(np.mean(np.abs(err[nz]) / y_true[nz]) * 100), 2),
        "R2": round(float(1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)), 4),
        "n_hours": int(len(y_true)),
    }


def fit_interval(X: pd.DataFrame, y: pd.Series, cal_days: int = 7, coverage: float = 0.8):
    """Conformalized quantile regression (Romano et al., 2019).

    P10/P90 models are fit on all but the last `cal_days`; on that calibration week we measure how far
    the truth falls outside the band and widen (or shrink) it by that margin, so the interval reaches
    its nominal coverage out of sample instead of being over-confident.
    """
    cut = y.index.max() - pd.Timedelta(days=cal_days)
    fit, cal = y.index <= cut, y.index > cut
    lo = _model("quantile", QUANTILES["p10"]).fit(X[fit], y[fit])
    hi = _model("quantile", QUANTILES["p90"]).fit(X[fit], y[fit])
    scores = np.maximum(lo.predict(X[cal]) - y[cal], y[cal] - hi.predict(X[cal]))
    n = int(cal.sum())
    margin = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * coverage) / n)))
    return lo, hi, margin


@dataclass
class Forecaster:
    models: dict = field(default_factory=dict)  # device -> {"mean", "p10", "p90" models + conformal "margin"}
    metrics: dict = field(default_factory=dict)

    @staticmethod
    def _dataset(hourly: pd.Series):
        idx = hourly.index[hourly.notna()]
        X = forecast_features(hourly, idx)
        keep = X["lag_168"].notna()  # need at least one week of history per sample
        return X[keep], hourly[idx][keep]

    def evaluate(self, device: str, hourly: pd.Series, test_days: int = 7) -> dict:
        """Hold out the last `test_days` and compare against naive baselines."""
        X, y = self._dataset(hourly)
        cutoff = y.index.max() - pd.Timedelta(days=test_days)
        tr, te = y.index <= cutoff, y.index > cutoff
        model = _model().fit(X[tr], y[tr])
        pred = model.predict(X[te])
        lo_m, hi_m, margin = fit_interval(X[tr], y[tr])
        lo, hi = lo_m.predict(X[te]) - margin, hi_m.predict(X[te]) + margin
        res = {
            "model": regression_metrics(y[te], pred),
            "baseline_yesterday": regression_metrics(y[te], X.loc[te, "lag_24"]),
            "baseline_last_week": regression_metrics(y[te], X.loc[te, "lag_168"]),
            "interval_coverage_p10_p90": round(float(np.mean((y[te] >= lo) & (y[te] <= hi))), 3),
            "interval_margin_kWh": round(margin, 4),
            "train_hours": int(tr.sum()),
        }
        self.metrics[device] = res
        return res

    def fit(self, readings: pd.DataFrame, test_days: int = 7) -> "Forecaster":
        for device, g in readings.groupby("device_id"):
            hourly = hourly_energy(g.sort_values("ts"))
            if hourly.notna().sum() < 24 * (14 + test_days):
                print(f"[forecast] {device}: not enough history ({hourly.notna().sum()} h), skipped")
                continue
            self.evaluate(device, hourly, test_days)
            X, y = self._dataset(hourly)  # final models use all the data
            lo, hi, margin = fit_interval(X, y)
            self.models[device] = {"mean": _model().fit(X, y), "p10": lo, "p90": hi, "margin": margin}
        return self

    def predict(self, device: str, readings: pd.DataFrame, horizon: int = 24) -> pd.DataFrame:
        """Forecast the next `horizon` hours starting at the current (partial) hour.

        Horizons beyond 24 h are produced recursively, 24 h at a time.
        """
        models = self.models[device]
        hourly = hourly_energy(readings.sort_values("ts"))
        start = hourly.index.max()  # current hour (partial) is re-forecast
        hist = hourly[hourly.index < start].copy()
        out = []
        for chunk_start in range(0, horizon, 24):
            idx = pd.date_range(start + pd.Timedelta(hours=chunk_start), periods=min(24, horizon - chunk_start),
                                freq="1h")
            X = forecast_features(hist, idx)
            mean = models["mean"].predict(X)
            chunk = pd.DataFrame({
                "mean": mean,
                "p10": np.minimum(models["p10"].predict(X) - models["margin"], mean),
                "p90": np.maximum(models["p90"].predict(X) + models["margin"], mean),
            }, index=idx).clip(lower=0)
            hist = pd.concat([hist, chunk["mean"]])
            out.append(chunk)
        res = pd.concat(out).rename(columns={"mean": "kwh"})
        res.index.name = "hour_utc"
        return res.round(4)
