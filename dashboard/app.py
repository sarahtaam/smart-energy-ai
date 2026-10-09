"""Smart Energy AI dashboard.

    streamlit run dashboard/app.py
"""
import os
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000")
TZ = "Africa/Tunis"

# Reference palette: categorical slots 1-2, reserved status colors, recessive chrome
ACTUAL = "#2a78d6"     # series 1 - measured consumption
FORECAST = "#eb6834"   # series 2 - forecast
CRITICAL = "#d03b3b"   # status: anomaly
GRID = "rgba(137,135,129,0.25)"

KIND_LABELS = {
    "outage": "⛔ Power outage",
    "stuck_sensor": "🧊 Frozen sensor",
    "voltage_sag": "📉 Voltage sag",
    "voltage_surge": "📈 Voltage surge",
    "meter_inconsistency": "⚠️ Meter inconsistency",
    "high_consumption": "🔺 Abnormally high consumption",
    "low_consumption": "🔻 Abnormally low consumption",
    "device_offline": "📴 Device offline (no data)",
    "missing_field": "❓ Missing sensor values",
    "data_gap_24h": "🕳️ Incomplete data (24 h)",
}
HEALTH_LABELS = {"ok": "🟢 OK", "degraded": "🟠 Degraded", "offline": "🔴 Offline"}

st.set_page_config(page_title="Smart Energy AI", page_icon="⚡", layout="wide")


@st.cache_data(ttl=4)
def api(path: str, **params):
    r = requests.get(f"{API}{path}", params=params, timeout=15)
    if r.status_code >= 400:
        return None
    return r.json()


def to_local(ts_seconds: pd.Series) -> pd.Series:
    return pd.to_datetime(ts_seconds, unit="s", utc=True).dt.tz_convert(TZ)


def fmt_age(seconds) -> str:
    if seconds is None:
        return "never"
    if seconds < 120:
        return f"{seconds:.0f} s ago"
    if seconds < 7200:
        return f"{seconds / 60:.0f} min ago"
    return f"{seconds / 3600:.1f} h ago"


def base_layout(fig: go.Figure, y_title: str, height=360) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        yaxis_title=y_title,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, rangemode="tozero")
    return fig


# ------------------------------------------------------------------ sidebar
try:
    devices = api("/devices")
except requests.RequestException:
    devices = None
if not devices:
    st.error(f"Cannot reach the API at {API} or no data yet. Start it with "
             "`uvicorn smart_energy.api:app` and run the simulator or an ESP32 node.")
    st.stop()

st.sidebar.title("⚡ Smart Energy AI")
if os.getenv("DEMO_MODE"):
    st.sidebar.info("🧪 **Live demo.** Three simulated ESP32 nodes publish over MQTT every 5 s, and anomalies "
                    "are injected at random. Data resets when the demo restarts.")
names = [d["device_id"] for d in devices]
device = st.sidebar.selectbox("Device", names)
window_h = st.sidebar.select_slider("Live window", options=[1, 3, 6, 12, 24, 48], value=6,
                                    format_func=lambda h: f"{h} h")
history_days = st.sidebar.select_slider("History shown with forecast", options=[3, 7, 14, 30], value=7,
                                        format_func=lambda d: f"{d} days")
refresh = st.sidebar.toggle("Auto-refresh (5 s)", value=True)
st.sidebar.divider()
for d in devices:
    status = "🟢 online" if d["online"] else "⚪ offline"
    st.sidebar.caption(f"**{d['device_id']}** · {status} · {d['n_readings']:,} readings")


# ------------------------------------------------------------------ sensor health (all devices)
@st.fragment(run_every="5s" if refresh else None)
def health_section():
    health = api("/health/devices", hours=24)
    if not health:
        return
    # Banners first: a silent or degraded node must be visible without scrolling
    for h in health:
        if h["status"] == "offline":
            st.error(f"📴 **{h['device_id']}** is not sending data · last reading {fmt_age(h['age_s'])}")
        elif h["completeness"] is not None and h["completeness"] < 0.9:
            st.warning(f"🕳️ **{h['device_id']}**: only {h['completeness']:.0%} of the last 24 h received")
        elif h["missing_values"]:
            st.warning(f"❓ **{h['device_id']}**: missing values for {', '.join(h['missing_values'])} in the last 24 h")

    with st.expander("Sensor health · last 24 h", expanded=any(h["status"] != "ok" for h in health)):
        st.dataframe(pd.DataFrame([{
            "Device": h["device_id"],
            "Status": HEALTH_LABELS.get(h["status"], h["status"]),
            "Last reading": fmt_age(h["age_s"]),
            "Readings": h["n_readings"],
            "Interval (s)": h["interval_s"],
            "Data received": f"{h['completeness']:.1%}" if h["completeness"] is not None else "—",
            "Missing values": ", ".join(f"{f}: {n}" for f, n in h["missing_values"].items()) or "—",
            "Sensor faults": h["fault_events"],
        } for h in health]), hide_index=True, width="stretch")
        st.caption("Sensor faults = offline, missing values, incomplete data, frozen sensor and meter "
                   "inconsistency events in the last 24 h.")


health_section()


# ------------------------------------------------------------------ live section
@st.fragment(run_every="5s" if refresh else None)
def live_section():
    readings = pd.DataFrame(api(f"/readings/{device}", hours=window_h) or [])
    anomalies = pd.DataFrame(api("/anomalies", device_id=device, hours=24 * 30, limit=500) or [])
    fc = api(f"/forecast/{device}", horizon=24)
    today = pd.DataFrame(api(f"/consumption/{device}", days=1) or [])

    if readings.empty:
        st.info("No recent readings for this device.")
        return
    readings["time"] = to_local(readings["ts"])
    last = readings.iloc[-1]
    age = int(datetime.now(timezone.utc).timestamp() - last["ts"])

    # KPI tiles
    start_of_day = pd.Timestamp.now(tz=TZ).normalize()
    today_kwh = 0.0
    if not today.empty:
        today["time"] = pd.to_datetime(today["hour_utc"]).dt.tz_localize("UTC").dt.tz_convert(TZ)
        today_kwh = today.loc[today["time"] >= start_of_day, "kwh"].sum()
    recent_anoms = anomalies[anomalies["end_ts"] >= last["ts"] - 86400] if not anomalies.empty else anomalies

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Power now", f"{last['power']:,.0f} W", help=f"Last reading {age}s ago")
    c2.metric("Voltage", f"{last['voltage']:.1f} V", help="Nominal 230 V ±10 %")
    c3.metric("Energy today", f"{today_kwh:,.2f} kWh")
    c4.metric("Forecast next 24 h", f"{fc['total_kwh']:,.1f} kWh" if fc else "—",
              help="Gradient boosting day-ahead model")
    c5.metric("Anomaly events (24 h)", f"{len(recent_anoms)}")

    # Live power with anomaly markers
    st.subheader(f"Live power · last {window_h} h")
    fig = go.Figure()
    fig.add_scatter(x=readings["time"], y=readings["power"], mode="lines", name="Measured power",
                    line=dict(color=ACTUAL, width=2), hovertemplate="%{y:,.0f} W")
    if not anomalies.empty:
        a = anomalies[anomalies["end_ts"] >= readings["ts"].min()].copy()
        if not a.empty:
            a["time"] = to_local(a["ts"])
            a["label"] = a["kind"].map(KIND_LABELS).fillna(a["kind"])
            fig.add_scatter(x=a["time"], y=a["power"], mode="markers", name="Anomaly",
                            marker=dict(color=CRITICAL, size=10, symbol="diamond",
                                        line=dict(width=2, color="white")),
                            customdata=a[["label"]], hovertemplate="%{customdata[0]}<br>peak %{y:,.0f} W")
    st.plotly_chart(base_layout(fig, "Power (W)"), width="stretch", theme="streamlit")

    # Anomaly log
    st.subheader("Anomaly events")
    if anomalies.empty:
        st.success("No anomalies detected in the last 30 days.")
    else:
        t = anomalies.head(50).copy()
        t["Start"] = to_local(t["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
        t["Duration"] = pd.to_timedelta(t["duration_s"], unit="s").astype(str).str.replace("0 days ", "")
        t["Type"] = t["kind"].map(KIND_LABELS).fillna(t["kind"])
        t = t.rename(columns={"power": "Peak power (W)", "n_points": "Readings", "score": "IF score",
                              "detail": "Detail"})
        st.dataframe(t[["Start", "Type", "Duration", "Peak power (W)", "Readings", "IF score", "Detail"]],
                     hide_index=True, width="stretch", height=280)


live_section()

# ------------------------------------------------------------------ forecast section
st.subheader("Hourly consumption & 24 h forecast")
hist = pd.DataFrame(api(f"/consumption/{device}", days=history_days) or [])
fc = api(f"/forecast/{device}", horizon=24)
if hist.empty:
    st.info("Not enough data yet.")
else:
    hist["time"] = pd.to_datetime(hist["hour_utc"]).dt.tz_localize("UTC").dt.tz_convert(TZ)
    fig = go.Figure()
    fig.add_scatter(x=hist["time"], y=hist["kwh"], mode="lines", name="Actual",
                    line=dict(color=ACTUAL, width=2), hovertemplate="%{y:.2f} kWh")
    if fc:
        f = pd.DataFrame(fc["hours"])
        f["time"] = pd.to_datetime(f["hour_utc"]).dt.tz_localize("UTC").dt.tz_convert(TZ)
        fig.add_scatter(x=f["time"], y=f["p90"], mode="lines", line=dict(width=0), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=f["time"], y=f["p10"], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor="rgba(235,104,52,0.18)", name="P10–P90 interval", hoverinfo="skip")
        fig.add_scatter(x=f["time"], y=f["kwh"], mode="lines", name="Forecast",
                        line=dict(color=FORECAST, width=2, dash="dot"), hovertemplate="%{y:.2f} kWh")
    else:
        st.warning("Forecast model not trained yet: run `python -m smart_energy.ml.train`.")
    st.plotly_chart(base_layout(fig, "Energy (kWh per hour)", height=380), width="stretch",
                    theme="streamlit")

# ------------------------------------------------------------------ model metrics
with st.expander("Model evaluation"):
    m = api("/metrics")
    if not m:
        st.info("Models not trained yet.")
    else:
        st.caption(f"Trained {pd.to_datetime(m['trained_at'], unit='s', utc=True).tz_convert(TZ):%Y-%m-%d %H:%M} "
                   f"on {m['n_readings']:,} readings")
        if device in m.get("forecast", {}):
            fm = m["forecast"][device]
            rows = {"Gradient boosting (ours)": fm["model"], "Baseline: same hour yesterday": fm["baseline_yesterday"],
                    "Baseline: same hour last week": fm["baseline_last_week"]}
            st.markdown(f"**Forecasting, held-out last 7 days, {device}** · "
                        f"P10–P90 interval coverage: {fm['interval_coverage_p10_p90']:.0%} (target 80 %)")
            st.dataframe(pd.DataFrame(rows).T[["MAE_kWh", "RMSE_kWh", "MAPE_%", "R2"]], width="stretch")
        if "anomaly" in m:
            am = m["anomaly"]
            st.markdown("**Anomaly detection vs. simulator ground truth (all devices)**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Point precision", f"{am['point_precision']:.1%}")
            c2.metric("Point recall", f"{am['point_recall']:.1%}")
            c3.metric("Point F1", f"{am['point_f1']:.1%}")
            c4.metric("Event recall", f"{am['event_recall']:.1%}")
            st.dataframe(pd.DataFrame(am["events_by_kind"]).T.assign(
                recall=lambda d: (d["detected"] / d["events"]).map("{:.0%}".format)), width="stretch")
