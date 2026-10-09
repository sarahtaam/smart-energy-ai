# Smart Energy AI – IoT & Machine Learning

End-to-end energy monitoring platform: **ESP32 sensor nodes → MQTT → Python ingestion pipeline →
real-time anomaly detection → day-ahead consumption forecasting → REST API → live dashboard.**

```
 ┌──────────────┐  MQTT/JSON   ┌───────────┐   ┌────────────────────────────┐   ┌──────────┐   ┌───────────┐
 │ ESP32 node   │ ───────────▶ │ Mosquitto │──▶│ Ingestor                   │──▶│ SQLite   │──▶│ FastAPI   │
 │ PZEM-004T    │ smartenergy/ │  broker   │   │ • pydantic validation      │   │ readings │   │ /readings │
 │ DHT22        │ <id>/telemetry└───────────┘   │ • rules + z-score + IForest│   │ anomalies│   │ /anomalies│
 └──────────────┘                              │ • event merging, hot reload│   └────┬─────┘   │ /forecast │
 ┌──────────────┐                              └────────────────────────────┘        │         └─────┬─────┘
 │ Simulator    │ (same MQTT contract, labelled anomalies)       ┌───────────────────┴──┐            │
 └──────────────┘                                                │ Training (offline)   │     ┌──────▼──────┐
                                                                 │ IsolationForest +    │     │ Streamlit   │
                                                                 │ GBM forecaster (P10, │     │ dashboard   │
                                                                 │ mean, P90) + metrics │     └─────────────┘
                                                                 └──────────────────────┘
```

## Project layout

| Path | Role |
|---|---|
| `firmware/esp32_energy_node/` | Arduino firmware: PZEM-004T (V, I, P, kWh, Hz, PF) + DHT22, Wi-Fi, NTP, MQTT with last-will |
| `smart_energy/simulator.py` | Realistic multi-device simulator (office hours, weekends, Sfax climate cooling load) with labelled anomalies |
| `smart_energy/ingestor.py` | MQTT subscriber: validation → storage → live anomaly scoring → event log, plus sensor watchdog |
| `smart_energy/health.py` | Sensor health: online status, 24 h data completeness, missing values |
| `smart_energy/alerts.py` | Optional Telegram / e-mail notifications of new events |
| `smart_energy/ml/features.py` | Hour-of-week profiles, anomaly features, lag/calendar forecasting features |
| `smart_energy/ml/anomaly.py` | Hybrid detector: physical rules + robust z-score + per-device Isolation Forest |
| `smart_energy/ml/forecast.py` | Day-ahead hourly forecasting (HistGradientBoosting) with P10/P90 quantile intervals |
| `smart_energy/ml/train.py` | Trains, evaluates (vs ground truth and naive baselines), saves models and `metrics.json` |
| `smart_energy/api.py` | FastAPI REST API (Swagger UI at `/docs`) |
| `dashboard/app.py` | Streamlit live dashboard |
| `streamlit_app.py`, `packages.txt` | Live demo on Streamlit Community Cloud (starts the whole stack in one app) |
| `deploy/huggingface/` | Same demo as a single Docker container |
| `tests/` | pytest suite (detector, forecaster, ingestor end-to-end) |

## Quick start (Windows, without Docker)

You need an MQTT broker. Either start the Mosquitto container
(`docker compose up -d mosquitto`, requires Docker Desktop running) or install
[Mosquitto for Windows](https://mosquitto.org/download/) and run it as a service.

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python -m smart_energy.simulator backfill --days 60   # 60 days of history + ground-truth labels
python -m smart_energy.ml.train                       # train + evaluate both models
pytest                                                # run the test suite
```

Then, in four terminals (each with the venv activated):

```powershell
python -m smart_energy.ingestor                       # 1. MQTT -> DB + live anomaly detection
python -m smart_energy.simulator live                 # 2. fake ESP32 nodes (skip when real nodes publish)
uvicorn smart_energy.api:app --reload                 # 3. API  -> http://localhost:8000/docs
streamlit run dashboard/app.py                        # 4. UI   -> http://localhost:8501
```

## Quick start (Docker, everything in one command)

```powershell
docker compose up --build
```

The `init` service generates the history and trains the models on first launch.

## Live demo

### Streamlit Community Cloud (free)

[`streamlit_app.py`](streamlit_app.py) runs the whole stack behind a single Streamlit process,
with simulated nodes: an internal Mosquitto broker (installed from [`packages.txt`](packages.txt)),
60 days of history and model training, the ingestor, the API and the live simulator. Only the
dashboard is public. After a restart the page shows a progress screen for about 2–3 minutes.

1. Push this repository to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), sign in with GitHub → **Create app** →
   **Deploy a public app from GitHub**.
3. Repository `<user>/smart-energy-ai`, branch `main`, main file path `streamlit_app.py`, and an
   app URL such as `smart-energy-ai`. Under **Advanced settings**, choose Python 3.11.
4. **Deploy**. Every later push to `main` redeploys the app.

The app goes to sleep when nobody visits it for a while; visitors can wake it up with one click.

### Docker (any host)

[`deploy/huggingface/`](deploy/huggingface/) packages the same demo as one container that exposes
only the dashboard, on port 7860:

```powershell
docker build -f deploy/huggingface/Dockerfile -t smart-energy-demo .
docker run --rm -p 7860:7860 smart-energy-demo        # -> http://localhost:7860
```

It also runs as a Hugging Face Docker Space, which now requires a PRO subscription:
`python deploy/huggingface/upload.py <hf-user>/smart-energy-ai` (token from `hf auth login` or `HF_TOKEN`).

## Connecting a real ESP32

1. Arduino IDE → install the ESP32 board package and the libraries listed at the top of the `.ino`.
2. Copy `firmware/esp32_energy_node/config.h.example` to `config.h`; set Wi-Fi, a unique `DEVICE_ID`
   and `MQTT_HOST` = the **LAN IP of the PC running Mosquitto** (`ipconfig`). Allow port 1883 in the
   Windows firewall.
3. Wire it (see the header of the `.ino`). ⚠ The PZEM-004T is connected to 230 V mains:
   wire it with the power off and have it checked by a qualified person.
4. Flash and open the Serial Monitor (115200 baud): you should see `PUB {...}` lines, and the device
   appears in the dashboard.
5. Debug the broker with `mosquitto_sub -h localhost -t "smartenergy/#" -v`.

A new real device is protected immediately by the rule-based checks (outage, voltage, frozen sensor,
meter inconsistency). After ~3 weeks of data, run `python -m smart_energy.ml.train` again: the
ingestor and API hot-reload the new models, and the device gets its own profile, Isolation Forest
and forecaster.

## Data contract

Topic `smartenergy/<device_id>/telemetry`, JSON payload (ts = unix seconds UTC, optional):

```json
{"device_id":"lab_a","ts":1759480000,"voltage":231.2,"current":2.31,"power":480.5,
 "energy":123.456,"frequency":50.0,"pf":0.9,"temperature":24.5,"humidity":55.0}
```

Status topic `smartenergy/<device_id>/status`: `online` / `offline` (MQTT last-will).

## Machine learning

### Anomaly detection (hybrid, real-time)

| Layer | Detects | How |
|---|---|---|
| Physical rules | outage, voltage sag/surge (230 V ±10 %), frozen sensor, meter inconsistency | V < 50 V; V outside 207–253 V; zero rolling std of P, I, V over 10 readings; \|P − V·I·PF\| / P > 25 % |
| Profile z-score | abnormally high/low consumption (e.g. equipment left on at night) | (P − median at that hour-of-week) / robust spread (1.4826·MAD) |
| Isolation Forest | unusual combinations of z, power, voltage, PF, consistency | per device, trained on rule-clean data, threshold at the 99.5th score percentile |

Consecutive anomalous readings of the same kind are merged into **events** (start, end, peak power,
count), so a 2-hour incident is one alert, not 1,440.

### Sensor health (watchdog)

Every minute the ingestor also checks the nodes themselves, including those that stopped sending
anything (no message means the detector above never runs):

| Event | Condition |
|---|---|
| `device_offline` | no reading for more than `OFFLINE_AFTER_S` (120 s), or MQTT last-will `offline` received |
| `missing_field` | a reading has a `null` value (e.g. DHT22 unplugged; `energy` during an outage is ignored) |
| `data_gap_24h` | less than `MIN_COMPLETENESS` (90 %) of the last 24 h covered by readings (gaps > 3× the usual interval) |

The dashboard shows a banner for each offline/degraded node and a "Sensor health · last 24 h" table
(`GET /health/devices`).

### Notifications (optional)

Each new event (anomaly or sensor fault) can be pushed to Telegram and/or e-mail. Set the variables
before starting the ingestor; a channel stays disabled while its variables are empty.

```powershell
$env:TELEGRAM_BOT_TOKEN = "123456:ABC..."   # from @BotFather
$env:TELEGRAM_CHAT_ID   = "123456789"       # your chat id (e.g. from @userinfobot)
$env:SMTP_HOST = "smtp.gmail.com"; $env:SMTP_PORT = "587"
$env:SMTP_USER = "you@gmail.com"; $env:SMTP_PASSWORD = "app-password"; $env:ALERT_EMAIL_TO = "you@example.com"
$env:ALERT_KINDS = "device_offline,missing_field,data_gap_24h,stuck_sensor,outage"   # optional filter
python -m smart_energy.ingestor
```

### Consumption forecasting (day-ahead)

* Target: hourly energy (kWh) per device = mean power over the hour, robust to sampling rate and
  meter resets.
* Features: hour, weekday, weekend flag, cyclic hour encoding, lags 24/48/72/168/336 h, mean of the
  same hour over the past 7 days, mean/max of the previous day. **All lags ≥ 24 h**, so the full
  next day is predicted directly, without feeding predictions back in (longer horizons are
  produced recursively 24 h at a time).
* Models: `HistGradientBoostingRegressor` (squared error) plus two quantile models (P10, P90) for
  an 80 % prediction interval.
* Evaluation: last 7 days held out, compared to the "same hour yesterday" and "same hour last week"
  baselines.

## Results

Measured on 60 days × 3 devices of simulated 1-minute data (259,200 readings), reproducible with
`python -m smart_energy.simulator backfill --days 60 --seed 42 && python -m smart_energy.ml.train`.

**Anomaly detection** (vs. the simulator's ground-truth labels)

| Metric | Value |
|---|---|
| Point precision | 98.6% |
| Point recall | 75.8% |
| Point F1 | 85.7% |
| Event recall (≥1 reading flagged) | 98.7% |

| Anomaly type | Detected events |
|---|---|
| Equipment left on overnight | 18/19 |
| Power outage | 15/15 |
| Power spike (1–5 min) | 69/70 |
| Frozen sensor | 20/20 |
| Voltage sag | 24/24 |

Where point recall is lost (known limitations):

* **Frozen sensor**: the first 9 readings of each episode are missed by design, because 10 identical
  readings are needed to confirm the freeze.
* **Overnight load on `server_room`**: 90 % of those readings are missed. The extra load is only about
  8 % of a 2.4 kW base (z ≈ 2.4), which is inside the normal noise for any single reading. It is
  caught on `lab_a`/`office_b` (0 % missed). A CUSUM / rolling-mean detector for small *sustained*
  shifts would close this gap.

**Forecasting** (hourly kWh, last 7 days held out)

| Device | Model MAE | Model MAPE | Baseline: yesterday MAE | Baseline: last week MAE | P10–P90 coverage |
|---|---|---|---|---|---|
| lab_a | 0.083 kWh | 10.7 % | 0.311 kWh | 0.121 kWh | 80% |
| office_b | 0.050 kWh | 16.3 % | 0.182 kWh | 0.122 kWh | 92% |
| server_room | 0.090 kWh | 3.7 % | 0.128 kWh | 0.177 kWh | 80% |

Prediction intervals are calibrated with conformalized quantile regression: raw quantile
models covered only ~50 % of real values; after calibration coverage is close to the 80 % target.

These numbers are on simulated data; re-run training on real ESP32 data to report real-world results.

## API

| Endpoint | Description |
|---|---|
| `GET /health` | ingestion lag |
| `GET /devices` | devices with last reading and online status |
| `GET /health/devices?hours=24` | sensor health: status, data completeness, missing values, sensor-fault events |
| `GET /readings/{device}?hours=6` | raw readings (downsampled for charts) |
| `GET /consumption/{device}?days=7` | hourly kWh |
| `GET /anomalies?device_id=&hours=168` | anomaly events |
| `GET /forecast/{device}?horizon=24` | hourly forecast with P10/P90 |
| `GET /metrics` | evaluation metrics of the trained models |

## Possible extensions

* TimescaleDB/InfluxDB + Grafana instead of SQLite for many devices.
* Weather forecast API (temperature) as an exogenous forecasting feature.
* TLS + authentication on Mosquitto; OTA firmware updates.
* CUSUM detector for small sustained consumption shifts (see limitations above).
* LSTM/Temporal Fusion Transformer comparison against the GBM model.
