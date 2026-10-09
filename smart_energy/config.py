"""Central configuration, overridable through environment variables."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USER = os.getenv("MQTT_USER") or None
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD") or None
TOPIC_PREFIX = os.getenv("TOPIC_PREFIX", "smartenergy")

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
MODELS_DIR = Path(os.getenv("MODELS_DIR", ROOT / "models"))
DB_PATH = Path(os.getenv("DB_PATH", DATA_DIR / "energy.db"))

API_URL = os.getenv("API_URL", "http://localhost:8000")

NOMINAL_VOLTAGE = 230.0  # Tunisia (STEG): 230 V / 50 Hz
VOLTAGE_TOLERANCE = 0.10  # +/-10 %

# Sensor watchdog
OFFLINE_AFTER_S = int(os.getenv("OFFLINE_AFTER_S", "120"))  # no reading for this long -> device_offline
MIN_COMPLETENESS = float(os.getenv("MIN_COMPLETENESS", "0.9"))  # share of the last 24 h covered by readings
WATCHDOG_INTERVAL_S = 60

# Optional notifications of new events (each channel is disabled while its variables are unset)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or None
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or None
SMTP_HOST = os.getenv("SMTP_HOST") or None
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER") or None
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD") or None
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM") or SMTP_USER
ALERT_EMAIL_TO = os.getenv("ALERT_EMAIL_TO") or None
ALERT_KINDS = {k.strip() for k in os.getenv("ALERT_KINDS", "").split(",") if k.strip()}  # empty = all kinds

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
