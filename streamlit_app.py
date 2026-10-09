"""Entry point for Streamlit Community Cloud (or any host that runs a single Streamlit process).

Starts the whole demo stack once per server: an internal MQTT broker, 60 days of simulated history,
model training, the ingestor, the API and the live simulator. Then it serves the dashboard.
The data is simulated and regenerated every time the app restarts.

    streamlit run streamlit_app.py
"""
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent
WORK = Path(tempfile.gettempdir()) / "smart-energy-demo"
API_URL = "http://127.0.0.1:8000"
ENV = {**os.environ, "DATA_DIR": str(WORK / "data"), "MODELS_DIR": str(WORK / "models"),
       "MQTT_HOST": "127.0.0.1", "PYTHONUNBUFFERED": "1"}
PY = sys.executable
SERVICES = {  # long-running processes, restarted if they die
    "mqtt": ["mosquitto", "-c", str(WORK / "mosquitto.conf")],
    "ingestor": [PY, "-m", "smart_energy.ingestor"],
    "api": [PY, "-m", "uvicorn", "smart_energy.api:app", "--host", "127.0.0.1", "--port", "8000"],
    "simulator": [PY, "-m", "smart_energy.simulator", "live", "--interval", "5", "--anomaly-prob", "0.01"],
}


def _api_ok() -> bool:
    try:
        return requests.get(f"{API_URL}/devices", timeout=2).ok
    except requests.RequestException:
        return False


class DemoStack:
    """Boots the services in a background thread, so the page can show progress meanwhile."""

    def __init__(self):
        self.status = "starting"  # starting | ready | failed
        self.message = "Starting the MQTT broker"
        self.procs: dict[str, subprocess.Popen] = {}
        threading.Thread(target=self._run, daemon=True).start()

    def _spawn(self, name: str) -> None:
        self.procs[name] = subprocess.Popen(SERVICES[name], cwd=ROOT, env=ENV)

    def _step(self, message: str, *args: str) -> None:
        self.message = message
        subprocess.run([PY, "-m", *args], cwd=ROOT, env=ENV, check=True)

    def _run(self) -> None:
        try:
            if _api_ok():  # stack left running by a previous Streamlit process in the same container
                self.status = "ready"
                return
            if not shutil.which("mosquitto"):
                raise RuntimeError("the mosquitto broker is not installed (it is listed in packages.txt)")
            WORK.mkdir(parents=True, exist_ok=True)
            (WORK / "mosquitto.conf").write_text("listener 1883 127.0.0.1\nallow_anonymous true\npersistence false\n"
                                                 "log_type error\nlog_type warning\n")
            self._spawn("mqtt")
            self._step("Generating 60 days of simulated history", "smart_energy.simulator", "backfill", "--days", "60")
            self._step("Training the anomaly detection and forecasting models", "smart_energy.ml.train")
            self.message = "Starting the ingestor, the API and the live simulator"
            for name in ("ingestor", "api", "simulator"):
                self._spawn(name)
            deadline = time.time() + 120
            while not _api_ok():
                if time.time() > deadline:
                    raise RuntimeError("the API did not start, see the app logs")
                time.sleep(1)
            self.status = "ready"
            while True:
                time.sleep(15)
                for name, proc in list(self.procs.items()):
                    if proc.poll() is not None:
                        print(f"[demo] {name} exited with code {proc.returncode}, restarting", flush=True)
                        self._spawn(name)
        except Exception as e:
            self.status, self.message = "failed", f"{type(e).__name__}: {e}"
            print(f"[demo] startup failed: {self.message}", flush=True)


@st.cache_resource
def demo_stack() -> DemoStack:
    return DemoStack()


st.set_page_config(page_title="Smart Energy AI", page_icon="⚡", layout="wide")
stack = demo_stack()
if stack.status != "ready":
    st.title("⚡ Smart Energy AI: live demo")
    if stack.status == "failed":
        st.error(f"The demo could not start: {stack.message}")
        st.stop()
    st.info(f"⏳ Starting the demo, about 2–3 minutes after a restart. **{stack.message}…**")
    st.caption("Three simulated ESP32 nodes, an MQTT broker, the ingestion pipeline and the ML models are "
               "being set up. This page refreshes by itself.")
    time.sleep(3)
    st.rerun()

os.environ["DEMO_MODE"] = "1"
os.environ["API_URL"] = API_URL  # read by the dashboard at import
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
