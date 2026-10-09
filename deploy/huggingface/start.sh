#!/bin/bash
# Starts the full demo stack in one container. Data is simulated and regenerated on every start.
set -e

mosquitto -c mosquitto.conf -d

echo "Generating 60 days of history and training the models (about 1 minute)..."
python -m smart_energy.simulator backfill --days 60
python -m smart_energy.ml.train

python -m smart_energy.ingestor &
uvicorn smart_energy.api:app --host 127.0.0.1 --port 8000 &
python -m smart_energy.simulator live --interval 5 --anomaly-prob 0.01 &
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 7860 --server.headless true \
    --browser.gatherUsageStats false &

# If any service dies, stop the container so the platform restarts it
wait -n
exit 1
