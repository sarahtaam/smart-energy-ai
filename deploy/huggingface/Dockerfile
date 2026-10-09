# Live demo for Hugging Face Spaces: the whole stack in one container, with simulated ESP32 nodes.
# Only the dashboard (port 7860) is exposed; the MQTT broker and the API stay internal to the container.
#
#   docker build -f deploy/huggingface/Dockerfile -t smart-energy-demo .
#   docker run --rm -p 7860:7860 smart-energy-demo        ->  http://localhost:7860
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends mosquitto \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
ENV PYTHONUNBUFFERED=1 DEMO_MODE=1

COPY --chown=user smart_energy ./smart_energy
COPY --chown=user dashboard ./dashboard
COPY --chown=user deploy/huggingface/start.sh deploy/huggingface/mosquitto.conf ./

EXPOSE 7860
CMD ["bash", "start.sh"]
