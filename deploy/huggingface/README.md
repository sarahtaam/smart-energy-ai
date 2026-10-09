---
title: Smart Energy AI
emoji: ⚡
colorFrom: blue
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
short_description: IoT energy monitoring, anomaly detection, forecasts
---

# Smart Energy AI: live demo

Three simulated ESP32 energy nodes publish readings every 5 s over MQTT. The pipeline validates and
stores them, detects anomalies and sensor faults in real time, and forecasts the next 24 h of
consumption. Anomalies (spikes, outages, voltage sags, frozen sensors, equipment left on overnight)
are injected at random.

The data is simulated and regenerated each time the Space restarts.

Source code and documentation: GITHUB_REPO_URL
