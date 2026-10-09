FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY smart_energy ./smart_energy
COPY dashboard ./dashboard
ENV PYTHONUNBUFFERED=1 DATA_DIR=/app/data MODELS_DIR=/app/models
