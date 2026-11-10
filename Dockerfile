# Reproducible runtime for the whole pipeline (Python 3.12, pinned via requirements.txt)
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV MLFLOW_DISABLE_AGENT_HINT=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
CMD ["python", "run_pipeline.py"]
