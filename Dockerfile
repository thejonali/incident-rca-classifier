FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV INCIDENT_MODELS_DIR=/app/models
ENV INCIDENT_FEATURE_PROFILE=alert_only

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN pip install --no-cache-dir fastapi uvicorn numpy pandas scikit-learn joblib \
    && pip install --no-cache-dir --no-deps .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).read()"

CMD ["uvicorn", "incident_data_classification.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
