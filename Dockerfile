FROM python:3.12-slim

# Don't write .pyc, unbuffered logs for container-friendly output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Run as a non-root user — never run a service as root.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data && chown -R appuser /app/data
USER appuser

EXPOSE 8000

# Simple container healthcheck against the liveness endpoint.
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
