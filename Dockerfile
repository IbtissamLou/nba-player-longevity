FROM python:3.10-slim

# Prevent Python from writing .pyc files and force stdout/stderr flushing
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# App lives here
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY app /app/app
COPY ml /app/ml

COPY ml/packaging/packages /app/ml/packaging/packages

# Expose API port
EXPOSE 8000

# Default model package directory
ENV MODEL_PACKAGE_DIR=ml/packaging/packages/latest

# Start FastAPI with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]