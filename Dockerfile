# ---------- builder stage (build wheels to speed up installs) ----------
FROM python:3.10-slim AS builder
WORKDIR /build
COPY requirements.txt .
# build wheels first to improve caching
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ build-essential \
 && pip wheel --no-cache-dir --wheel-dir /build/wheels -r requirements.txt \
 && apt-get purge -y --auto-remove gcc g++ build-essential

# ---------- runtime stage ----------
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_HOME=/app

# create non-root user
RUN useradd --create-home appuser
WORKDIR $APP_HOME

# install wheels if present, else fallback to pip install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache-dir /wheels/* || pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .

# create model dir 
RUN mkdir -p $APP_HOME/model && chown -R appuser:appuser $APP_HOME

# copy entrypoint script
COPY entrypoint.sh /app/entrypoint.sh
RUN chown appuser:appuser /app/entrypoint.sh

USER appuser
EXPOSE 8000
ENTRYPOINT ["/app/entrypoint.sh"]



