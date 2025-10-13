#!/bin/sh
set -e

MODEL_PATH=${MODEL_PATH:-/app/model/rf_model.pkl}
TRIES=${TRIES:-30}
SLEEP=${SLEEP:-2}

i=0
while [ ! -f "$MODEL_PATH" ] && [ $i -lt $TRIES ]; do
  echo "Waiting for model at $MODEL_PATH ($i/$TRIES)..."
  sleep $SLEEP
  i=$((i+1))
done

exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}




