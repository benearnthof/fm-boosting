#!/bin/bash

LOG_PATH="/workspace/checkpoints/QURE_AUGMENTED_2025-05-08-18-20-03/wandb/latest-run/files/output.log"
STEP_THRESHOLD=20100
PYTHON_PID=$(pgrep -f train.py)
EPOCH_LEN=8360

echo "[INFO] Monitoring training progress via progress bar in: $LOG_PATH"

if [ ! -f "$LOG_PATH" ]; then
  echo "[ERROR] Log file not found: $LOG_PATH"
  exit 1
fi

while true; do
  # Extract the latest progress line with epoch and step/8360
  PROGRESS_LINE=$(grep -oP 'Epoch\s+\d+:.*\d+/\d+' "$LOG_PATH" | tail -n 1)

  if [ -n "$PROGRESS_LINE" ]; then
    EPOCH=$(echo "$PROGRESS_LINE" | grep -oP 'Epoch\s+\K\d+')
    STEP_IN_EPOCH=$(echo "$PROGRESS_LINE" | grep -oP '\d+/\d+' | cut -d'/' -f1)

    if [ -n "$EPOCH" ] && [ -n "$STEP_IN_EPOCH" ]; then
      GLOBAL_STEP=$((EPOCH * EPOCH_LEN + STEP_IN_EPOCH))
      echo "[INFO] Epoch $EPOCH | Step $STEP_IN_EPOCH/$EPOCH_LEN → Global step: $GLOBAL_STEP"

      if [ "$GLOBAL_STEP" -ge "$STEP_THRESHOLD" ]; then
        echo "[INFO] Step threshold reached. Sending SIGTERM to training (PID $PYTHON_PID)..."
        kill -SIGTERM $PYTHON_PID

        echo "[INFO] Waiting for process $PYTHON_PID to terminate..."
        while ps -p $PYTHON_PID > /dev/null; do
          sleep 5
        done

        echo "[INFO] Training shut down. Terminating pod..."
        shutdown now
        exit 0
      fi
    fi
  else
    echo "[WARN] Parser error..."
  fi

  sleep 60
done
