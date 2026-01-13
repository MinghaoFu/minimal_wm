#!/bin/bash

set -euo pipefail

PID_FILE="logs/train_tasuw.pids"
SIGNAL="-TERM"

if [ "${1:-}" = "-9" ]; then
    SIGNAL="-KILL"
fi

if [ ! -f "$PID_FILE" ]; then
    echo "PID file not found: $PID_FILE"
    exit 1
fi

echo "Killing processes listed in $PID_FILE with signal $SIGNAL"
while IFS= read -r pid; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$SIGNAL" "$pid" || true
        echo "  Sent $SIGNAL to PID $pid"
    fi
done < "$PID_FILE"
