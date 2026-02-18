#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="8765"
PID_FILE="$ROOT_DIR/.nhanes_dashboard_server.pid"
LOG_FILE="$ROOT_DIR/dashboard/server.log"
URL="http://127.0.0.1:${PORT}/dashboard/index.html"

mkdir -p "$ROOT_DIR/dashboard"

# Always restart the local server to avoid stale processes/cached data issues.
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${OLD_PID}" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 0.5
  fi
fi

PORT_PID="$(lsof -ti tcp:${PORT} 2>/dev/null || true)"
if [[ -n "${PORT_PID}" ]]; then
  kill "$PORT_PID" 2>/dev/null || true
  sleep 0.5
fi

nohup python3 -m http.server "$PORT" --directory "$ROOT_DIR" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"
sleep 1

open "$URL"
echo "Dashboard opened at: $URL"
