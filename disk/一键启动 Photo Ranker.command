#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
  PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" ]]; then
  osascript -e 'display dialog "python3 was not found. Please install Python 3 first." buttons {"OK"} default button "OK"'
  exit 1
fi

APP_URL="http://127.0.0.1:8123"
LOG_PATH="${PROJECT_ROOT}/data/photo-ranker-web.log"
PORT=8123

if "$PYTHON_BIN" - <<'PY'
import tkinter as tk
root = tk.Tk()
root.destroy()
PY
then
  exec "$PYTHON_BIN" -m app.desktop
fi

mkdir -p "${PROJECT_ROOT}/data"

OLD_PIDS="$(lsof -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null || true)"
if [[ -n "$OLD_PIDS" ]]; then
  kill $OLD_PIDS >/dev/null 2>&1 || true
  sleep 1
fi

nohup "$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port "$PORT" >"$LOG_PATH" 2>&1 &

for _ in {1..20}; do
  if curl -fsS "${APP_URL}/api/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

open "$APP_URL"
