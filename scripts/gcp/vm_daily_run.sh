#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-/opt/tradingagents/runtime.env}"
VENV_DIR="${VENV_DIR:-/opt/tradingagents/venv}"

if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${RUNTIME_ENV_FILE}"
fi

APP_HOME="${TRADINGAGENTS_HOME:-/opt/tradingagents/.tradingagents}"
mkdir -p "${APP_HOME}/logs"

LOG_FILE="${APP_HOME}/logs/daily-run.log"
{
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Starting daily wrapper"

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  cd "${REPO_ROOT}"

  AS_OF_DATE="$(python - <<'PY'
from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.calendar_utils import default_as_of_date

settings = load_settings()
print(default_as_of_date(settings.run.market_timezone).isoformat())
PY
)"

  ALREADY_DONE="$(AS_OF_DATE="${AS_OF_DATE}" python - <<'PY'
import os
from datetime import date
from tradingagents.system.config import load_settings
from tradingagents.system.storage.repository import TradingRepository

settings = load_settings()
repo = TradingRepository(settings.paths.database_path)
as_of = date.fromisoformat(os.environ["AS_OF_DATE"])
summary = repo.get_run_summary_for_date(as_of)
print("yes" if summary is not None and summary.status == "completed" else "no")
PY
)"

  if [[ "${ALREADY_DONE}" == "yes" ]]; then
    echo "Run already completed for ${AS_OF_DATE}; skipping."
    exit 0
  fi

  tradingagents run-once --as-of "${AS_OF_DATE}" --execute
  tradingagents export-artifacts --as-of "${AS_OF_DATE}"
  tradingagents publish-artifacts --as-of "${AS_OF_DATE}"
  echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Daily wrapper completed for ${AS_OF_DATE}"
} >>"${LOG_FILE}" 2>&1
