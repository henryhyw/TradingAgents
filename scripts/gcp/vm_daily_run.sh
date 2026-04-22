#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-/opt/tradingagents/runtime.env}"
VENV_DIR="${VENV_DIR:-/opt/tradingagents/venv}"

if [[ -f "${RUNTIME_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  set -a
  source "${RUNTIME_ENV_FILE}"
  set +a
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

  RUN_SUCCESS=0
  LIVE_RUN_TIMEOUT_SECONDS="${LIVE_RUN_TIMEOUT_SECONDS:-3600}"
  LIVE_RUN_FAILED_REASON="unknown"
  if command -v timeout >/dev/null 2>&1; then
    if timeout --signal=TERM "${LIVE_RUN_TIMEOUT_SECONDS}" tradingagents run-once --as-of "${AS_OF_DATE}" --execute; then
      RUN_SUCCESS=1
    else
      exit_code=$?
      if [[ "${exit_code}" -eq 124 || "${exit_code}" -eq 137 ]]; then
        LIVE_RUN_FAILED_REASON="live_run_timeout"
      else
        LIVE_RUN_FAILED_REASON="live_run_error"
      fi
    fi
  elif tradingagents run-once --as-of "${AS_OF_DATE}" --execute; then
    RUN_SUCCESS=1
  else
    LIVE_RUN_FAILED_REASON="live_run_error"
  fi

  if [[ "${RUN_SUCCESS}" -eq 1 ]]; then
    echo "Primary live run completed for ${AS_OF_DATE}."
  else
    echo "Primary live run failed for ${AS_OF_DATE} (${LIVE_RUN_FAILED_REASON}); attempting deterministic fallback." >&2
    FALLBACK_SYMBOLS="$("${VENV_DIR}/bin/python" - <<'PY'
from tradingagents.system.config import load_settings
from tradingagents.system.storage.repository import TradingRepository

settings = load_settings()
repository = TradingRepository(settings.paths.database_path)
snapshot = repository.get_latest_portfolio_snapshot()

preferred = []
if snapshot is not None:
    preferred.extend([position.symbol for position in snapshot.positions if position.quantity > 0])
preferred.extend(["SPY", "QQQ", "IWM", "XLK", "XLF"])

deduped = []
for symbol in preferred:
    symbol = symbol.strip().upper()
    if symbol and symbol not in deduped:
        deduped.append(symbol)

print(",".join(deduped[:12]))
PY
)"
    echo "Deterministic fallback symbols: ${FALLBACK_SYMBOLS}"
    if tradingagents run-once --as-of "${AS_OF_DATE}" --execute --deterministic-research --symbols "${FALLBACK_SYMBOLS}"; then
      RUN_SUCCESS=1
      echo "Deterministic fallback run completed for ${AS_OF_DATE}."
    else
      echo "Deterministic fallback run failed for ${AS_OF_DATE}." >&2
    fi
  fi

  if tradingagents export-artifacts --as-of "${AS_OF_DATE}"; then
    echo "Artifacts exported for ${AS_OF_DATE}."
  else
    echo "Artifact export failed for ${AS_OF_DATE}." >&2
  fi

  if tradingagents publish-artifacts --as-of "${AS_OF_DATE}"; then
    echo "Artifacts published for ${AS_OF_DATE}."
  else
    echo "Artifact publish failed for ${AS_OF_DATE}." >&2
  fi

  if [[ "${RUN_SUCCESS}" -eq 1 ]]; then
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Daily wrapper completed for ${AS_OF_DATE}"
  else
    echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] Daily wrapper failed for ${AS_OF_DATE}" >&2
    exit 1
  fi
} >>"${LOG_FILE}" 2>&1
