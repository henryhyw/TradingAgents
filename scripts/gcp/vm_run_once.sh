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
source "${VENV_DIR}/bin/activate"
cd "${REPO_ROOT}"

AS_OF_DATE="${1:-}"
if [[ -z "${AS_OF_DATE}" ]]; then
  AS_OF_DATE="$(python - <<'PY'
from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.calendar_utils import default_as_of_date

settings = load_settings()
print(default_as_of_date(settings.run.market_timezone).isoformat())
PY
)"
fi

echo "Running tradingagents run-once for ${AS_OF_DATE}"
tradingagents run-once --as-of "${AS_OF_DATE}" --execute
tradingagents export-artifacts --as-of "${AS_OF_DATE}"
tradingagents publish-artifacts --as-of "${AS_OF_DATE}"

echo "Run complete for ${AS_OF_DATE}"
