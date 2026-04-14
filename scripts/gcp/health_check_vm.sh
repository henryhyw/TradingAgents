#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '
set -euo pipefail
source \"${RUNTIME_ENV_FILE}\"
source \"${VENV_DIR}/bin/activate\"
cd \"${REPO_DIR}\"
tradingagents health-check
'"
