#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

gcp_vm_ssh_retry --command "bash -lc '
set -euo pipefail
set -a
source \"${RUNTIME_ENV_FILE}\"
set +a
source \"${VENV_DIR}/bin/activate\"
cd \"${REPO_DIR}\"
tradingagents health-check
'"
