#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

gcp_vm_ssh_retry --command "bash -lc '
set -euo pipefail
\"${REPO_DIR}/scripts/gcp/vm_daily_run.sh\"
'"
