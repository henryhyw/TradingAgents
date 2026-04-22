#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

RUN_SCRIPT="${REPO_DIR}/scripts/gcp/vm_daily_run.sh"

echo "Installing cron entry on VM ${VM_NAME}"
gcp_vm_ssh_retry --command "bash -lc '
set -euo pipefail
chmod +x \"${RUN_SCRIPT}\"
mkdir -p \"${APP_HOME}/logs\"
TMP_CRON=\$(mktemp)
{
  echo \"CRON_TZ=America/New_York\"
  crontab -l 2>/dev/null | grep -v \"vm_daily_run.sh\" | grep -v \"^CRON_TZ=America/New_York\$\" || true
  echo \"${CRON_SCHEDULE_NY} ${RUN_SCRIPT} >> ${APP_HOME}/logs/cron.log 2>&1\"
} > \"\${TMP_CRON}\"
crontab \"\${TMP_CRON}\"
rm -f \"\${TMP_CRON}\"
crontab -l
'"

echo "Cron installation complete."
