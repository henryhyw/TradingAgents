#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

RUN_SCRIPT="${REPO_DIR}/scripts/gcp/vm_daily_run.sh"
CRON_LINE="${CRON_MARKET_CLOSE_MINUTE_NY} * * * 1-5 if [[ \"\$(TZ=America/New_York date +\\%H)\" == \"${CRON_MARKET_CLOSE_HOUR_NY}\" ]]; then ${RUN_SCRIPT} >> ${APP_HOME}/logs/cron.log 2>&1; fi"

echo "Installing cron entry on VM ${VM_NAME}"
gcp_vm_ssh_retry --command "bash -s" <<EOF
set -euo pipefail
RUN_SCRIPT="${RUN_SCRIPT}"
APP_HOME="${APP_HOME}"
CRON_LINE='${CRON_LINE}'

chmod +x "\${RUN_SCRIPT}"
mkdir -p "\${APP_HOME}/logs"
TMP_CRON="\$(mktemp)"
{
  echo "SHELL=/bin/bash"
  echo "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
  crontab -l 2>/dev/null \
    | grep -v "vm_daily_run.sh" \
    | grep -v "^CRON_TZ=America/New_York$" \
    | grep -v "^SHELL=/bin/bash$" \
    | grep -v "^PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin$" \
    || true
  echo "\${CRON_LINE}"
} > "\${TMP_CRON}"
crontab "\${TMP_CRON}"
rm -f "\${TMP_CRON}"
crontab -l
EOF

echo "Cron installation complete."
