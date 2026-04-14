#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Bootstrapping runtime on VM ${VM_NAME}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '
set -euo pipefail
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip
mkdir -p \"${VM_ROOT_DIR}\" \"${APP_HOME}/logs\" \"${APP_HOME}/cache\" \"${APP_HOME}/db\" \"${APP_HOME}/reports\" \"${APP_HOME}/artifacts\"
if [[ ! -d \"${VENV_DIR}\" ]]; then
  python3 -m venv \"${VENV_DIR}\"
fi
source \"${VENV_DIR}/bin/activate\"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e \"${REPO_DIR}\"
cat > \"${RUNTIME_ENV_FILE}\" <<EOF
TRADINGAGENTS_HOME=${APP_HOME}
TRADINGAGENTS_LLM_PROVIDER=vertex
TRADINGAGENTS_LLM_MODEL=gemini-2.5-flash
TRADINGAGENTS_LLM_DEEP_MODEL=gemini-2.5-flash
TRADINGAGENTS_LLM_QUICK_MODEL=gemini-2.5-flash
TRADINGAGENTS_VERTEX_PROJECT=${PROJECT_ID}
TRADINGAGENTS_VERTEX_REGION=${REGION}
TRADINGAGENTS_GCP_PROJECT_ID=${PROJECT_ID}
TRADINGAGENTS_GCP_REGION=${REGION}
TRADINGAGENTS_GCP_ZONE=${ZONE}
TRADINGAGENTS_GCP_VM_NAME=${VM_NAME}
TRADINGAGENTS_GCP_SERVICE_ACCOUNT_NAME=${SERVICE_ACCOUNT_NAME}
TRADINGAGENTS_GCS_BUCKET=${BUCKET_NAME}
TRADINGAGENTS_PUBLISH_ON_RUN=true
EOF
source \"${RUNTIME_ENV_FILE}\"
\"${VENV_DIR}/bin/tradingagents\" show-config
'"

echo "VM bootstrap complete."
