#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Bootstrapping runtime on VM ${VM_NAME}"
gcp_vm_ssh_retry --ssh-flag="-T" --command "bash -s" <<EOF
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip

sudo mkdir -p "${VM_ROOT_DIR}"
sudo chown "\$USER:\$USER" "${VM_ROOT_DIR}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "Repository directory missing: ${REPO_DIR}" >&2
  exit 1
fi

mkdir -p "${APP_HOME}/logs" "${APP_HOME}/cache" "${APP_HOME}/db" "${APP_HOME}/reports" "${APP_HOME}/artifacts"

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO_DIR}"

cat > "${RUNTIME_ENV_FILE}" <<RUNTIME_ENV
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
RUNTIME_ENV

set -a
source "${RUNTIME_ENV_FILE}"
set +a
cd "${REPO_DIR}"
"${VENV_DIR}/bin/tradingagents" show-config
EOF

echo "VM bootstrap complete."
