#!/usr/bin/env bash
set -euo pipefail

if ! command -v gcloud >/dev/null 2>&1; then
  if [[ -x "${HOME}/google-cloud-sdk/bin/gcloud" ]]; then
    export PATH="${HOME}/google-cloud-sdk/bin:${PATH}"
  fi
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud CLI not found on PATH. Install Cloud SDK or export PATH before running scripts/gcp/*." >&2
  exit 127
fi

PROJECT_ID="${PROJECT_ID:-ta-henry-2026}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-ta-runner-01}"
BUCKET_NAME="${BUCKET_NAME:-ta-artifacts-ta-henry-2026}"
GCP_USE_IAP="${GCP_USE_IAP:-true}"
GCP_SSH_RETRIES="${GCP_SSH_RETRIES:-4}"
GCP_SSH_RETRY_BACKOFF_SECONDS="${GCP_SSH_RETRY_BACKOFF_SECONDS:-3}"

SERVICE_ACCOUNT_NAME="${SERVICE_ACCOUNT_NAME:-ta-runner-sa}"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_EMAIL:-${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com}"

MACHINE_TYPE="${MACHINE_TYPE:-e2-micro}"
BOOT_DISK_SIZE_GB="${BOOT_DISK_SIZE_GB:-20}"
BOOT_DISK_TYPE="${BOOT_DISK_TYPE:-pd-standard}"
OS_IMAGE_FAMILY="${OS_IMAGE_FAMILY:-ubuntu-2204-lts}"
OS_IMAGE_PROJECT="${OS_IMAGE_PROJECT:-ubuntu-os-cloud}"

VM_ROOT_DIR="${VM_ROOT_DIR:-/opt/tradingagents}"
REPO_DIR="${REPO_DIR:-${VM_ROOT_DIR}/TradingAgents}"
VENV_DIR="${VENV_DIR:-${VM_ROOT_DIR}/venv}"
RUNTIME_ENV_FILE="${RUNTIME_ENV_FILE:-${VM_ROOT_DIR}/runtime.env}"
APP_HOME="${APP_HOME:-${VM_ROOT_DIR}/.tradingagents}"

CRON_SCHEDULE_NY="${CRON_SCHEDULE_NY:-45 15 * * 1-5}"

GCP_SSH_FLAGS=()
if [[ "${GCP_USE_IAP}" == "true" ]]; then
  GCP_SSH_FLAGS+=(--tunnel-through-iap)
fi

gcp_vm_ssh() {
  gcloud compute ssh "${VM_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    "${GCP_SSH_FLAGS[@]}" \
    "$@"
}

gcp_vm_scp() {
  gcloud compute scp \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    "${GCP_SSH_FLAGS[@]}" \
    "$@"
}

gcp_vm_scp_retry() {
  local attempt=1
  local rc=0
  while (( attempt <= GCP_SSH_RETRIES )); do
    if gcp_vm_scp "$@"; then
      return 0
    fi
    rc=$?
    if (( attempt == GCP_SSH_RETRIES )); then
      return "${rc}"
    fi
    local delay=$(( GCP_SSH_RETRY_BACKOFF_SECONDS * attempt ))
    echo "gcp_vm_scp attempt ${attempt}/${GCP_SSH_RETRIES} failed (rc=${rc}); retrying in ${delay}s..." >&2
    sleep "${delay}"
    attempt=$(( attempt + 1 ))
  done
  return "${rc}"
}

gcp_vm_ssh_retry() {
  local attempt=1
  local rc=0
  while (( attempt <= GCP_SSH_RETRIES )); do
    if gcp_vm_ssh "$@"; then
      return 0
    fi
    rc=$?
    if (( attempt == GCP_SSH_RETRIES )); then
      return "${rc}"
    fi
    local delay=$(( GCP_SSH_RETRY_BACKOFF_SECONDS * attempt ))
    echo "gcp_vm_ssh attempt ${attempt}/${GCP_SSH_RETRIES} failed (rc=${rc}); retrying in ${delay}s..." >&2
    sleep "${delay}"
    attempt=$(( attempt + 1 ))
  done
  return "${rc}"
}
