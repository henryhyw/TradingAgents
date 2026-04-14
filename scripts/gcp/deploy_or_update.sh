#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

ARCHIVE_PATH="$(mktemp /tmp/tradingagents-src.XXXXXX.tar.gz)"
trap 'rm -f "${ARCHIVE_PATH}"' EXIT

echo "Packaging repository from ${REPO_ROOT}"
tar \
  --exclude=".git" \
  --exclude=".venv" \
  --exclude=".pytest_cache" \
  --exclude=".ruff_cache" \
  --exclude="__pycache__" \
  --exclude="*.pyc" \
  -czf "${ARCHIVE_PATH}" \
  -C "${REPO_ROOT}" .

echo "Copying archive to VM ${VM_NAME}"
gcloud compute scp \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  "${ARCHIVE_PATH}" \
  "${VM_NAME}:/tmp/tradingagents-src.tar.gz"

echo "Expanding source on VM into ${REPO_DIR}"
gcloud compute ssh "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --command "bash -lc '
set -euo pipefail
sudo mkdir -p \"${VM_ROOT_DIR}\"
sudo chown \"\${USER}:\${USER}\" \"${VM_ROOT_DIR}\"
rm -rf \"${REPO_DIR}\"
mkdir -p \"${REPO_DIR}\"
tar -xzf /tmp/tradingagents-src.tar.gz -C \"${REPO_DIR}\" --strip-components=1
'"

echo "Code deployment complete."
