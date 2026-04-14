#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

REPO_URL="$(git -C "${REPO_ROOT}" remote get-url origin)"
BRANCH_NAME="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
REMOTE_HEAD="$(git -C "${REPO_ROOT}" rev-parse "origin/${BRANCH_NAME}")"

echo "Deploying via VM-side git sync"
echo "repo=${REPO_URL}"
echo "branch=${BRANCH_NAME}"
echo "target_commit=${REMOTE_HEAD}"

gcp_vm_ssh_retry --command "bash -s" <<EOF
set -euo pipefail

sudo apt-get update -y
sudo apt-get install -y git
sudo mkdir -p "${VM_ROOT_DIR}"
sudo chown "\$USER:\$USER" "${VM_ROOT_DIR}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  rm -rf "${REPO_DIR}"
  git clone --branch "${BRANCH_NAME}" "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"
git remote set-url origin "${REPO_URL}"
git fetch --prune origin "${BRANCH_NAME}"
git checkout "${BRANCH_NAME}" || git checkout -b "${BRANCH_NAME}" "origin/${BRANCH_NAME}"
git reset --hard "origin/${BRANCH_NAME}"
git clean -fd

echo "VM repo now at:"
git rev-parse HEAD
EOF

echo "Code deployment complete."
