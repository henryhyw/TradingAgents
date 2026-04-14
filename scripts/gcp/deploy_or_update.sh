#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${SCRIPT_DIR}/env.sh"

DEPLOY_SOURCE="${DEPLOY_SOURCE:-auto}" # auto | remote | local
REPO_URL="$(git -C "${REPO_ROOT}" remote get-url origin)"
BRANCH_NAME="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD)"
LOCAL_HEAD="$(git -C "${REPO_ROOT}" rev-parse HEAD)"

git -C "${REPO_ROOT}" fetch --quiet origin "${BRANCH_NAME}" || true
REMOTE_HEAD="$(git -C "${REPO_ROOT}" rev-parse --verify "origin/${BRANCH_NAME}" 2>/dev/null || true)"

if [[ "${DEPLOY_SOURCE}" == "auto" ]]; then
  if [[ -n "${REMOTE_HEAD}" && "${LOCAL_HEAD}" == "${REMOTE_HEAD}" ]]; then
    DEPLOY_SOURCE="remote"
  else
    DEPLOY_SOURCE="local"
  fi
fi

if [[ "${DEPLOY_SOURCE}" != "remote" && "${DEPLOY_SOURCE}" != "local" ]]; then
  echo "Invalid DEPLOY_SOURCE=${DEPLOY_SOURCE}. Use auto, remote, or local." >&2
  exit 2
fi

echo "Deploy source: ${DEPLOY_SOURCE}"
echo "repo=${REPO_URL}"
echo "branch=${BRANCH_NAME}"
echo "local_commit=${LOCAL_HEAD}"
echo "remote_commit=${REMOTE_HEAD:-missing}"

if [[ "${DEPLOY_SOURCE}" == "remote" ]]; then
  if [[ -z "${REMOTE_HEAD}" ]]; then
    echo "origin/${BRANCH_NAME} not found; cannot perform remote deployment." >&2
    exit 1
  fi
  echo "Deploying via VM-side git sync"

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
else
  echo "Deploying from local workspace archive"
  ARCHIVE_PATH="$(mktemp /tmp/tradingagents-src.XXXXXX)"
  trap 'rm -f "${ARCHIVE_PATH}"' EXIT

  COPYFILE_DISABLE=1 tar \
    --no-xattrs \
    --no-mac-metadata \
    --exclude=".git" \
    --exclude=".venv" \
    --exclude=".pytest_cache" \
    --exclude=".ruff_cache" \
    --exclude="__pycache__" \
    --exclude="*.pyc" \
    -czf "${ARCHIVE_PATH}" \
    -C "${REPO_ROOT}" .

  gcp_vm_scp_retry "${ARCHIVE_PATH}" "${VM_NAME}:/tmp/tradingagents-src.tar.gz"
  gcp_vm_ssh_retry --command "bash -s" <<EOF
set -euo pipefail
sudo mkdir -p "${VM_ROOT_DIR}"
sudo chown "\$USER:\$USER" "${VM_ROOT_DIR}"
rm -rf "${REPO_DIR}"
mkdir -p "${REPO_DIR}"
tar -xzf /tmp/tradingagents-src.tar.gz -C "${REPO_DIR}"
rm -f /tmp/tradingagents-src.tar.gz

echo "VM repo unpacked from local archive."
EOF
fi

echo "Code deployment complete."
