#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-ta-henry-2026}"
REGION="${REGION:-us-central1}"
ZONE="${ZONE:-us-central1-a}"
VM_NAME="${VM_NAME:-ta-runner-01}"
BUCKET_NAME="${BUCKET_NAME:-ta-artifacts-ta-henry-2026}"

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
