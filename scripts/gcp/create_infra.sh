#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/env.sh"

echo "Using project=${PROJECT_ID} zone=${ZONE} vm=${VM_NAME} bucket=${BUCKET_NAME}"
gcloud config set project "${PROJECT_ID}" >/dev/null

if ! gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project "${PROJECT_ID}" >/dev/null 2>&1; then
  echo "Creating service account ${SERVICE_ACCOUNT_EMAIL}"
  gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
    --project "${PROJECT_ID}" \
    --display-name "TradingAgents Runner Service Account"
else
  echo "Service account exists: ${SERVICE_ACCOUNT_EMAIL}"
fi

echo "Granting Vertex AI role to service account"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role "roles/aiplatform.user" >/dev/null

echo "Granting bucket object role to service account"
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
  --member "serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role "roles/storage.objectAdmin" >/dev/null

if gcloud compute instances describe "${VM_NAME}" --project "${PROJECT_ID}" --zone "${ZONE}" >/dev/null 2>&1; then
  echo "VM already exists; updating attached service account/scopes."
  gcloud compute instances set-service-account "${VM_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    --service-account "${SERVICE_ACCOUNT_EMAIL}" \
    --scopes "https://www.googleapis.com/auth/cloud-platform" >/dev/null
else
  echo "Creating VM ${VM_NAME}"
  gcloud compute instances create "${VM_NAME}" \
    --project "${PROJECT_ID}" \
    --zone "${ZONE}" \
    --machine-type "${MACHINE_TYPE}" \
    --boot-disk-size "${BOOT_DISK_SIZE_GB}GB" \
    --boot-disk-type "${BOOT_DISK_TYPE}" \
    --image-family "${OS_IMAGE_FAMILY}" \
    --image-project "${OS_IMAGE_PROJECT}" \
    --service-account "${SERVICE_ACCOUNT_EMAIL}" \
    --scopes "https://www.googleapis.com/auth/cloud-platform"
fi

echo "Infrastructure ready."
