from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GCSPublishResult:
    bucket: str
    prefix: str
    uploaded_objects: list[str]


def publish_directory_to_gcs(
    *,
    local_root: Path,
    bucket_name: str,
    prefix: str = "",
    project_id: str | None = None,
) -> GCSPublishResult:
    if not local_root.exists():
        raise FileNotFoundError(f"Publish root does not exist: {local_root}")
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(
            "google-cloud-storage is required for GCS publishing. "
            "Install dependencies with `pip install -e .`."
        ) from exc

    prefix_clean = prefix.strip("/")
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    uploaded: list[str] = []

    for path in sorted(local_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(local_root).as_posix()
        object_name = f"{prefix_clean}/{relative}" if prefix_clean else relative
        blob = bucket.blob(object_name)
        blob.upload_from_filename(str(path))
        uploaded.append(object_name)
        logger.info("Uploaded artifact: gs://%s/%s", bucket_name, object_name)

    return GCSPublishResult(bucket=bucket_name, prefix=prefix_clean, uploaded_objects=uploaded)
