from __future__ import annotations

import json
import sys
import types
from datetime import date, datetime, timezone
from pathlib import Path

from tradingagents.system.cloud.gcs_publisher import publish_directory_to_gcs
from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.artifacts import export_publishable_artifacts
from tradingagents.system.schemas import DailyRunSummary, PortfolioSnapshot, RunMode
from tradingagents.system.storage.repository import TradingRepository


def _make_report_files(report_root: Path, as_of: date) -> None:
    report_dir = report_root / as_of.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "summary.md").write_text("# Test Report\n", encoding="utf-8")
    (report_dir / "summary.json").write_text(json.dumps({"date": as_of.isoformat()}), encoding="utf-8")


def test_export_publishable_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    repo = TradingRepository(settings.paths.database_path)
    as_of = date(2026, 4, 13)
    _make_report_files(settings.paths.reports_dir, as_of)

    summary = DailyRunSummary(
        mode=RunMode.DRY_RUN,
        as_of_date=as_of,
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        status="completed",
        report_path=str(settings.paths.reports_dir / as_of.isoformat() / "summary.md"),
    )
    repo.save_daily_run_summary(summary)
    portfolio = PortfolioSnapshot(
        as_of_date=as_of,
        cash=100_000.0,
        equity=100_000.0,
        gross_exposure=0.0,
        positions=[],
    )

    exported = export_publishable_artifacts(
        settings=settings,
        repository=repo,
        as_of_date=as_of,
        portfolio_snapshot=portfolio,
        summary=summary,
    )
    assert exported.report_markdown.exists()
    assert exported.report_json.exists()
    assert exported.latest_positions.exists()
    assert exported.latest_orders.exists()
    assert exported.latest_run_summary.exists()

    positions_payload = json.loads(exported.latest_positions.read_text(encoding="utf-8"))
    assert positions_payload["as_of_date"] == as_of.isoformat()
    assert positions_payload["positions"] == []


def test_publish_directory_to_gcs_with_mocked_client(tmp_path, monkeypatch):
    publish_root = tmp_path / "publish"
    (publish_root / "snapshots").mkdir(parents=True, exist_ok=True)
    (publish_root / "snapshots" / "latest_positions.json").write_text("{}", encoding="utf-8")
    (publish_root / "reports" / "2026-04-13").mkdir(parents=True, exist_ok=True)
    (publish_root / "reports" / "2026-04-13" / "summary.json").write_text("{}", encoding="utf-8")

    uploads: list[str] = []

    class FakeBlob:
        def __init__(self, name: str):
            self.name = name

        def upload_from_filename(self, filename: str) -> None:
            uploads.append(self.name)
            assert Path(filename).exists()

    class FakeBucket:
        def blob(self, name: str) -> FakeBlob:
            return FakeBlob(name)

    class FakeClient:
        def __init__(self, project=None):  # noqa: ANN001
            self.project = project

        def bucket(self, name: str) -> FakeBucket:  # noqa: ARG002
            return FakeBucket()

    fake_storage_module = types.SimpleNamespace(Client=FakeClient)
    fake_cloud_module = types.SimpleNamespace(storage=fake_storage_module)
    fake_google_module = types.ModuleType("google")
    fake_google_module.cloud = fake_cloud_module

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.cloud", fake_cloud_module)
    monkeypatch.setitem(sys.modules, "google.cloud.storage", fake_storage_module)

    result = publish_directory_to_gcs(
        local_root=publish_root,
        bucket_name="test-bucket",
        project_id="project-x",
    )
    assert result.bucket == "test-bucket"
    assert len(result.uploaded_objects) == 2
    assert sorted(result.uploaded_objects) == sorted(uploads)
