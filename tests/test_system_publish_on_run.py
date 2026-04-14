from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.schemas import RunMode
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_publish_on_run_exports_and_calls_publisher(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_PUBLISH_ON_RUN", "true")
    monkeypatch.setenv("TRADINGAGENTS_GCS_BUCKET", "unit-test-bucket")
    settings = load_settings()
    as_of = date(2026, 4, 13)

    histories = symbols_with_same_history(
        ["AAA"],
        make_price_history(as_of, periods=140, start_price=90, step=0.7, volume=6_000_000),
    )
    histories.update(
        symbols_with_same_history(
            settings.data.regime_proxies,
            make_price_history(as_of, periods=220, start_price=100, step=0.3, volume=8_000_000),
        )
    )
    provider = FakeMarketDataProvider(histories)
    repository = TradingRepository(settings.paths.database_path)
    runner = TradingSystemRunner(
        settings=settings,
        deterministic_research=True,
        repository=repository,
        provider=provider,
    )

    calls = {"count": 0}

    def fake_publish_directory_to_gcs(*, local_root, bucket_name, prefix="", project_id=None):  # noqa: ANN001
        calls["count"] += 1
        assert local_root.exists()
        assert bucket_name == "unit-test-bucket"

        class Result:
            def __init__(self):
                self.uploaded_objects = ["snapshots/latest_run_summary.json"]
                self.bucket = bucket_name
                self.prefix = prefix

        return Result()

    monkeypatch.setattr(
        "tradingagents.system.orchestration.runner.publish_directory_to_gcs",
        fake_publish_directory_to_gcs,
    )

    summary = runner.run_once(
        as_of_date=as_of,
        mode=RunMode.DRY_RUN,
        execute=False,
        symbols=["AAA"],
    )

    assert summary.status == "completed"
    assert calls["count"] == 1
    export_root = settings.paths.artifacts_dir / "publish"
    assert (export_root / settings.gcp.snapshots_prefix / "latest_positions.json").exists()
