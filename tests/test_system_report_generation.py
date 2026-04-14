from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.schemas import RunMode
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_generate_report_from_storage_uses_persisted_symbols(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 13)
    histories = symbols_with_same_history(
        ["AAA"],
        make_price_history(as_of, periods=120, start_price=90, step=0.4, volume=3_000_000),
    )
    provider = FakeMarketDataProvider(histories)
    repository = TradingRepository(settings.paths.database_path)
    runner = TradingSystemRunner(
        settings=settings,
        deterministic_research=True,
        repository=repository,
        provider=provider,
    )
    runner.run_once(
        as_of_date=as_of,
        mode=RunMode.DRY_RUN,
        execute=False,
        symbols=["AAA"],
    )
    report_path = runner.generate_report_from_storage(as_of)
    assert report_path is not None
    assert settings.paths.reports_dir.joinpath(as_of.isoformat(), "summary.md").exists()
