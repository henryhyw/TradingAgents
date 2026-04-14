from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.schemas import RunMode
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_end_to_end_dry_run_with_deterministic_research(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 13)
    histories = symbols_with_same_history(
        ["AAA", "BBB"],
        make_price_history(as_of, periods=140, start_price=80, step=0.6, volume=4_000_000),
    )
    provider = FakeMarketDataProvider(histories)
    repository = TradingRepository(settings.paths.database_path)
    runner = TradingSystemRunner(
        settings=settings,
        deterministic_research=True,
        repository=repository,
        provider=provider,
    )
    summary = runner.run_once(
        as_of_date=as_of,
        mode=RunMode.DRY_RUN,
        execute=False,
        symbols=["AAA", "BBB"],
    )
    assert summary.status == "completed"
    assert len(summary.shortlisted_symbols) == 2
    assert summary.orders_submitted == 0
    assert summary.report_path is not None
    assert settings.paths.reports_dir.joinpath(as_of.isoformat(), "summary.md").exists()
    counts = repository.dump_table_counts()
    assert counts["research_decisions"] >= 2
    assert counts["risk_decisions"] >= 2
    assert counts["daily_run_summaries"] >= 1
