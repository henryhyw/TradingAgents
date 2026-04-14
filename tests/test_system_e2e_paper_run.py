from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.schemas import RunMode
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_end_to_end_paper_run_with_deterministic_research(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_COMMISSION_PER_ORDER", "0")
    monkeypatch.setenv("TRADINGAGENTS_SLIPPAGE_BPS", "0")
    settings = load_settings()
    as_of = date(2026, 4, 13)
    histories = symbols_with_same_history(
        ["AAA"],
        make_price_history(as_of, periods=180, start_price=80, step=1.0, volume=7_000_000),
    )
    # Provide SPY so regime and relative-strength components have direct context.
    histories.update(
        symbols_with_same_history(
            ["SPY"],
            make_price_history(as_of, periods=180, start_price=100, step=0.4, volume=8_000_000),
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
    summary = runner.run_once(
        as_of_date=as_of,
        mode=RunMode.PAPER,
        execute=True,
        symbols=["AAA"],
    )
    assert summary.status == "completed"
    assert summary.orders_submitted >= 1
    assert summary.fills_completed >= 1
    positions = runner.broker.get_portfolio_snapshot(as_of).positions
    assert any(position.symbol == "AAA" and position.quantity > 0 for position in positions)
