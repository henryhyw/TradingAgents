from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.research import ResearchAdapter
from tradingagents.system.schemas import ResearchDecision, SourceMetadata, TradeAction
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history


class CountingResearchAdapter(ResearchAdapter):
    def __init__(self):
        self.calls: list[str] = []

    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        self.calls.append(symbol)
        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=TradeAction.HOLD,
            confidence=0.4,
            thesis="counting adapter",
            risk_flags=[],
            invalidation_conditions=["n/a"],
            time_horizon="1-4 weeks",
            source_metadata=SourceMetadata(
                research_adapter="counting_test",
                llm_provider="none",
                llm_model="none",
                parser_mode="deterministic",
            ),
        )


def _proxy_histories(settings, as_of: date):
    histories = {}
    for symbol in settings.data.regime_proxies:
        histories[symbol] = make_price_history(
            as_of,
            periods=220,
            start_price=100.0,
            step=0.4,
            volume=5_000_000,
        )
    return histories


def test_critical_symbol_missing_skips_research_call(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_MIN_SHORTLIST_DATA_COVERAGE", "0.0")
    settings = load_settings()
    as_of = date(2026, 4, 13)
    histories = _proxy_histories(settings, as_of)
    histories["NVDA"] = make_price_history(as_of, periods=180, start_price=120, step=0.9, volume=6_000_000)
    provider = FakeMarketDataProvider(histories)
    repository = TradingRepository(settings.paths.database_path)
    adapter = CountingResearchAdapter()
    runner = TradingSystemRunner(
        settings=settings,
        deterministic_research=False,
        repository=repository,
        provider=provider,
        research_adapter=adapter,
    )
    summary = runner.run_once(
        as_of_date=as_of,
        execute=False,
        symbols=["AAA", "NVDA"],  # AAA has no history and must be skipped before research.
    )
    assert "AAA" in summary.rejected_symbols
    assert "critical_history_missing" in summary.rejected_symbols["AAA"]
    assert "AAA" not in adapter.calls


def test_degraded_regime_proxy_set_aborts_live_research(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 13)
    histories = {
        "SPY": make_price_history(as_of, periods=220, start_price=100, step=0.3, volume=8_000_000),
        "NVDA": make_price_history(as_of, periods=180, start_price=120, step=0.8, volume=6_000_000),
    }
    provider = FakeMarketDataProvider(histories)
    repository = TradingRepository(settings.paths.database_path)
    adapter = CountingResearchAdapter()
    runner = TradingSystemRunner(
        settings=settings,
        deterministic_research=False,
        repository=repository,
        provider=provider,
        research_adapter=adapter,
    )
    try:
        runner.run_once(as_of_date=as_of, execute=False, symbols=["NVDA"])
    except RuntimeError as exc:
        assert "regime data quality" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected run_once to abort when regime proxy coverage is degraded")
    assert adapter.calls == []
