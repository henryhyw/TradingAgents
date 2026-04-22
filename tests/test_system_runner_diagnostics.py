from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.research import ResearchAdapter
from tradingagents.system.schemas import ResearchDecision, SourceMetadata, TradeAction
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history


class FallbackSellAdapter(ResearchAdapter):
    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=TradeAction.SELL,
            confidence=0.35,
            thesis="Fallback-driven bearish memo.",
            risk_flags=["upstream_graph_failure"],
            invalidation_conditions=["n/a"],
            time_horizon="N/A",
            desired_position_fraction=0.0,
            source_metadata=SourceMetadata(
                research_adapter="unit_test",
                llm_provider="none",
                llm_model="none",
                parser_mode="upstream_error_no_entry",
                extra={
                    "upstream_retry_count": 2,
                    "upstream_failure_counts": {"ResourceExhausted": 1},
                    "upstream_fallback_mode": "research_error_no_entry",
                },
            ),
        )


def test_runner_emits_action_and_block_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    histories = {
        symbol: make_price_history(as_of, periods=220, start_price=100, step=0.4, volume=5_000_000)
        for symbol in settings.data.regime_proxies
    }
    histories["AAA"] = make_price_history(as_of, periods=180, start_price=120, step=0.6, volume=6_000_000)
    provider = FakeMarketDataProvider(histories)
    repository = TradingRepository(settings.paths.database_path)
    runner = TradingSystemRunner(
        settings=settings,
        deterministic_research=False,
        repository=repository,
        provider=provider,
        research_adapter=FallbackSellAdapter(),
    )
    summary = runner.run_once(as_of_date=as_of, execute=False, symbols=["AAA"])

    assert summary.research_action_counts.get("avoid") == 1
    assert summary.upstream_retry_count == 2
    assert summary.upstream_failure_counts.get("ResourceExhausted") == 1
    assert summary.block_reason_counts.get("upstream_fallback", 0) >= 1
    assert summary.block_reason_counts.get("no_entry", 0) >= 1
    assert isinstance(summary.blocked_buy_due_to_fallback_count, int)
    assert summary.fallback_origin_decision_count >= 1
    assert summary.promoted_buy_count == 0
    assert summary.flat_book_suppressed
