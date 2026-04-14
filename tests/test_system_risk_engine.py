from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.data import EarningsEvent, MarketBar
from tradingagents.system.risk import RiskEngine
from tradingagents.system.schemas import (
    PortfolioSnapshot,
    PositionSnapshot,
    ResearchDecision,
    SourceMetadata,
    TradeAction,
)


def _settings(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_COMMISSION_PER_ORDER", "0")
    monkeypatch.setenv("TRADINGAGENTS_SLIPPAGE_BPS", "0")
    return load_settings()


def _decision(action: TradeAction) -> ResearchDecision:
    return ResearchDecision(
        symbol="AAPL",
        as_of_date=date(2026, 4, 13),
        action=action,
        confidence=0.7,
        thesis="Unit test thesis",
        risk_flags=[],
        invalidation_conditions=["invalidates"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.05 if action == TradeAction.BUY else 0.0,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )


def _portfolio(equity: float = 100_000, cash: float = 100_000, gross: float = 0) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_date=date(2026, 4, 13),
        cash=cash,
        equity=equity,
        gross_exposure=gross,
        positions=[],
    )


def _bar(price: float = 120.0) -> MarketBar:
    return MarketBar(
        symbol="AAPL",
        date=date(2026, 4, 13),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=5_000_000,
    )


def test_risk_engine_rejects_buy_when_daily_loss_limit_breached(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    engine = RiskEngine(settings)
    result = engine.evaluate(
        decision=_decision(TradeAction.BUY),
        portfolio=_portfolio(),
        current_position=None,
        market_bar=_bar(),
        avg_dollar_volume_20d=50_000_000,
        earnings_event=EarningsEvent(symbol="AAPL", earnings_date=None, reliable=False),
        daily_pnl_fraction=-0.03,
        opening_trades_today=0,
        losing_exits_today=0,
        as_of_date=date(2026, 4, 13),
    )
    assert not result.approved
    assert "daily_loss_limit_breached" in (result.rejection_reason or "")


def test_risk_engine_rejects_sell_when_no_existing_long(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    engine = RiskEngine(settings)
    result = engine.evaluate(
        decision=_decision(TradeAction.SELL),
        portfolio=_portfolio(),
        current_position=None,
        market_bar=_bar(),
        avg_dollar_volume_20d=50_000_000,
        earnings_event=EarningsEvent(symbol="AAPL", earnings_date=None, reliable=False),
        daily_pnl_fraction=0.0,
        opening_trades_today=0,
        losing_exits_today=0,
        as_of_date=date(2026, 4, 13),
    )
    assert not result.approved
    assert "no_long_position_to_exit" in (result.rejection_reason or "")


def test_risk_engine_caps_position_size_by_config(monkeypatch, tmp_path):
    settings = _settings(monkeypatch, tmp_path)
    engine = RiskEngine(settings)
    result = engine.evaluate(
        decision=_decision(TradeAction.BUY),
        portfolio=_portfolio(),
        current_position=PositionSnapshot(
            symbol="AAPL",
            quantity=0,
            avg_cost=0,
            market_price=120,
            market_value=0,
            cost_basis=0,
            unrealized_pnl=0,
            realized_pnl_day=0,
        ),
        market_bar=_bar(120),
        avg_dollar_volume_20d=50_000_000,
        earnings_event=EarningsEvent(symbol="AAPL", earnings_date=None, reliable=False),
        daily_pnl_fraction=0.0,
        opening_trades_today=0,
        losing_exits_today=0,
        as_of_date=date(2026, 4, 13),
    )
    assert result.approved
    assert result.approved_size_fraction <= settings.risk.max_position_size_fraction
