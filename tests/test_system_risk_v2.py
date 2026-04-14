from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.data import EarningsEvent, MarketBar
from tradingagents.system.risk import RiskEngine
from tradingagents.system.schemas import (
    CandidateAssessment,
    PortfolioSnapshot,
    RegimeLabel,
    RegimeSnapshot,
    ResearchDecision,
    SourceMetadata,
    TradeAction,
)


def _decision() -> ResearchDecision:
    return ResearchDecision(
        symbol="AAPL",
        as_of_date=date(2026, 4, 13),
        action=TradeAction.BUY,
        confidence=0.75,
        thesis="Risk-v2 test buy",
        risk_flags=[],
        invalidation_conditions=["invalidates"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.05,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )


def _bar(price: float = 150.0) -> MarketBar:
    return MarketBar(
        symbol="AAPL",
        date=date(2026, 4, 13),
        open=price,
        high=price,
        low=price,
        close=price,
        volume=3_000_000,
    )


def _portfolio() -> PortfolioSnapshot:
    return PortfolioSnapshot(
        as_of_date=date(2026, 4, 13),
        cash=100_000,
        equity=100_000,
        gross_exposure=10_000,
        positions=[],
    )


def _candidate() -> CandidateAssessment:
    return CandidateAssessment(
        symbol="AAPL",
        as_of_date=date(2026, 4, 13),
        name="Apple",
        asset_type="Equity",
        sector="Technology",
        eligible=True,
        close=150.0,
        avg_dollar_volume_20d=80_000_000,
        return_20d=0.05,
        return_60d=0.10,
        volatility_20d=0.25,
    )


def _regime() -> RegimeSnapshot:
    return RegimeSnapshot(
        as_of_date=date(2026, 4, 13),
        label=RegimeLabel.BALANCED,
        volatility_regime="normal",
        trend_regime="mixed",
        risk_on_score=0.1,
        risk_budget_multiplier=1.0,
        max_gross_exposure_fraction=0.3,
    )


def test_risk_engine_blocks_over_sector_exposure(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    engine = RiskEngine(settings)
    result = engine.evaluate(
        decision=_decision(),
        portfolio=_portfolio(),
        current_position=None,
        market_bar=_bar(),
        avg_dollar_volume_20d=80_000_000,
        earnings_event=EarningsEvent(symbol="AAPL"),
        daily_pnl_fraction=0.0,
        opening_trades_today=0,
        losing_exits_today=0,
        as_of_date=date(2026, 4, 13),
        candidate=_candidate(),
        regime=_regime(),
        sector_exposure_fraction=settings.risk.max_sector_exposure_fraction + 0.01,
    )
    assert not result.approved
    assert "sector_exposure_limit" in (result.rejection_reason or "")


def test_risk_engine_scales_down_when_correlation_high(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    engine = RiskEngine(settings)
    result = engine.evaluate(
        decision=_decision(),
        portfolio=_portfolio(),
        current_position=None,
        market_bar=_bar(),
        avg_dollar_volume_20d=80_000_000,
        earnings_event=EarningsEvent(symbol="AAPL"),
        daily_pnl_fraction=0.0,
        opening_trades_today=0,
        losing_exits_today=0,
        as_of_date=date(2026, 4, 13),
        candidate=_candidate(),
        regime=_regime(),
        sector_exposure_fraction=0.05,
        max_correlation_to_book=settings.risk.correlation_threshold + 0.01,
    )
    assert result.approved
    assert result.approved_size_fraction < 0.05
