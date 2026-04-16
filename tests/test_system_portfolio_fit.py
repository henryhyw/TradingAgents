from __future__ import annotations

from datetime import date

from tradingagents.system.data import MarketBar
from tradingagents.system.portfolio import PortfolioService
from tradingagents.system.schemas import (
    ExecutionConstraints,
    OrderIntentType,
    PortfolioSnapshot,
    ResearchDecision,
    RiskDecision,
    SourceMetadata,
    TradeAction,
)


def test_portfolio_service_generates_new_entry_plan_and_intent():
    service = PortfolioService()
    as_of = date(2026, 4, 13)
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.BUY,
        confidence=0.7,
        thesis="Test thesis",
        risk_flags=[],
        invalidation_conditions=["invalid"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.03,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )
    risk = RiskDecision(
        source_decision_id=decision.decision_id,
        symbol="AAPL",
        as_of_date=as_of,
        approved=True,
        approved_size_fraction=0.03,
        execution_constraints=ExecutionConstraints(),
    )
    portfolio = PortfolioSnapshot(as_of_date=as_of, cash=100_000, equity=100_000, gross_exposure=0.0, positions=[])
    bar = MarketBar(symbol="AAPL", date=as_of, open=100, high=100, low=100, close=100, volume=1_000_000)

    fit = service.assess_portfolio_fit(decision, risk, portfolio, current_position=None, market_bar=bar)
    assert fit.fits_portfolio
    assert fit.recommended_action == OrderIntentType.NEW_ENTRY

    plan = service.build_execution_plan(fit, decision, portfolio, market_bar=bar, current_position=None)
    assert plan.side is not None
    assert plan.quantity is not None and plan.quantity > 0

    intent = service.build_order_intent_from_plan(plan, decision, risk)
    assert intent is not None
    assert intent.intent_type == OrderIntentType.NEW_ENTRY


def test_portfolio_service_maps_avoid_to_non_actionable_no_entry():
    service = PortfolioService()
    as_of = date(2026, 4, 13)
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.AVOID,
        confidence=0.45,
        thesis="No-entry test",
        risk_flags=[],
        invalidation_conditions=["invalid"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.0,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )
    risk = RiskDecision(
        source_decision_id=decision.decision_id,
        symbol="AAPL",
        as_of_date=as_of,
        approved=False,
        approved_size_fraction=0.0,
        rejection_reason="avoid_signal_no_entry",
        execution_constraints=ExecutionConstraints(),
    )
    portfolio = PortfolioSnapshot(as_of_date=as_of, cash=100_000, equity=100_000, gross_exposure=0.0, positions=[])
    bar = MarketBar(symbol="AAPL", date=as_of, open=100, high=100, low=100, close=100, volume=1_000_000)
    fit = service.assess_portfolio_fit(decision, risk, portfolio, current_position=None, market_bar=bar)
    assert not fit.fits_portfolio
    assert fit.recommended_action == OrderIntentType.AVOID
