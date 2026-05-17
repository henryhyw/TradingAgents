from __future__ import annotations

from datetime import date

from tradingagents.system.data import MarketBar
from tradingagents.system.portfolio import PortfolioService
from tradingagents.system.schemas import (
    ExecutionConstraints,
    OrderIntentType,
    PositionSnapshot,
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


def test_portfolio_service_generates_starter_entry_plan_and_intent():
    service = PortfolioService()
    as_of = date(2026, 5, 15)
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.BUY,
        confidence=0.55,
        thesis="Starter entry thesis",
        risk_flags=[],
        invalidation_conditions=["invalid"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.015,
        position_lifecycle_state=OrderIntentType.STARTER_ENTRY,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
            extra={"starter_entry_due_to_risk_on_bias": True},
        ),
    )
    risk = RiskDecision(
        source_decision_id=decision.decision_id,
        symbol="AAPL",
        as_of_date=as_of,
        approved=True,
        approved_size_fraction=0.015,
        execution_constraints=ExecutionConstraints(),
    )
    portfolio = PortfolioSnapshot(as_of_date=as_of, cash=100_000, equity=100_000, gross_exposure=0.0, positions=[])
    bar = MarketBar(symbol="AAPL", date=as_of, open=100, high=100, low=100, close=100, volume=1_000_000)

    fit = service.assess_portfolio_fit(decision, risk, portfolio, current_position=None, market_bar=bar)
    assert fit.fits_portfolio
    assert fit.recommended_action == OrderIntentType.STARTER_ENTRY
    assert fit.target_weight == 0.015

    plan = service.build_execution_plan(fit, decision, portfolio, market_bar=bar, current_position=None)
    assert plan.intent_type == OrderIntentType.STARTER_ENTRY
    assert plan.side is not None
    assert plan.quantity == 15

    intent = service.build_order_intent_from_plan(plan, decision, risk)
    assert intent is not None
    assert intent.intent_type == OrderIntentType.STARTER_ENTRY


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


def test_portfolio_service_maps_sell_lifecycle_to_trim_partial():
    service = PortfolioService()
    as_of = date(2026, 4, 13)
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.SELL,
        confidence=0.62,
        thesis="Trim winner.",
        risk_flags=[],
        invalidation_conditions=["invalid"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.0,
        position_lifecycle_state=OrderIntentType.TRIM_PARTIAL,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
            extra={"scale_out_fraction": 0.4, "exit_type": "regime_exit"},
        ),
    )
    risk = RiskDecision(
        source_decision_id=decision.decision_id,
        symbol="AAPL",
        as_of_date=as_of,
        approved=True,
        approved_size_fraction=0.0,
        execution_constraints=ExecutionConstraints(),
    )
    current_position = PositionSnapshot(
        symbol="AAPL",
        quantity=100,
        avg_cost=90.0,
        market_price=100.0,
        market_value=10000.0,
        cost_basis=9000.0,
        unrealized_pnl=1000.0,
    )
    portfolio = PortfolioSnapshot(as_of_date=as_of, cash=50_000, equity=100_000, gross_exposure=10_000, positions=[current_position])
    bar = MarketBar(symbol="AAPL", date=as_of, open=100, high=101, low=99, close=100, volume=1_500_000)
    fit = service.assess_portfolio_fit(decision, risk, portfolio, current_position=current_position, market_bar=bar)
    assert fit.fits_portfolio
    assert fit.recommended_action == OrderIntentType.TRIM_PARTIAL
    assert fit.target_weight < fit.current_weight


def test_portfolio_service_maps_sell_lifecycle_to_reduce_to_core():
    service = PortfolioService()
    as_of = date(2026, 4, 13)
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.SELL,
        confidence=0.66,
        thesis="Reduce to core.",
        risk_flags=[],
        invalidation_conditions=["invalid"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.02,
        position_lifecycle_state=OrderIntentType.REDUCE_TO_CORE,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
            extra={"reduce_to_core_target_fraction": 0.02, "exit_type": "take_profit_reduce_to_core"},
        ),
    )
    risk = RiskDecision(
        source_decision_id=decision.decision_id,
        symbol="AAPL",
        as_of_date=as_of,
        approved=True,
        approved_size_fraction=0.0,
        execution_constraints=ExecutionConstraints(),
    )
    current_position = PositionSnapshot(
        symbol="AAPL",
        quantity=100,
        avg_cost=90.0,
        market_price=100.0,
        market_value=10000.0,
        cost_basis=9000.0,
        unrealized_pnl=1000.0,
    )
    portfolio = PortfolioSnapshot(as_of_date=as_of, cash=50_000, equity=100_000, gross_exposure=10_000, positions=[current_position])
    bar = MarketBar(symbol="AAPL", date=as_of, open=100, high=101, low=99, close=100, volume=1_500_000)
    fit = service.assess_portfolio_fit(decision, risk, portfolio, current_position=current_position, market_bar=bar)
    assert fit.fits_portfolio
    assert fit.recommended_action == OrderIntentType.REDUCE_TO_CORE
    assert fit.target_weight == 0.02
