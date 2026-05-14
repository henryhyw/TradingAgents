from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from tradingagents.system.schemas import (
    DailyRunSummary,
    EntryMode,
    ExecutionConstraints,
    OrderIntent,
    OrderIntentType,
    OrderSide,
    ResearchDecision,
    RiskDecision,
    RunMode,
    SourceMetadata,
    TradeAction,
    utc_now,
)


def test_research_decision_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        ResearchDecision(
            symbol="AAPL",
            as_of_date=date(2026, 4, 13),
            action=TradeAction.BUY,
            confidence=1.2,
            thesis="Invalid confidence should fail validation.",
            risk_flags=[],
            invalidation_conditions=[],
            time_horizon="1-4 weeks",
            source_metadata=SourceMetadata(
                research_adapter="unit_test",
                llm_provider="none",
                llm_model="none",
                parser_mode="deterministic",
            ),
        )


def test_order_intent_requires_positive_quantity():
    with pytest.raises(ValidationError):
        OrderIntent(
            as_of_date=date(2026, 4, 13),
            symbol="AAPL",
            side=OrderSide.BUY,
            quantity=0,
            source_decision_id="rd_test",
            source_risk_decision_id="rk_test",
        )


def test_risk_decision_approved_size_fraction_is_bounded():
    with pytest.raises(ValidationError):
        RiskDecision(
            source_decision_id="rd_test",
            symbol="AAPL",
            as_of_date=date(2026, 4, 13),
            approved=True,
            approved_size_fraction=1.1,
            execution_constraints=ExecutionConstraints(),
        )


def test_research_decision_defaults_new_strategy_fields():
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=date(2026, 4, 13),
        action=TradeAction.HOLD,
        confidence=0.5,
        thesis="Neutral",
        risk_flags=[],
        invalidation_conditions=[],
        time_horizon="1-4 weeks",
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )
    assert decision.entry_mode == EntryMode.NONE
    assert decision.position_lifecycle_state is None


def test_order_intent_accepts_reduce_to_core_type():
    intent = OrderIntent(
        as_of_date=date(2026, 4, 13),
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=5,
        intent_type=OrderIntentType.REDUCE_TO_CORE,
        source_decision_id="rd_test",
        source_risk_decision_id="rk_test",
    )
    assert intent.intent_type == OrderIntentType.REDUCE_TO_CORE


def test_daily_run_summary_defaults_strategy_balance_diagnostics():
    summary = DailyRunSummary(
        as_of_date=date(2026, 5, 13),
        mode=RunMode.DRY_RUN,
        started_at=utc_now(),
        completed_at=utc_now(),
        status="completed",
    )

    assert summary.buy_near_miss_count == 0
    assert summary.buy_near_miss_due_to_breakout_confirmation == 0
    assert summary.buy_near_miss_due_to_pullback_confirmation == 0
    assert summary.risk_on_participation_bias_applied_count == 0
    assert summary.full_exit_due_to_risk_reduction_count == 0
    assert summary.full_exit_rejected_in_favor_of_trim_count == 0
    assert summary.full_exit_rejected_in_favor_of_reduce_to_core_count == 0
    assert summary.starter_position_kept_due_to_regime_count == 0
    assert summary.went_flat_in_risk_on_count == 0
    assert summary.risk_on_flattening_justification_count == 0
