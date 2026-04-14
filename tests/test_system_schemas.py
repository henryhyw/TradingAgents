from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from tradingagents.system.schemas import (
    ExecutionConstraints,
    OrderIntent,
    OrderSide,
    ResearchDecision,
    RiskDecision,
    SourceMetadata,
    TradeAction,
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
