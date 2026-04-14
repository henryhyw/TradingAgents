from __future__ import annotations

import math

from tradingagents.system.data import MarketBar
from tradingagents.system.schemas import OrderIntent, OrderSide, PortfolioSnapshot, PositionSnapshot, ResearchDecision, RiskDecision, TradeAction


class PortfolioService:
    def build_order_intent(
        self,
        decision: ResearchDecision,
        risk_decision: RiskDecision,
        portfolio: PortfolioSnapshot,
        current_position: PositionSnapshot | None,
        market_bar: MarketBar | None,
    ) -> OrderIntent | None:
        if not risk_decision.approved or market_bar is None:
            return None

        if decision.action == TradeAction.HOLD:
            return None

        if decision.action == TradeAction.SELL:
            if current_position is None or current_position.quantity <= 0:
                return None
            return OrderIntent(
                as_of_date=risk_decision.as_of_date,
                symbol=decision.symbol,
                side=OrderSide.SELL,
                quantity=current_position.quantity,
                source_decision_id=decision.decision_id,
                source_risk_decision_id=risk_decision.risk_decision_id,
                notes=["full_exit"],
            )

        target_value = risk_decision.approved_size_fraction * portfolio.equity
        current_value = current_position.market_value if current_position else 0.0
        incremental_value = max(0.0, target_value - current_value)
        quantity = math.floor(incremental_value / market_bar.close)
        if quantity <= 0:
            return None

        return OrderIntent(
            as_of_date=risk_decision.as_of_date,
            symbol=decision.symbol,
            side=OrderSide.BUY,
            quantity=quantity,
            source_decision_id=decision.decision_id,
            source_risk_decision_id=risk_decision.risk_decision_id,
            notes=[f"target_fraction={risk_decision.approved_size_fraction:.4f}"],
        )
