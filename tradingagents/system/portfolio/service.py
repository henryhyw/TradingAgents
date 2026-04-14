from __future__ import annotations

import math

from tradingagents.system.data import MarketBar
from tradingagents.system.schemas import (
    CandidateAssessment,
    ExecutionPlan,
    OrderIntent,
    OrderIntentType,
    OrderSide,
    PortfolioFitAssessment,
    PortfolioSnapshot,
    PositionSnapshot,
    RegimeSnapshot,
    ResearchDecision,
    RiskDecision,
    TradeAction,
)


class PortfolioService:
    @staticmethod
    def _position_weight(portfolio: PortfolioSnapshot, position: PositionSnapshot | None) -> float:
        if position is None or portfolio.equity <= 0:
            return 0.0
        return max(0.0, position.market_value / portfolio.equity)

    def assess_portfolio_fit(
        self,
        decision: ResearchDecision,
        risk_decision: RiskDecision,
        portfolio: PortfolioSnapshot,
        current_position: PositionSnapshot | None,
        market_bar: MarketBar | None,
        candidate: CandidateAssessment | None = None,
        regime: RegimeSnapshot | None = None,
        max_correlation_to_book: float | None = None,
    ) -> PortfolioFitAssessment:
        current_weight = self._position_weight(portfolio, current_position)
        conflicts: list[str] = []
        warnings: list[str] = []
        rationale = "No trade."
        target_weight = current_weight
        recommended_action = OrderIntentType.HOLD
        fits = False

        if market_bar is None:
            conflicts.append("missing_market_data")
            return PortfolioFitAssessment(
                symbol=decision.symbol,
                as_of_date=risk_decision.as_of_date,
                fits_portfolio=False,
                recommended_action=OrderIntentType.HOLD,
                current_weight=current_weight,
                target_weight=current_weight,
                rationale="Cannot form execution plan without market price.",
                conflicts=conflicts,
                warnings=warnings,
            )

        if not risk_decision.approved:
            conflicts.append(risk_decision.rejection_reason or "risk_rejected")
            return PortfolioFitAssessment(
                symbol=decision.symbol,
                as_of_date=risk_decision.as_of_date,
                fits_portfolio=False,
                recommended_action=OrderIntentType.HOLD,
                current_weight=current_weight,
                target_weight=current_weight,
                rationale="Risk committee rejected this idea.",
                conflicts=conflicts,
                warnings=warnings,
            )

        if decision.action == TradeAction.SELL:
            if current_position is None or current_position.quantity <= 0:
                conflicts.append("no_inventory_to_reduce")
                return PortfolioFitAssessment(
                    symbol=decision.symbol,
                    as_of_date=risk_decision.as_of_date,
                    fits_portfolio=False,
                    recommended_action=OrderIntentType.HOLD,
                    current_weight=current_weight,
                    target_weight=current_weight,
                    rationale="Sell recommendation exists but there is no long inventory.",
                    conflicts=conflicts,
                    warnings=warnings,
                )
            fits = True
            target_weight = 0.0
            recommended_action = OrderIntentType.EXIT
            rationale = "Exit aligns with risk-approved inventory reduction."
        elif decision.action == TradeAction.BUY:
            target_weight = max(current_weight, risk_decision.approved_size_fraction)
            buffer = 0.0015
            fits = True
            if current_weight <= buffer:
                recommended_action = OrderIntentType.NEW_ENTRY
                rationale = "Initiating new position within approved risk budget."
            elif target_weight > current_weight + buffer:
                recommended_action = OrderIntentType.ADD
                rationale = "Adding to existing position toward approved target weight."
            elif target_weight < current_weight - buffer:
                recommended_action = OrderIntentType.TRIM
                rationale = "Reducing position to align with target risk budget."
            else:
                recommended_action = OrderIntentType.HOLD
                fits = False
                rationale = "Current position is already close to target weight."
        else:
            rationale = "Hold action keeps current allocation unchanged."
            fits = False
            recommended_action = OrderIntentType.HOLD

        if regime is not None:
            warnings.append(f"regime_context:{regime.label.value}")
        if candidate is not None and candidate.watchlist_only:
            conflicts.append("watchlist_only_candidate")
        if max_correlation_to_book is not None and max_correlation_to_book >= 0.9:
            warnings.append("high_overlap_with_existing_positions")

        return PortfolioFitAssessment(
            symbol=decision.symbol,
            as_of_date=risk_decision.as_of_date,
            fits_portfolio=fits,
            recommended_action=recommended_action,
            current_weight=current_weight,
            target_weight=target_weight,
            rationale=rationale,
            conflicts=conflicts,
            warnings=warnings,
        )

    def build_execution_plan(
        self,
        fit: PortfolioFitAssessment,
        decision: ResearchDecision,
        portfolio: PortfolioSnapshot,
        market_bar: MarketBar | None,
        current_position: PositionSnapshot | None,
    ) -> ExecutionPlan:
        side = None
        quantity = None
        notes = [fit.rationale]

        if not fit.fits_portfolio or market_bar is None:
            return ExecutionPlan(
                symbol=fit.symbol,
                as_of_date=fit.as_of_date,
                intent_type=OrderIntentType.HOLD,
                side=None,
                target_weight=fit.target_weight,
                quantity=None,
                notes=notes + fit.conflicts,
            )

        current_qty = 0 if current_position is None else current_position.quantity
        current_value = 0.0 if current_position is None else current_position.market_value
        target_value = fit.target_weight * portfolio.equity

        if fit.recommended_action in {OrderIntentType.NEW_ENTRY, OrderIntentType.ADD}:
            side = OrderSide.BUY
            delta_value = max(0.0, target_value - current_value)
            quantity = math.floor(delta_value / market_bar.close)
        elif fit.recommended_action in {OrderIntentType.TRIM, OrderIntentType.EXIT}:
            side = OrderSide.SELL
            if fit.recommended_action == OrderIntentType.EXIT:
                quantity = current_qty
            else:
                delta_value = max(0.0, current_value - target_value)
                quantity = min(current_qty, math.ceil(delta_value / market_bar.close))

        if quantity is not None and quantity <= 0:
            quantity = None
            side = None
            notes.append("quantity_below_one_share")

        return ExecutionPlan(
            symbol=fit.symbol,
            as_of_date=fit.as_of_date,
            intent_type=fit.recommended_action,
            side=side,
            target_weight=fit.target_weight,
            quantity=quantity,
            notes=notes + fit.warnings,
        )

    def build_order_intent_from_plan(
        self,
        plan: ExecutionPlan,
        decision: ResearchDecision,
        risk_decision: RiskDecision,
    ) -> OrderIntent | None:
        if plan.side is None or plan.quantity is None or plan.quantity <= 0:
            return None
        return OrderIntent(
            as_of_date=risk_decision.as_of_date,
            symbol=decision.symbol,
            side=plan.side,
            quantity=plan.quantity,
            intent_type=plan.intent_type,
            target_position_fraction=plan.target_weight,
            source_decision_id=decision.decision_id,
            source_risk_decision_id=risk_decision.risk_decision_id,
            notes=plan.notes,
        )

    def build_order_intent(
        self,
        decision: ResearchDecision,
        risk_decision: RiskDecision,
        portfolio: PortfolioSnapshot,
        current_position: PositionSnapshot | None,
        market_bar: MarketBar | None,
        candidate: CandidateAssessment | None = None,
        regime: RegimeSnapshot | None = None,
        max_correlation_to_book: float | None = None,
    ) -> OrderIntent | None:
        fit = self.assess_portfolio_fit(
            decision=decision,
            risk_decision=risk_decision,
            portfolio=portfolio,
            current_position=current_position,
            market_bar=market_bar,
            candidate=candidate,
            regime=regime,
            max_correlation_to_book=max_correlation_to_book,
        )
        plan = self.build_execution_plan(
            fit=fit,
            decision=decision,
            portfolio=portfolio,
            market_bar=market_bar,
            current_position=current_position,
        )
        return self.build_order_intent_from_plan(plan, decision, risk_decision)
