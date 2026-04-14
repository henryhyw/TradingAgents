from __future__ import annotations

from datetime import date

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import EarningsEvent, MarketBar
from tradingagents.system.schemas import ExecutionConstraints, PortfolioSnapshot, PositionSnapshot, ResearchDecision, RiskDecision, TradeAction


class RiskEngine:
    def __init__(self, settings: SystemSettings):
        self.settings = settings

    def evaluate(
        self,
        decision: ResearchDecision,
        portfolio: PortfolioSnapshot,
        current_position: PositionSnapshot | None,
        market_bar: MarketBar | None,
        avg_dollar_volume_20d: float,
        earnings_event: EarningsEvent,
        daily_pnl_fraction: float,
        opening_trades_today: int,
        losing_exits_today: int,
        as_of_date: date,
    ) -> RiskDecision:
        reasons: list[str] = []
        warnings: list[str] = []
        current_fraction = 0.0

        if portfolio.equity > 0 and current_position is not None:
            current_fraction = current_position.market_value / portfolio.equity

        if decision.action == TradeAction.HOLD:
            reasons.append("hold_signal_no_order")

        if decision.action == TradeAction.SELL and (current_position is None or current_position.quantity <= 0):
            reasons.append("no_long_position_to_exit")

        if market_bar is None:
            reasons.append("missing_market_data")
        elif decision.action == TradeAction.BUY and market_bar.close < self.settings.data.min_price:
            reasons.append("price_below_minimum")

        if decision.action == TradeAction.BUY and avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
            reasons.append("liquidity_below_minimum")

        if decision.action == TradeAction.BUY and daily_pnl_fraction <= -self.settings.risk.daily_loss_limit_fraction:
            reasons.append("daily_loss_limit_breached")

        if decision.action == TradeAction.BUY and losing_exits_today >= self.settings.risk.stop_opening_after_losing_exits:
            reasons.append("too_many_losing_exits_today")

        if decision.action == TradeAction.BUY and opening_trades_today >= self.settings.risk.max_new_opening_trades_per_symbol_per_day:
            reasons.append("opening_trade_limit_reached")

        if decision.action == TradeAction.BUY:
            if earnings_event.reliable and earnings_event.earnings_date is not None:
                days_to_earnings = abs((earnings_event.earnings_date - as_of_date).days)
                if days_to_earnings <= self.settings.data.earnings_blackout_days:
                    reasons.append("earnings_blackout_window")
            else:
                warnings.append("earnings_signal_unavailable")

        constraints = ExecutionConstraints(
            regular_session_only=True,
            fill_model=self.settings.paper.fill_model,
            max_slippage_bps=self.settings.paper.slippage_bps,
            latest_acceptable_trade_date=as_of_date,
            notes=[],
        )

        if reasons:
            return RiskDecision(
                source_decision_id=decision.decision_id,
                symbol=decision.symbol,
                as_of_date=as_of_date,
                approved=False,
                approved_size_fraction=0.0,
                rejection_reason="; ".join(reasons),
                execution_constraints=constraints,
                warnings=warnings,
            )

        if decision.action == TradeAction.SELL:
            return RiskDecision(
                source_decision_id=decision.decision_id,
                symbol=decision.symbol,
                as_of_date=as_of_date,
                approved=True,
                approved_size_fraction=0.0,
                rejection_reason=None,
                execution_constraints=constraints,
                warnings=warnings,
            )

        equity = max(portfolio.equity, 1.0)
        gross_fraction = portfolio.gross_exposure / equity if equity else 0.0
        remaining_gross_room = max(0.0, self.settings.risk.max_gross_exposure_fraction - gross_fraction)
        available_cash_fraction = max(
            0.0,
            (portfolio.cash - (self.settings.risk.minimum_cash_buffer_fraction * equity)) / equity,
        )
        raw_target = decision.desired_position_fraction
        if raw_target is None:
            raw_target = self.settings.risk.max_position_size_fraction * max(0.25, decision.confidence)

        max_target_fraction = min(
            self.settings.risk.max_position_size_fraction,
            current_fraction + remaining_gross_room,
            current_fraction + available_cash_fraction,
        )
        approved_target = min(raw_target, max_target_fraction)

        if approved_target <= current_fraction:
            return RiskDecision(
                source_decision_id=decision.decision_id,
                symbol=decision.symbol,
                as_of_date=as_of_date,
                approved=False,
                approved_size_fraction=0.0,
                rejection_reason="no_incremental_capacity_available",
                execution_constraints=constraints,
                warnings=warnings,
            )

        return RiskDecision(
            source_decision_id=decision.decision_id,
            symbol=decision.symbol,
            as_of_date=as_of_date,
            approved=True,
            approved_size_fraction=approved_target,
            rejection_reason=None,
            execution_constraints=constraints,
            warnings=warnings,
        )
