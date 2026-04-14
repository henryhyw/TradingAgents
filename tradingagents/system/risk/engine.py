from __future__ import annotations

from datetime import date

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import EarningsEvent, MarketBar
from tradingagents.system.schemas import (
    CandidateAssessment,
    ExecutionConstraints,
    PortfolioSnapshot,
    PositionSnapshot,
    RegimeSnapshot,
    ResearchDecision,
    RiskDecision,
    TradeAction,
)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


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
        candidate: CandidateAssessment | None = None,
        regime: RegimeSnapshot | None = None,
        sector_exposure_fraction: float = 0.0,
        max_correlation_to_book: float | None = None,
        cooldown_active: bool = False,
    ) -> RiskDecision:
        reasons: list[str] = []
        warnings: list[str] = []
        committee_notes: list[str] = []
        risk_checks: dict[str, float | str | bool] = {}
        current_fraction = 0.0

        if portfolio.equity > 0 and current_position is not None:
            current_fraction = current_position.market_value / portfolio.equity
        risk_checks["current_weight"] = current_fraction

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

        if decision.action == TradeAction.BUY and cooldown_active:
            reasons.append("symbol_cooldown_active")

        if decision.action == TradeAction.BUY and candidate is not None and candidate.watchlist_only:
            reasons.append("watchlist_only_candidate")

        if decision.action == TradeAction.BUY:
            if earnings_event.reliable and earnings_event.earnings_date is not None:
                days_to_earnings = abs((earnings_event.earnings_date - as_of_date).days)
                risk_checks["days_to_earnings"] = days_to_earnings
                if days_to_earnings <= self.settings.data.earnings_blackout_days:
                    reasons.append("earnings_blackout_window")
            else:
                warnings.append("earnings_signal_unavailable")

        if decision.action == TradeAction.BUY:
            risk_checks["sector_exposure_fraction"] = sector_exposure_fraction
            if sector_exposure_fraction >= self.settings.risk.max_sector_exposure_fraction:
                reasons.append("sector_exposure_limit")

        if decision.action == TradeAction.BUY and max_correlation_to_book is not None:
            risk_checks["max_correlation_to_book"] = max_correlation_to_book
            if max_correlation_to_book >= self.settings.risk.correlation_threshold + 0.05:
                reasons.append("correlation_too_high")
            elif max_correlation_to_book >= self.settings.risk.correlation_threshold:
                committee_notes.append("High correlation detected; size will be scaled down.")

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
                proposed_size_fraction=0.0,
                rejection_reason="; ".join(reasons),
                execution_constraints=constraints,
                committee_notes=committee_notes,
                risk_checks=risk_checks,
                warnings=warnings,
            )

        if decision.action == TradeAction.SELL:
            return RiskDecision(
                source_decision_id=decision.decision_id,
                symbol=decision.symbol,
                as_of_date=as_of_date,
                approved=True,
                approved_size_fraction=0.0,
                proposed_size_fraction=0.0,
                rejection_reason=None,
                execution_constraints=constraints,
                committee_notes=committee_notes + ["Sell decision approved for inventory reduction."],
                risk_checks=risk_checks,
                warnings=warnings,
            )

        equity = max(portfolio.equity, 1.0)
        gross_fraction = portfolio.gross_exposure / equity
        regime_gross_cap = (
            regime.max_gross_exposure_fraction if regime is not None else self.settings.risk.max_gross_exposure_fraction
        )
        remaining_gross_room = max(0.0, regime_gross_cap - gross_fraction)
        available_cash_fraction = max(
            0.0,
            (portfolio.cash - (self.settings.risk.minimum_cash_buffer_fraction * equity)) / equity,
        )
        risk_checks["gross_fraction"] = gross_fraction
        risk_checks["regime_gross_cap"] = regime_gross_cap
        risk_checks["remaining_gross_room"] = remaining_gross_room
        risk_checks["available_cash_fraction"] = available_cash_fraction

        raw_target = decision.desired_position_fraction
        if raw_target is None:
            raw_target = self.settings.risk.max_position_size_fraction * max(0.25, decision.confidence)
        risk_checks["raw_target_fraction"] = raw_target

        regime_multiplier = 1.0 if regime is None else regime.risk_budget_multiplier
        volatility = self.settings.risk.volatility_target_annual
        if candidate is not None and candidate.volatility_20d > 0:
            volatility = candidate.volatility_20d
        volatility = _clamp(
            volatility,
            self.settings.risk.volatility_floor_annual,
            self.settings.risk.volatility_ceiling_annual,
        )
        vol_scale = _clamp(
            self.settings.risk.volatility_target_annual / volatility,
            0.35,
            1.25,
        )
        correlation_scale = 1.0
        if max_correlation_to_book is not None and max_correlation_to_book >= self.settings.risk.correlation_threshold:
            correlation_scale = self.settings.risk.high_correlation_scale
        sector_room = max(0.0, self.settings.risk.max_sector_exposure_fraction - sector_exposure_fraction)

        proposed_target = raw_target * regime_multiplier * vol_scale * correlation_scale
        proposed_target = min(
            proposed_target,
            self.settings.risk.max_position_size_fraction,
            current_fraction + remaining_gross_room,
            current_fraction + available_cash_fraction,
            current_fraction + max(0.0, sector_room),
        )
        risk_checks["regime_multiplier"] = regime_multiplier
        risk_checks["volatility_input"] = volatility
        risk_checks["volatility_scale"] = vol_scale
        risk_checks["correlation_scale"] = correlation_scale
        risk_checks["sector_room"] = sector_room
        risk_checks["proposed_target"] = proposed_target

        if regime is not None:
            committee_notes.append(
                f"Regime {regime.label.value} applied with risk budget multiplier {regime.risk_budget_multiplier:.2f}."
            )
        if volatility > self.settings.risk.volatility_target_annual:
            committee_notes.append("Volatility-aware sizing reduced target weight.")

        if proposed_target <= current_fraction + self.settings.run.portfolio_rebalance_buffer:
            return RiskDecision(
                source_decision_id=decision.decision_id,
                symbol=decision.symbol,
                as_of_date=as_of_date,
                approved=False,
                approved_size_fraction=0.0,
                proposed_size_fraction=proposed_target,
                rejection_reason="no_incremental_capacity_available",
                execution_constraints=constraints,
                committee_notes=committee_notes,
                risk_checks=risk_checks,
                warnings=warnings,
            )

        return RiskDecision(
            source_decision_id=decision.decision_id,
            symbol=decision.symbol,
            as_of_date=as_of_date,
            approved=True,
            approved_size_fraction=proposed_target,
            proposed_size_fraction=proposed_target,
            rejection_reason=None,
            execution_constraints=constraints,
            committee_notes=committee_notes,
            risk_checks=risk_checks,
            warnings=warnings,
        )
