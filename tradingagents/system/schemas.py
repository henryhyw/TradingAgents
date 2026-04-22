from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def make_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TradeAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    AVOID = "avoid"


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"


class TimeInForce(str, Enum):
    DAY = "day"


class OrderStatus(str, Enum):
    NEW = "new"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    PENDING = "pending"


class RunMode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    DAILY = "daily"
    REPLAY = "replay"


class RegimeLabel(str, Enum):
    RISK_ON = "risk_on"
    BALANCED = "balanced"
    RISK_OFF = "risk_off"
    HIGH_VOLATILITY = "high_volatility"


class OrderIntentType(str, Enum):
    NEW_ENTRY = "new_entry"
    ADD = "add"
    TRIM = "trim"
    EXIT = "exit"
    HOLD = "hold"
    AVOID = "avoid"


class SourceMetadata(StrictModel):
    research_adapter: str
    llm_provider: str
    llm_model: str
    parser_mode: str
    upstream_rating: str | None = None
    upstream_artifact_path: str | None = None
    notes: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class ExecutionConstraints(StrictModel):
    regular_session_only: bool = True
    fill_model: str = "same_bar_close"
    max_slippage_bps: float = 10.0
    latest_acceptable_trade_date: date | None = None
    notes: list[str] = Field(default_factory=list)


class RegimeSnapshot(StrictModel):
    regime_snapshot_id: str = Field(default_factory=lambda: make_id("rg"))
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    label: RegimeLabel
    volatility_regime: str
    trend_regime: str
    risk_on_score: float
    risk_budget_multiplier: float
    max_gross_exposure_fraction: float
    signals: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data_quality: str = "ok"

    @field_validator("risk_on_score")
    @classmethod
    def _validate_risk_score(cls, value: float) -> float:
        return max(-1.0, min(1.0, value))

    @field_validator("risk_budget_multiplier")
    @classmethod
    def _validate_multiplier(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("risk_budget_multiplier must be positive")
        return value

    @field_validator("max_gross_exposure_fraction")
    @classmethod
    def _validate_gross_cap(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("max_gross_exposure_fraction must be between 0 and 1")
        return value


class CandidateAssessment(StrictModel):
    candidate_id: str = Field(default_factory=lambda: make_id("ca"))
    symbol: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    name: str
    asset_type: str
    sector: str
    style_tags: list[str] = Field(default_factory=list)
    benchmark_symbol: str | None = None
    peer_group: str | None = None
    eligible: bool
    watchlist_only: bool = False
    eligibility_reasons: list[str] = Field(default_factory=list)
    close: float
    avg_dollar_volume_20d: float
    return_20d: float
    return_60d: float
    volatility_20d: float
    relative_strength_20d: float = 0.0
    regime_fit_score: float = 0.0
    ranking_score: float = 0.0
    ranking_breakdown: dict[str, float] = Field(default_factory=dict)
    shortlist_reason: str | None = None
    data_quality_warnings: list[str] = Field(default_factory=list)


class AnalystMemo(StrictModel):
    memo_id: str = Field(default_factory=lambda: make_id("am"))
    symbol: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    role: str
    signal: Literal["bullish", "bearish", "neutral", "mixed"] = "neutral"
    confidence: float = 0.5
    summary: str
    evidence: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value


class BullCaseMemo(StrictModel):
    memo_id: str = Field(default_factory=lambda: make_id("bc"))
    symbol: str
    as_of_date: date
    summary: str
    catalysts: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    conviction: float = 0.5

    @field_validator("conviction")
    @classmethod
    def _validate_conviction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("conviction must be between 0 and 1")
        return value


class BearCaseMemo(StrictModel):
    memo_id: str = Field(default_factory=lambda: make_id("br"))
    symbol: str
    as_of_date: date
    summary: str
    risks: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    conviction: float = 0.5

    @field_validator("conviction")
    @classmethod
    def _validate_conviction(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("conviction must be between 0 and 1")
        return value


class DebateSummary(StrictModel):
    debate_id: str = Field(default_factory=lambda: make_id("db"))
    symbol: str
    as_of_date: date
    adjudication: str
    winning_side: Literal["bull", "bear", "draw"]
    confidence_balance: float = 0.5
    final_action: TradeAction = TradeAction.HOLD
    aligned_with_final_action: bool = True
    override_reason: str | None = None
    falsifiers: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)

    @field_validator("confidence_balance")
    @classmethod
    def _validate_balance(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence_balance must be between 0 and 1")
        return value


class PortfolioFitAssessment(StrictModel):
    fit_id: str = Field(default_factory=lambda: make_id("pfit"))
    symbol: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    fits_portfolio: bool
    recommended_action: OrderIntentType
    current_weight: float = 0.0
    target_weight: float = 0.0
    rationale: str
    conflicts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("current_weight", "target_weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("weights must be between 0 and 1")
        return value


class ExecutionPlan(StrictModel):
    plan_id: str = Field(default_factory=lambda: make_id("ep"))
    symbol: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    intent_type: OrderIntentType
    side: OrderSide | None = None
    target_weight: float = 0.0
    quantity: int | None = None
    order_type: OrderType = OrderType.MARKET
    time_in_force: TimeInForce = TimeInForce.DAY
    notes: list[str] = Field(default_factory=list)

    @field_validator("target_weight")
    @classmethod
    def _validate_target_weight(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("target_weight must be between 0 and 1")
        return value


class ResearchBundle(StrictModel):
    bundle_id: str = Field(default_factory=lambda: make_id("rb"))
    symbol: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    candidate_id: str | None = None
    regime_snapshot_id: str | None = None
    analyst_memos: list[AnalystMemo] = Field(default_factory=list)
    bull_case: BullCaseMemo
    bear_case: BearCaseMemo
    debate_summary: DebateSummary
    trader_note: str
    risk_committee_note: str | None = None
    portfolio_fit_id: str | None = None
    execution_plan_id: str | None = None
    final_decision_id: str
    warnings: list[str] = Field(default_factory=list)


class ResearchDecision(StrictModel):
    decision_id: str = Field(default_factory=lambda: make_id("rd"))
    symbol: str
    timestamp: datetime = Field(default_factory=utc_now)
    as_of_date: date
    action: TradeAction
    confidence: float
    thesis: str
    risk_flags: list[str] = Field(default_factory=list)
    invalidation_conditions: list[str] = Field(default_factory=list)
    time_horizon: str
    source_metadata: SourceMetadata
    desired_position_fraction: float | None = None

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return value

    @field_validator("desired_position_fraction")
    @classmethod
    def _validate_fraction(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("desired_position_fraction must be between 0 and 1")
        return value


class RiskDecision(StrictModel):
    risk_decision_id: str = Field(default_factory=lambda: make_id("rk"))
    source_decision_id: str
    symbol: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    approved: bool
    approved_size_fraction: float = 0.0
    proposed_size_fraction: float | None = None
    rejection_reason: str | None = None
    execution_constraints: ExecutionConstraints
    committee_notes: list[str] = Field(default_factory=list)
    risk_checks: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("approved_size_fraction")
    @classmethod
    def _validate_size(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("approved_size_fraction must be between 0 and 1")
        return value


class OrderIntent(StrictModel):
    intent_id: str = Field(default_factory=lambda: make_id("oi"))
    timestamp: datetime = Field(default_factory=utc_now)
    as_of_date: date
    symbol: str
    side: OrderSide
    quantity: int
    intent_type: OrderIntentType = OrderIntentType.NEW_ENTRY
    target_position_fraction: float | None = None
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    time_in_force: TimeInForce = TimeInForce.DAY
    source_decision_id: str
    source_risk_decision_id: str
    notes: list[str] = Field(default_factory=list)

    @field_validator("quantity")
    @classmethod
    def _validate_quantity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("quantity must be positive")
        return value


class OrderRecord(StrictModel):
    order_id: str = Field(default_factory=lambda: make_id("or"))
    intent_id: str
    as_of_date: date
    timestamp: datetime = Field(default_factory=utc_now)
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType
    time_in_force: TimeInForce
    status: OrderStatus
    requested_limit_price: float | None = None
    fill_price: float | None = None
    fill_timestamp: datetime | None = None
    commission: float = 0.0
    notes: list[str] = Field(default_factory=list)
    lifecycle: list[str] = Field(default_factory=list)


class FillRecord(StrictModel):
    fill_id: str = Field(default_factory=lambda: make_id("fl"))
    order_id: str
    as_of_date: date
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    commission: float
    slippage_bps: float
    timestamp: datetime = Field(default_factory=utc_now)
    realized_pnl: float | None = None


class PositionSnapshot(StrictModel):
    symbol: str
    timestamp: datetime = Field(default_factory=utc_now)
    quantity: int
    avg_cost: float
    market_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl_day: float = 0.0


class PortfolioSnapshot(StrictModel):
    snapshot_id: str = Field(default_factory=lambda: make_id("pf"))
    timestamp: datetime = Field(default_factory=utc_now)
    as_of_date: date
    cash: float
    equity: float
    gross_exposure: float
    positions: list[PositionSnapshot] = Field(default_factory=list)
    daily_realized_pnl: float = 0.0
    daily_unrealized_pnl: float = 0.0


class DailyRunSummary(StrictModel):
    run_id: str = Field(default_factory=lambda: make_id("run"))
    mode: RunMode
    as_of_date: date
    started_at: datetime
    completed_at: datetime
    status: str
    universe_size: int = 0
    eligible_universe_size: int = 0
    regime_label: str | None = None
    regime_risk_budget: float | None = None
    shortlisted_symbols: list[str] = Field(default_factory=list)
    watchlist_symbols: list[str] = Field(default_factory=list)
    approved_symbols: list[str] = Field(default_factory=list)
    rejected_symbols: dict[str, str] = Field(default_factory=dict)
    orders_submitted: int = 0
    fills_completed: int = 0
    research_action_counts: dict[str, int] = Field(default_factory=dict)
    block_reason_counts: dict[str, int] = Field(default_factory=dict)
    upstream_retry_count: int = 0
    upstream_failure_counts: dict[str, int] = Field(default_factory=dict)
    flat_book_suppressed: bool = False
    promoted_buy_count: int = 0
    promoted_buy_from_debate_count: int = 0
    blocked_buy_due_to_fallback_count: int = 0
    blocked_buy_due_to_thesis_inconsistency_count: int = 0
    action_thesis_mismatch_count: int = 0
    fallback_origin_decision_count: int = 0
    final_action_changed_count: int = 0
    report_path: str | None = None
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HealthCheckResult(StrictModel):
    name: str
    status: str
    detail: str
