from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any
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
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    PENDING = "pending"


class RunMode(str, Enum):
    DRY_RUN = "dry_run"
    PAPER = "paper"
    DAILY = "daily"
    REPLAY = "replay"


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
    rejection_reason: str | None = None
    execution_constraints: ExecutionConstraints
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
    shortlisted_symbols: list[str] = Field(default_factory=list)
    approved_symbols: list[str] = Field(default_factory=list)
    rejected_symbols: dict[str, str] = Field(default_factory=dict)
    orders_submitted: int = 0
    fills_completed: int = 0
    report_path: str | None = None
    notes: list[str] = Field(default_factory=list)


class HealthCheckResult(StrictModel):
    name: str
    status: str
    detail: str
