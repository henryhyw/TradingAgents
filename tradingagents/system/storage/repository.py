from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from tradingagents.system.schemas import (
    DailyRunSummary,
    FillRecord,
    OrderIntent,
    OrderRecord,
    OrderStatus,
    PortfolioSnapshot,
    PositionSnapshot,
    ResearchDecision,
    RiskDecision,
)
from .db import connect, initialize_database


ModelT = TypeVar("ModelT", bound=BaseModel)


class TradingRepository:
    def __init__(self, database_path: Path):
        self.database_path = database_path
        initialize_database(self.database_path)

    def _insert(self, statement: str, parameters: tuple) -> None:
        with connect(self.database_path) as connection:
            connection.execute(statement, parameters)
            connection.commit()

    def _upsert(self, statement: str, parameters: tuple) -> None:
        self._insert(statement, parameters)

    @staticmethod
    def _to_json(model: BaseModel) -> str:
        return model.model_dump_json()

    @staticmethod
    def _from_json(model_type: type[ModelT], payload: str) -> ModelT:
        return model_type.model_validate_json(payload)

    def save_research_decision(self, decision: ResearchDecision) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO research_decisions
            (decision_id, symbol, as_of_date, timestamp, action, confidence, thesis, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.decision_id,
                decision.symbol,
                decision.as_of_date.isoformat(),
                decision.timestamp.isoformat(),
                decision.action.value,
                decision.confidence,
                decision.thesis,
                self._to_json(decision),
            ),
        )

    def save_risk_decision(self, decision: RiskDecision) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO risk_decisions
            (risk_decision_id, source_decision_id, symbol, as_of_date, timestamp, approved, approved_size_fraction, rejection_reason, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.risk_decision_id,
                decision.source_decision_id,
                decision.symbol,
                decision.as_of_date.isoformat(),
                decision.timestamp.isoformat(),
                int(decision.approved),
                decision.approved_size_fraction,
                decision.rejection_reason,
                self._to_json(decision),
            ),
        )

    def save_order_intent(self, intent: OrderIntent, status: OrderStatus) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO order_intents
            (intent_id, symbol, as_of_date, source_decision_id, source_risk_decision_id, side, quantity, status, timestamp, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                intent.intent_id,
                intent.symbol,
                intent.as_of_date.isoformat(),
                intent.source_decision_id,
                intent.source_risk_decision_id,
                intent.side.value,
                intent.quantity,
                status.value,
                intent.timestamp.isoformat(),
                self._to_json(intent),
            ),
        )

    def save_order_record(self, order: OrderRecord) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO orders
            (order_id, intent_id, symbol, as_of_date, side, quantity, status, timestamp, fill_timestamp, fill_price, commission, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                order.order_id,
                order.intent_id,
                order.symbol,
                order.as_of_date.isoformat(),
                order.side.value,
                order.quantity,
                order.status.value,
                order.timestamp.isoformat(),
                order.fill_timestamp.isoformat() if order.fill_timestamp else None,
                order.fill_price,
                order.commission,
                self._to_json(order),
            ),
        )

    def save_fill_record(self, fill: FillRecord) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO fills
            (fill_id, order_id, symbol, as_of_date, side, quantity, price, commission, slippage_bps, timestamp, realized_pnl, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fill.fill_id,
                fill.order_id,
                fill.symbol,
                fill.as_of_date.isoformat(),
                fill.side.value,
                fill.quantity,
                fill.price,
                fill.commission,
                fill.slippage_bps,
                fill.timestamp.isoformat(),
                fill.realized_pnl,
                self._to_json(fill),
            ),
        )

    def upsert_position(self, snapshot: PositionSnapshot) -> None:
        self._upsert(
            """
            INSERT INTO positions
            (symbol, timestamp, quantity, avg_cost, market_price, market_value, cost_basis, unrealized_pnl, realized_pnl_day, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                timestamp=excluded.timestamp,
                quantity=excluded.quantity,
                avg_cost=excluded.avg_cost,
                market_price=excluded.market_price,
                market_value=excluded.market_value,
                cost_basis=excluded.cost_basis,
                unrealized_pnl=excluded.unrealized_pnl,
                realized_pnl_day=excluded.realized_pnl_day,
                json=excluded.json
            """,
            (
                snapshot.symbol,
                snapshot.timestamp.isoformat(),
                snapshot.quantity,
                snapshot.avg_cost,
                snapshot.market_price,
                snapshot.market_value,
                snapshot.cost_basis,
                snapshot.unrealized_pnl,
                snapshot.realized_pnl_day,
                self._to_json(snapshot),
            ),
        )

    def save_portfolio_snapshot(self, snapshot: PortfolioSnapshot) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO portfolio_snapshots
            (snapshot_id, as_of_date, timestamp, cash, equity, gross_exposure, json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                snapshot.snapshot_id,
                snapshot.as_of_date.isoformat(),
                snapshot.timestamp.isoformat(),
                snapshot.cash,
                snapshot.equity,
                snapshot.gross_exposure,
                self._to_json(snapshot),
            ),
        )

    def save_daily_run_summary(self, summary: DailyRunSummary) -> None:
        self._insert(
            """
            INSERT OR REPLACE INTO daily_run_summaries
            (run_id, as_of_date, mode, status, started_at, completed_at, report_path, json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.run_id,
                summary.as_of_date.isoformat(),
                summary.mode.value,
                summary.status,
                summary.started_at.isoformat(),
                summary.completed_at.isoformat(),
                summary.report_path,
                self._to_json(summary),
            ),
        )

    def list_positions(self) -> list[PositionSnapshot]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT json FROM positions WHERE quantity <> 0 ORDER BY symbol"
            ).fetchall()
        return [self._from_json(PositionSnapshot, row["json"]) for row in rows]

    def get_position(self, symbol: str) -> PositionSnapshot | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT json FROM positions WHERE symbol = ?",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return self._from_json(PositionSnapshot, row["json"])

    def list_recent_orders(self, limit: int = 20) -> list[OrderRecord]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT json FROM orders ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._from_json(OrderRecord, row["json"]) for row in rows]

    def list_fills_for_date(self, as_of_date: date) -> list[FillRecord]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT json FROM fills WHERE as_of_date = ? ORDER BY timestamp DESC",
                (as_of_date.isoformat(),),
            ).fetchall()
        return [self._from_json(FillRecord, row["json"]) for row in rows]

    def count_opening_orders_for_symbol(self, symbol: str, as_of_date: date) -> int:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM fills
                WHERE symbol = ?
                  AND side = 'buy'
                  AND as_of_date = ?
                """,
                (symbol, as_of_date.isoformat()),
            ).fetchone()
        return int(row["count"])

    def count_losing_exits(self, as_of_date: date) -> int:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM fills
                WHERE side = 'sell'
                  AND realized_pnl < 0
                  AND as_of_date = ?
                """,
                (as_of_date.isoformat(),),
            ).fetchone()
        return int(row["count"])

    def get_latest_portfolio_snapshot(self) -> PortfolioSnapshot | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT json FROM portfolio_snapshots ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return self._from_json(PortfolioSnapshot, row["json"])

    def get_cash_balance(self, default_cash: float) -> float:
        snapshot = self.get_latest_portfolio_snapshot()
        return default_cash if snapshot is None else snapshot.cash

    def latest_run_summary(self) -> DailyRunSummary | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT json FROM daily_run_summaries ORDER BY completed_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return self._from_json(DailyRunSummary, row["json"])

    def get_first_portfolio_snapshot_for_date(self, as_of_date: date) -> PortfolioSnapshot | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT json
                FROM portfolio_snapshots
                WHERE as_of_date = ?
                ORDER BY timestamp ASC
                LIMIT 1
                """,
                (as_of_date.isoformat(),),
            ).fetchone()
        if row is None:
            return None
        return self._from_json(PortfolioSnapshot, row["json"])

    def list_research_decisions_for_date(self, as_of_date: date) -> list[ResearchDecision]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                "SELECT json FROM research_decisions WHERE as_of_date = ? ORDER BY timestamp ASC",
                (as_of_date.isoformat(),),
            ).fetchall()
        return [self._from_json(ResearchDecision, row["json"]) for row in rows]

    def list_risk_decisions_for_date(self, as_of_date: date) -> list[RiskDecision]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT json
                FROM risk_decisions
                WHERE as_of_date = ?
                ORDER BY timestamp ASC
                """,
                (as_of_date.isoformat(),),
            ).fetchall()
        return [self._from_json(RiskDecision, row["json"]) for row in rows]

    def get_run_summary_for_date(self, as_of_date: date) -> DailyRunSummary | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT json
                FROM daily_run_summaries
                WHERE as_of_date = ?
                ORDER BY completed_at DESC
                LIMIT 1
                """,
                (as_of_date.isoformat(),),
            ).fetchone()
        if row is None:
            return None
        return self._from_json(DailyRunSummary, row["json"])

    def dump_table_counts(self) -> dict[str, int]:
        tables = [
            "research_decisions",
            "risk_decisions",
            "order_intents",
            "orders",
            "fills",
            "positions",
            "portfolio_snapshots",
            "daily_run_summaries",
        ]
        counts: dict[str, int] = {}
        with connect(self.database_path) as connection:
            for table in tables:
                row = connection.execute(f"SELECT COUNT(*) AS count FROM {table}").fetchone()
                counts[table] = int(row["count"])
        return counts
