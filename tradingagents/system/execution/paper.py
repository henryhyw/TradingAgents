from __future__ import annotations

from datetime import date

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import MarketDataProvider
from tradingagents.system.schemas import FillRecord, OrderIntent, OrderRecord, OrderStatus, OrderSide, PortfolioSnapshot, PositionSnapshot
from tradingagents.system.storage.repository import TradingRepository

from .base import BrokerAdapter


class PaperBroker(BrokerAdapter):
    def __init__(self, settings: SystemSettings, repository: TradingRepository, provider: MarketDataProvider):
        self.settings = settings
        self.repository = repository
        self.provider = provider

    def bootstrap(self, as_of_date: date) -> PortfolioSnapshot:
        snapshot = self.repository.get_latest_portfolio_snapshot()
        if snapshot is not None:
            return self.get_portfolio_snapshot(as_of_date)

        initial = PortfolioSnapshot(
            as_of_date=as_of_date,
            cash=self.settings.paper.starting_cash,
            equity=self.settings.paper.starting_cash,
            gross_exposure=0.0,
            positions=[],
            daily_realized_pnl=0.0,
            daily_unrealized_pnl=0.0,
        )
        self.repository.save_portfolio_snapshot(initial)
        return initial

    def get_portfolio_snapshot(self, as_of_date: date) -> PortfolioSnapshot:
        cash = self.repository.get_cash_balance(self.settings.paper.starting_cash)
        positions = self.repository.list_positions()
        fills_today = self.repository.list_fills_for_date(as_of_date)
        realized_by_symbol = {}
        for fill in fills_today:
            realized_by_symbol[fill.symbol] = realized_by_symbol.get(fill.symbol, 0.0) + (fill.realized_pnl or 0.0)

        updated_positions: list[PositionSnapshot] = []
        gross_exposure = 0.0
        unrealized_total = 0.0
        for position in positions:
            if position.quantity <= 0:
                continue
            latest_bar = self.provider.get_latest_bar(position.symbol, as_of_date)
            market_price = position.market_price if latest_bar is None else latest_bar.close
            market_value = market_price * position.quantity
            cost_basis = position.avg_cost * position.quantity
            unrealized = market_value - cost_basis
            gross_exposure += market_value
            unrealized_total += unrealized
            updated_positions.append(
                PositionSnapshot(
                    symbol=position.symbol,
                    quantity=position.quantity,
                    avg_cost=position.avg_cost,
                    market_price=market_price,
                    market_value=market_value,
                    cost_basis=cost_basis,
                    unrealized_pnl=unrealized,
                    realized_pnl_day=realized_by_symbol.get(position.symbol, 0.0),
                )
            )

        snapshot = PortfolioSnapshot(
            as_of_date=as_of_date,
            cash=cash,
            equity=cash + gross_exposure,
            gross_exposure=gross_exposure,
            positions=updated_positions,
            daily_realized_pnl=sum(realized_by_symbol.values()),
            daily_unrealized_pnl=unrealized_total,
        )
        self.repository.save_portfolio_snapshot(snapshot)
        return snapshot

    def _fill_price(self, close_price: float, side: OrderSide) -> float:
        slippage_factor = self.settings.paper.slippage_bps / 10_000
        if side == OrderSide.BUY:
            return close_price * (1 + slippage_factor)
        return close_price * (1 - slippage_factor)

    def submit_order(self, intent: OrderIntent, as_of_date: date) -> tuple[OrderRecord, FillRecord | None]:
        portfolio = self.get_portfolio_snapshot(as_of_date)
        latest_bar = self.provider.get_latest_bar(intent.symbol, as_of_date)

        if self.settings.paper.fill_model != "same_bar_close":
            raise RuntimeError(f"Unsupported paper fill model: {self.settings.paper.fill_model}")

        if latest_bar is None:
            order = OrderRecord(
                intent_id=intent.intent_id,
                as_of_date=as_of_date,
                symbol=intent.symbol,
                side=intent.side,
                quantity=intent.quantity,
                order_type=intent.order_type,
                time_in_force=intent.time_in_force,
                status=OrderStatus.REJECTED,
                commission=0.0,
                notes=["missing_market_data"] + intent.notes,
                lifecycle=["new", "rejected"],
            )
            self.repository.save_order_record(order)
            return order, None

        fill_price = self._fill_price(latest_bar.close, intent.side)
        commission = self.settings.paper.commission_per_order
        current_position = self.repository.get_position(intent.symbol)

        if intent.side == OrderSide.BUY:
            total_cost = intent.quantity * fill_price + commission
            if total_cost > portfolio.cash:
                order = OrderRecord(
                    intent_id=intent.intent_id,
                    as_of_date=as_of_date,
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    order_type=intent.order_type,
                    time_in_force=intent.time_in_force,
                    status=OrderStatus.REJECTED,
                    commission=0.0,
                    notes=["insufficient_cash"] + intent.notes,
                    lifecycle=["new", "rejected"],
                )
                self.repository.save_order_record(order)
                return order, None
            prev_qty = 0 if current_position is None else current_position.quantity
            prev_cost_basis = 0.0 if current_position is None else current_position.avg_cost * current_position.quantity
            new_quantity = prev_qty + intent.quantity
            new_cost_basis = prev_cost_basis + (intent.quantity * fill_price) + commission
            new_avg_cost = new_cost_basis / new_quantity
            new_cash = portfolio.cash - total_cost
            realized_pnl = None
        else:
            if current_position is None or current_position.quantity < intent.quantity:
                order = OrderRecord(
                    intent_id=intent.intent_id,
                    as_of_date=as_of_date,
                    symbol=intent.symbol,
                    side=intent.side,
                    quantity=intent.quantity,
                    order_type=intent.order_type,
                    time_in_force=intent.time_in_force,
                    status=OrderStatus.REJECTED,
                    commission=0.0,
                    notes=["insufficient_position"] + intent.notes,
                    lifecycle=["new", "rejected"],
                )
                self.repository.save_order_record(order)
                return order, None
            new_quantity = current_position.quantity - intent.quantity
            new_avg_cost = current_position.avg_cost if new_quantity > 0 else 0.0
            new_cost_basis = new_quantity * new_avg_cost
            new_cash = portfolio.cash + (intent.quantity * fill_price) - commission
            realized_pnl = ((fill_price - current_position.avg_cost) * intent.quantity) - commission

        order = OrderRecord(
            intent_id=intent.intent_id,
            as_of_date=as_of_date,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            order_type=intent.order_type,
            time_in_force=intent.time_in_force,
            status=OrderStatus.FILLED,
            fill_price=fill_price,
            commission=commission,
            notes=intent.notes,
            lifecycle=["new", "submitted", "filled"],
        )
        fill = FillRecord(
            order_id=order.order_id,
            as_of_date=as_of_date,
            symbol=intent.symbol,
            side=intent.side,
            quantity=intent.quantity,
            price=fill_price,
            commission=commission,
            slippage_bps=self.settings.paper.slippage_bps,
            realized_pnl=realized_pnl,
        )
        order.fill_timestamp = fill.timestamp

        position = PositionSnapshot(
            symbol=intent.symbol,
            quantity=new_quantity,
            avg_cost=new_avg_cost,
            market_price=latest_bar.close,
            market_value=new_quantity * latest_bar.close,
            cost_basis=new_cost_basis,
            unrealized_pnl=(new_quantity * latest_bar.close) - new_cost_basis,
            realized_pnl_day=0.0 if realized_pnl is None else realized_pnl,
        )

        self.repository.save_order_record(order)
        self.repository.save_fill_record(fill)
        self.repository.upsert_position(position)

        updated_snapshot = self.get_portfolio_snapshot(as_of_date)
        updated_snapshot = PortfolioSnapshot(
            as_of_date=updated_snapshot.as_of_date,
            cash=new_cash,
            equity=new_cash + updated_snapshot.gross_exposure,
            gross_exposure=updated_snapshot.gross_exposure,
            positions=updated_snapshot.positions,
            daily_realized_pnl=updated_snapshot.daily_realized_pnl,
            daily_unrealized_pnl=updated_snapshot.daily_unrealized_pnl,
        )
        self.repository.save_portfolio_snapshot(updated_snapshot)

        return order, fill
