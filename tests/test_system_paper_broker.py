from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.execution import PaperBroker
from tradingagents.system.schemas import OrderIntent, OrderSide
from tradingagents.system.storage.repository import TradingRepository

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_paper_broker_tracks_cash_positions_and_fills(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_COMMISSION_PER_ORDER", "0")
    monkeypatch.setenv("TRADINGAGENTS_SLIPPAGE_BPS", "0")
    settings = load_settings()
    repository = TradingRepository(settings.paths.database_path)
    as_of = date(2026, 4, 13)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], make_price_history(as_of, step=0.5)))
    broker = PaperBroker(settings, repository, provider)
    broker.bootstrap(as_of)

    buy_intent = OrderIntent(
        as_of_date=as_of,
        symbol="AAPL",
        side=OrderSide.BUY,
        quantity=10,
        source_decision_id="rd_buy",
        source_risk_decision_id="rk_buy",
    )
    buy_order, buy_fill = broker.submit_order(buy_intent, as_of)
    assert buy_fill is not None
    assert buy_order.status.value == "filled"

    after_buy = broker.get_portfolio_snapshot(as_of)
    assert after_buy.cash < settings.paper.starting_cash
    assert any(position.symbol == "AAPL" and position.quantity == 10 for position in after_buy.positions)

    sell_intent = OrderIntent(
        as_of_date=as_of,
        symbol="AAPL",
        side=OrderSide.SELL,
        quantity=10,
        source_decision_id="rd_sell",
        source_risk_decision_id="rk_sell",
    )
    sell_order, sell_fill = broker.submit_order(sell_intent, as_of)
    assert sell_fill is not None
    assert sell_order.status.value == "filled"

    after_sell = broker.get_portfolio_snapshot(as_of)
    remaining = [position for position in after_sell.positions if position.symbol == "AAPL"]
    assert remaining == [] or remaining[0].quantity == 0
