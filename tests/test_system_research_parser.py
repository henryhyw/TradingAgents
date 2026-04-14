from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.research import DeterministicResearchAdapter
from tradingagents.system.research.parser import extract_json_object, normalize_rating, rating_to_action
from tradingagents.system.schemas import TradeAction

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_extract_json_object_from_wrapped_text():
    payload = extract_json_object("prefix {\"action\":\"buy\",\"confidence\":0.62} suffix")
    assert payload["action"] == "buy"
    assert payload["confidence"] == 0.62


def test_rating_normalization_and_mapping():
    assert normalize_rating("Final: overweight") == "OVERWEIGHT"
    assert rating_to_action("OVERWEIGHT") == TradeAction.BUY
    assert rating_to_action("UNDERWEIGHT") == TradeAction.SELL
    assert rating_to_action("HOLD") == TradeAction.HOLD


def test_deterministic_adapter_generates_buy_signal_on_positive_trend(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 13)
    history = make_price_history(as_of, periods=120, start_price=100, step=1.0)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    adapter = DeterministicResearchAdapter(provider, settings)
    decision = adapter.research("AAPL", as_of)
    assert decision.action == TradeAction.BUY
    assert decision.desired_position_fraction and decision.desired_position_fraction > 0
