from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.research import DeterministicResearchAdapter, TradingAgentsResearchAdapter
from tradingagents.system.research.parser import extract_json_object, normalize_rating, rating_to_action
from tradingagents.system.schemas import ResearchDecision, SourceMetadata, TradeAction

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
    assert rating_to_action("NO_ENTRY") == TradeAction.AVOID


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


def test_tradingagents_adapter_returns_safe_hold_on_upstream_error(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = load_settings()
    adapter = TradingAgentsResearchAdapter(settings)

    class BrokenGraph:
        def propagate(self, symbol: str, as_of: str):
            raise RuntimeError("quota exceeded")

    adapter._graphs[tuple(settings.run.research_analysts)] = BrokenGraph()  # type: ignore[assignment]
    decision = adapter.research("AAPL", date(2026, 4, 13))
    assert decision.action == TradeAction.AVOID
    assert "upstream_graph_failure" in decision.risk_flags
    assert decision.source_metadata.parser_mode == "upstream_error_no_entry"


def test_tradingagents_adapter_resource_exhausted_retries(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = load_settings()
    adapter = TradingAgentsResearchAdapter(settings)

    class ResourceExhausted(Exception):
        pass

    class FlakyGraph:
        def __init__(self):
            self.calls = 0

        def propagate(self, symbol: str, as_of: str):
            self.calls += 1
            if self.calls == 1:
                raise ResourceExhausted("capacity")
            return {"final_trade_decision": "BUY thesis", "investment_plan": "plan"}, "BUY"

    graph = FlakyGraph()
    adapter._graphs[tuple(settings.run.research_analysts)] = graph  # type: ignore[assignment]
    monkeypatch.setattr(adapter, "_llm_parse", lambda *args, **kwargs: ResearchDecision(
        symbol="AAPL",
        as_of_date=date(2026, 4, 13),
        action=TradeAction.BUY,
        confidence=0.7,
        thesis="parsed",
        risk_flags=[],
        invalidation_conditions=["n/a"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.03,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="llm_json",
            extra={"upstream_retry_count": 1, "upstream_failure_counts": {"ResourceExhausted": 1}},
        ),
    ))
    decision = adapter.research("AAPL", date(2026, 4, 13))
    assert decision.action == TradeAction.BUY
    assert graph.calls == 2


def test_tradingagents_adapter_invalid_argument_uses_narrow_scope(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = load_settings()
    adapter = TradingAgentsResearchAdapter(settings)

    class InvalidArgument(Exception):
        pass

    class BrokenGraph:
        def propagate(self, symbol: str, as_of: str):
            raise InvalidArgument("bad tool payload")

    class NarrowGraph:
        def __init__(self):
            self.calls = 0

        def propagate(self, symbol: str, as_of: str):
            self.calls += 1
            return {"final_trade_decision": "UNDERWEIGHT stance", "investment_plan": "narrow"}, "UNDERWEIGHT"

    default_key = tuple(settings.run.research_analysts)
    narrow_key = tuple(adapter._NARROW_ANALYSTS)
    adapter._graphs[default_key] = BrokenGraph()  # type: ignore[assignment]
    narrow_graph = NarrowGraph()
    adapter._graphs[narrow_key] = narrow_graph  # type: ignore[assignment]
    monkeypatch.setattr(adapter, "_llm_parse", lambda *args, **kwargs: ResearchDecision(
        symbol="AAPL",
        as_of_date=date(2026, 4, 13),
        action=TradeAction.SELL,
        confidence=0.6,
        thesis="parsed narrow",
        risk_flags=[],
        invalidation_conditions=["n/a"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.0,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="llm_json",
            extra={"upstream_retry_count": 1, "upstream_failure_counts": {"InvalidArgument": 1}},
        ),
    ))
    decision = adapter.research("AAPL", date(2026, 4, 13))
    assert decision.action == TradeAction.SELL
    assert narrow_graph.calls == 1
