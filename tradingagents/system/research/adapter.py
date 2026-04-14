from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.system.config import SystemSettings
from tradingagents.system.data import MarketDataProvider
from tradingagents.system.schemas import ResearchDecision, SourceMetadata, TradeAction

from .parser import extract_json_object, normalize_rating, rating_to_action, rating_to_confidence


logger = logging.getLogger(__name__)


class ResearchAdapter(ABC):
    @abstractmethod
    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        raise NotImplementedError


class TradingAgentsResearchAdapter(ResearchAdapter):
    def __init__(self, settings: SystemSettings):
        self.settings = settings
        self._graph: TradingAgentsGraph | None = None

    def _ensure_graph(self) -> TradingAgentsGraph:
        if self._graph is None:
            if not self.settings.llm_ready():
                raise RuntimeError("OPENAI_API_KEY is required for live TradingAgents research runs.")
            self._graph = TradingAgentsGraph(
                selected_analysts=self.settings.run.research_analysts,
                debug=False,
                config=self.settings.as_tradingagents_config(),
            )
        return self._graph

    def _artifact_path(self, symbol: str, as_of_date: date) -> str:
        return str(
            self.settings.paths.artifacts_dir
            / symbol
            / "TradingAgentsStrategy_logs"
            / f"full_states_log_{as_of_date.isoformat()}.json"
        )

    def _llm_parse(
        self,
        symbol: str,
        as_of_date: date,
        rating: str,
        final_state: dict,
        artifact_path: str,
    ) -> ResearchDecision:
        graph = self._ensure_graph()
        raw_text = final_state["final_trade_decision"]
        investment_plan = final_state.get("investment_plan", "")
        parser_prompt = f"""
Return JSON only.

Convert the following multi-agent trading research into a strict object with these keys:
- action: one of buy, sell, hold
- confidence: float between 0 and 1
- thesis: concise but specific paragraph
- risk_flags: list of short strings
- invalidation_conditions: list of short strings
- time_horizon: short phrase
- desired_position_fraction: float between 0 and 0.05, or 0 for hold/sell

Rules:
- Map BUY and OVERWEIGHT to buy.
- Map SELL and UNDERWEIGHT to sell.
- Map HOLD to hold.
- Keep desired_position_fraction at or below 0.05.
- Use the supplied rating unless the supporting text clearly contradicts it.

Rating: {rating}
Symbol: {symbol}
As of date: {as_of_date.isoformat()}

Investment plan:
{investment_plan}

Final trade decision:
{raw_text}
"""
        raw_response = graph.quick_thinking_llm.invoke(parser_prompt).content
        payload = extract_json_object(raw_response)
        action = rating_to_action(str(payload.get("action", rating)))
        confidence = float(payload.get("confidence", rating_to_confidence(rating)))
        desired_position_fraction = payload.get("desired_position_fraction")
        if desired_position_fraction is not None:
            desired_position_fraction = max(0.0, min(0.05, float(desired_position_fraction)))

        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=action,
            confidence=max(0.0, min(1.0, confidence)),
            thesis=str(payload.get("thesis", raw_text[:700])).strip(),
            risk_flags=[str(item) for item in payload.get("risk_flags", [])][:6],
            invalidation_conditions=[str(item) for item in payload.get("invalidation_conditions", [])][:4],
            time_horizon=str(payload.get("time_horizon", "1-4 weeks")).strip(),
            desired_position_fraction=desired_position_fraction,
            source_metadata=SourceMetadata(
                research_adapter="tradingagents_graph",
                llm_provider=self.settings.llm.provider,
                llm_model=self.settings.llm.quick_model,
                parser_mode="llm_json",
                upstream_rating=rating,
                upstream_artifact_path=artifact_path,
                notes=[],
                extra={"investment_plan_excerpt": investment_plan[:1000], "decision_excerpt": raw_text[:1000]},
            ),
        )

    def _fallback(
        self,
        symbol: str,
        as_of_date: date,
        rating: str,
        final_state: dict,
        artifact_path: str,
    ) -> ResearchDecision:
        raw_text = final_state["final_trade_decision"]
        action = rating_to_action(rating)
        desired_position_fraction = 0.03 if action == TradeAction.BUY else 0.0
        risk_flags = []
        lowered = raw_text.lower()
        for keyword in ("earnings", "valuation", "momentum", "guidance", "volatility", "macro"):
            if keyword in lowered:
                risk_flags.append(keyword)
        risk_flags.append("parser_fallback_used")
        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=action,
            confidence=rating_to_confidence(rating),
            thesis=raw_text[:1200].strip(),
            risk_flags=risk_flags[:6],
            invalidation_conditions=[
                "Price trend materially reverses against the thesis.",
                "New fundamental or earnings information invalidates the setup.",
            ],
            time_horizon="1-4 weeks",
            desired_position_fraction=desired_position_fraction,
            source_metadata=SourceMetadata(
                research_adapter="tradingagents_graph",
                llm_provider=self.settings.llm.provider,
                llm_model=self.settings.llm.quick_model,
                parser_mode="deterministic_fallback",
                upstream_rating=rating,
                upstream_artifact_path=artifact_path,
                notes=["Structured parser fallback applied."],
                extra={"decision_excerpt": raw_text[:1000]},
            ),
        )

    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        graph = self._ensure_graph()
        final_state, rating = graph.propagate(symbol, as_of_date.isoformat())
        normalized_rating = normalize_rating(str(rating))
        artifact_path = self._artifact_path(symbol, as_of_date)
        try:
            return self._llm_parse(symbol, as_of_date, normalized_rating, final_state, artifact_path)
        except Exception as exc:  # pragma: no cover - fallback is safety path
            logger.warning("Structured parser fallback for %s on %s: %s", symbol, as_of_date, exc)
            return self._fallback(symbol, as_of_date, normalized_rating, final_state, artifact_path)


class DeterministicResearchAdapter(ResearchAdapter):
    """Deterministic adapter for tests and local smoke validation without API keys."""

    def __init__(self, provider: MarketDataProvider, settings: SystemSettings):
        self.provider = provider
        self.settings = settings

    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        history = self.provider.get_history(symbol, as_of_date, 90)
        if history.empty or len(history) < 30:
            action = TradeAction.HOLD
            confidence = 0.35
            thesis = "Insufficient recent price history to support a deterministic signal."
            target_fraction = 0.0
        else:
            close = float(history["Close"].iloc[-1])
            sma_20 = float(history["Close"].tail(20).mean())
            return_20d = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1) if len(history) > 21 else 0.0
            if close > sma_20 and return_20d > 0.04:
                action = TradeAction.BUY
                confidence = min(0.75, 0.45 + abs(return_20d))
                thesis = f"Deterministic smoke signal is positive: close {close:.2f} above 20-day average {sma_20:.2f} with 20-day return {return_20d:.2%}."
                target_fraction = 0.03
            elif return_20d < -0.05:
                action = TradeAction.SELL
                confidence = min(0.70, 0.45 + abs(return_20d))
                thesis = f"Deterministic smoke signal is defensive: 20-day return {return_20d:.2%} is materially negative."
                target_fraction = 0.0
            else:
                action = TradeAction.HOLD
                confidence = 0.40
                thesis = f"Deterministic smoke signal is neutral: close {close:.2f}, 20-day average {sma_20:.2f}, 20-day return {return_20d:.2%}."
                target_fraction = 0.0

        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=action,
            confidence=confidence,
            thesis=thesis,
            risk_flags=["deterministic_test_mode"],
            invalidation_conditions=["Deterministic ranking deteriorates materially on the next run."],
            time_horizon="1-2 weeks",
            desired_position_fraction=target_fraction,
            source_metadata=SourceMetadata(
                research_adapter="deterministic_smoke_adapter",
                llm_provider="none",
                llm_model="none",
                parser_mode="deterministic",
                upstream_rating=action.value.upper(),
                upstream_artifact_path=None,
                notes=["Used only for tests and local smoke runs without OPENAI_API_KEY."],
                extra={},
            ),
        )
