from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.research import ResearchAdapter, ResearchOrganization
from tradingagents.system.schemas import (
    CandidateAssessment,
    RegimeLabel,
    RegimeSnapshot,
    ResearchDecision,
    SourceMetadata,
    TradeAction,
)

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


class StaticActionAdapter(ResearchAdapter):
    def __init__(self, action: TradeAction):
        self.action = action

    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=self.action,
            confidence=0.58,
            thesis=f"Static adapter action {self.action.value}",
            risk_flags=[],
            invalidation_conditions=["n/a"],
            time_horizon="1-4 weeks",
            desired_position_fraction=0.0 if self.action != TradeAction.BUY else 0.03,
            source_metadata=SourceMetadata(
                research_adapter="unit_test",
                llm_provider="none",
                llm_model="none",
                parser_mode="deterministic",
            ),
        )


def _candidate(as_of: date) -> CandidateAssessment:
    return CandidateAssessment(
        symbol="AAPL",
        as_of_date=as_of,
        name="Apple Inc",
        asset_type="Equity",
        sector="Technology",
        eligible=True,
        watchlist_only=False,
        close=140.0,
        avg_dollar_volume_20d=80_000_000,
        return_20d=0.09,
        return_60d=0.18,
        volatility_20d=0.24,
        relative_strength_20d=0.06,
        ranking_score=0.86,
        regime_fit_score=0.72,
    )


def _regime(as_of: date) -> RegimeSnapshot:
    return RegimeSnapshot(
        as_of_date=as_of,
        label=RegimeLabel.RISK_ON,
        volatility_regime="contained",
        trend_regime="uptrend",
        risk_on_score=0.62,
        risk_budget_multiplier=1.10,
        max_gross_exposure_fraction=0.30,
    )


def test_research_org_recasts_sell_to_avoid_when_flat(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=110, step=0.7, volume=5_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.SELL))
    candidate = _candidate(as_of).model_copy(update={"relative_strength_20d": -0.02})

    decision, bundle = org.run("AAPL", as_of, candidate, _regime(as_of), current_position=None)

    assert decision.action == TradeAction.AVOID
    assert bundle.debate_summary.final_action == TradeAction.AVOID
    assert bundle.debate_summary.override_reason is not None
    assert "no_inventory" in bundle.debate_summary.override_reason


def test_research_org_promotes_bull_non_entry_to_buy(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=1.2, volume=7_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD))

    decision, bundle = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert bundle.debate_summary.winning_side == "bull"
    assert decision.action == TradeAction.BUY
    assert bundle.debate_summary.final_action == TradeAction.BUY
    assert bundle.debate_summary.override_reason is not None
    assert "entry_gate_passed" in bundle.debate_summary.override_reason
