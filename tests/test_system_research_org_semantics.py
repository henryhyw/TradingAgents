from __future__ import annotations

from datetime import date

import pandas as pd

from tradingagents.system.config import load_settings
from tradingagents.system.research import ResearchAdapter, ResearchOrganization
from tradingagents.system.schemas import (
    AnalystMemo,
    CandidateAssessment,
    DebateSummary,
    EntryMode,
    OrderIntentType,
    PositionSnapshot,
    RegimeLabel,
    RegimeSnapshot,
    ResearchDecision,
    SourceMetadata,
    TradeAction,
)

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


class StaticActionAdapter(ResearchAdapter):
    def __init__(
        self,
        action: TradeAction,
        *,
        confidence: float = 0.58,
        thesis: str | None = None,
        parser_mode: str = "deterministic",
        risk_flags: list[str] | None = None,
        source_extra: dict | None = None,
    ):
        self.action = action
        self.confidence = confidence
        self.thesis = thesis or f"Static adapter action {self.action.value}"
        self.parser_mode = parser_mode
        self.risk_flags = risk_flags or []
        self.source_extra = source_extra or {}

    def research(self, symbol: str, as_of_date: date) -> ResearchDecision:
        return ResearchDecision(
            symbol=symbol,
            as_of_date=as_of_date,
            action=self.action,
            confidence=self.confidence,
            thesis=self.thesis,
            risk_flags=self.risk_flags,
            invalidation_conditions=["n/a"],
            time_horizon="1-4 weeks",
            desired_position_fraction=0.0 if self.action != TradeAction.BUY else 0.03,
            source_metadata=SourceMetadata(
                research_adapter="unit_test",
                llm_provider="none",
                llm_model="none",
                parser_mode=self.parser_mode,
                extra=self.source_extra,
            ),
        )


class InconsistentBuyRewriteOrg(ResearchOrganization):
    def _synthesize_final_thesis(self, **kwargs):  # type: ignore[override]
        final_action = kwargs.get("final_action")
        if final_action == TradeAction.BUY:
            return "Avoid new entries and wait for pullback due to unfavorable risk/reward."
        return super()._synthesize_final_thesis(**kwargs)


def _custom_history(prices: list[float], as_of: date, volume: float = 7_000_000) -> pd.DataFrame:
    dates = pd.bdate_range(end=as_of, periods=len(prices))
    rows = []
    for ts, close in zip(dates, prices):
        rows.append(
            {
                "Date": ts.tz_localize(None),
                "Open": close * 0.995,
                "High": close * 1.005,
                "Low": close * 0.99,
                "Close": close,
                "Volume": volume,
            }
        )
    return pd.DataFrame(rows)


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


def _balanced_regime(as_of: date) -> RegimeSnapshot:
    return RegimeSnapshot(
        as_of_date=as_of,
        label=RegimeLabel.BALANCED,
        volatility_regime="normal",
        trend_regime="mixed",
        risk_on_score=0.10,
        risk_budget_multiplier=1.0,
        max_gross_exposure_fraction=0.25,
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


def test_research_org_blocks_buy_promotion_for_fallback_origin(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=1.2, volume=7_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(
        settings=settings,
        provider=provider,
        adapter=StaticActionAdapter(
            TradeAction.AVOID,
            confidence=0.80,
            parser_mode="upstream_error_no_entry",
            thesis="Upstream fallback no-entry state due to ResourceExhausted.",
            risk_flags=["upstream_graph_failure", "insufficient_research_confidence"],
            source_extra={"upstream_fallback_mode": "research_error_no_entry", "upstream_failure_type": "ResourceExhausted"},
        ),
    )

    decision, bundle = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert bundle.debate_summary.winning_side == "bull"
    assert decision.action == TradeAction.AVOID
    assert bundle.debate_summary.final_action == TradeAction.AVOID
    assert decision.source_metadata.extra.get("buy_promotion_applied") is False
    assert decision.source_metadata.extra.get("buy_blocked_due_to_fallback") is True
    assert "insufficient research confidence" in decision.thesis.lower()


def test_research_org_blocks_bearish_buy_thesis_when_flat(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=1.2, volume=7_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(
        settings=settings,
        provider=provider,
        adapter=StaticActionAdapter(
            TradeAction.BUY,
            confidence=0.71,
            thesis="Definitive SELL: avoid new entries, trim risk, and wait for pullback due to unfavorable risk/reward.",
        ),
    )

    decision, bundle = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert decision.action == TradeAction.BUY
    assert bundle.debate_summary.final_action == TradeAction.BUY
    assert decision.source_metadata.extra.get("buy_rewrite_attempted") is True
    assert decision.source_metadata.extra.get("buy_rewrite_success") is True
    assert "entry rationale" in decision.thesis.lower()


def test_research_org_hard_bans_fallback_origin_buy(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=1.2, volume=7_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(
        settings=settings,
        provider=provider,
        adapter=StaticActionAdapter(
            TradeAction.BUY,
            confidence=0.82,
            parser_mode="upstream_error_no_entry",
            thesis="Research adapter fallback after ResourceExhausted.",
            risk_flags=["research_error:ResourceExhausted", "upstream_graph_failure", "insufficient_research_confidence"],
            source_extra={"upstream_fallback_mode": "research_error_no_entry", "upstream_failure_type": "ResourceExhausted"},
        ),
    )

    decision, bundle = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert decision.action == TradeAction.AVOID
    assert bundle.debate_summary.final_action == TradeAction.AVOID
    assert decision.source_metadata.extra.get("fallback_buy_blocked") is True
    assert decision.source_metadata.extra.get("final_action_downgraded") is True
    assert decision.source_metadata.extra.get("inconsistent_buy_prevented") is True


def test_research_org_downgrades_buy_when_rewrite_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=1.2, volume=7_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = InconsistentBuyRewriteOrg(
        settings=settings,
        provider=provider,
        adapter=StaticActionAdapter(TradeAction.BUY, confidence=0.71, thesis="Buy setup from upstream."),
    )

    decision, bundle = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert decision.action == TradeAction.AVOID
    assert bundle.debate_summary.final_action == TradeAction.AVOID
    assert decision.source_metadata.extra.get("buy_rewrite_attempted") is True
    assert decision.source_metadata.extra.get("buy_rewrite_failure") is True
    assert decision.source_metadata.extra.get("final_action_downgraded") is True


def test_research_org_assigns_breakout_entry_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    base = [100 + (i * 0.35) for i in range(170)]
    prices = base + [base[-1] * 1.01, base[-1] * 1.02, base[-1] * 1.03, base[-1] * 1.035, base[-1] * 1.04]
    history = _custom_history(prices, as_of, volume=8_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD, confidence=0.72))

    decision, _ = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert decision.action == TradeAction.BUY
    assert decision.entry_mode == EntryMode.BREAKOUT
    assert "breakout" in (decision.entry_trigger_reason or "")


def test_research_org_assigns_pullback_entry_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    trend = [100 + (i * 0.45) for i in range(150)]
    pullback = [trend[-1] * 0.97, trend[-1] * 0.965, trend[-1] * 0.972, trend[-1] * 0.981, trend[-1] * 0.989]
    prices = trend + pullback + [
        trend[-1] * 0.979,
        trend[-1] * 0.983,
        trend[-1] * 0.988,
        trend[-1] * 0.992,
        trend[-1] * 0.995,
    ]
    history = _custom_history(prices, as_of, volume=7_500_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.BUY, confidence=0.74))

    decision, _ = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert decision.action == TradeAction.BUY
    assert decision.entry_mode == EntryMode.PULLBACK
    assert "pullback" in (decision.entry_trigger_reason or "")


def test_research_org_relaxes_near_breakout_in_risk_on(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=0.8, volume=8_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD))

    entry_mode, reason, blockers, _, _ = org._entry_mode_and_gates(
        candidate=_candidate(as_of),
        regime=_regime(as_of),
        technical_memo=AnalystMemo(
            symbol="AAPL",
            as_of_date=as_of,
            role="Technical Analyst",
            signal="bullish",
            confidence=0.55,
            summary="Strong trend with near-breakout confirmation.",
        ),
        technical_state={
            "insufficient_history": False,
            "extension_over_ma20": 0.045,
            "extension_over_ma50": 0.09,
            "rsi14": 66.0,
            "trend_alignment": True,
            "ret_3d": 0.003,
            "ret_20d": 0.065,
            "ret_60d": 0.18,
            "breakout_distance": -0.002,
            "pullback_distance_ma20": 0.08,
            "pullback_distance_ma50": 0.12,
        },
    )

    assert entry_mode == EntryMode.BREAKOUT
    assert reason == "risk_on_relaxed_breakout_confirmation"
    assert blockers == []


def test_research_org_applies_risk_on_participation_bias(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    history = make_price_history(as_of, periods=180, start_price=90, step=0.8, volume=8_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD))
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.AVOID,
        confidence=0.53,
        thesis="Constructive entry setup with trend support, but original signal was cautious.",
        risk_flags=[],
        invalidation_conditions=["close below trend support"],
        time_horizon="1-4 weeks",
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )
    debate = DebateSummary(
        symbol="AAPL",
        as_of_date=as_of,
        adjudication="Bear case is not strong enough to suppress a validated risk-on starter entry.",
        winning_side="bear",
        confidence_balance=0.60,
    )
    technical_memo = AnalystMemo(
        symbol="AAPL",
        as_of_date=as_of,
        role="Technical Analyst",
        signal="bullish",
        confidence=0.56,
        summary="Trend aligned with near breakout confirmation.",
    )
    technical_state = {
        "insufficient_history": False,
        "close": 120.0,
        "sma20": 115.0,
        "sma50": 110.0,
        "extension_over_ma20": 0.043,
        "extension_over_ma50": 0.091,
        "rsi14": 65.0,
        "trend_alignment": True,
        "ret_3d": 0.004,
        "ret_20d": 0.07,
        "ret_60d": 0.18,
        "breakout_distance": 0.003,
        "pullback_distance_ma20": 0.07,
        "pullback_distance_ma50": 0.11,
    }

    updated, updated_debate = org._adjudicate_long_only_action(
        decision=decision,
        debate=debate,
        candidate=_candidate(as_of),
        regime=_regime(as_of),
        technical_memo=technical_memo,
        current_position=None,
        technical_state=technical_state,
    )

    assert updated.action == TradeAction.BUY
    assert updated.desired_position_fraction == settings.risk.risk_on_starter_position_fraction
    assert updated.source_metadata.extra.get("risk_on_participation_bias_applied") is True
    assert updated.source_metadata.extra.get("buy_promotion_source") == "risk_on_participation_bias"
    assert "risk_on_participation_bias" in (updated_debate.override_reason or "")


def test_research_org_blocks_overheated_breakout_buy(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    prices = [100 + (i * 0.3) for i in range(150)] + [180 + (i * 3.2) for i in range(30)]
    history = _custom_history(prices, as_of, volume=9_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.BUY, confidence=0.82))

    decision, _ = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=None)

    assert decision.action == TradeAction.AVOID
    assert decision.source_metadata.extra.get("buy_blocked_due_to_overheat") is True
    assert decision.source_metadata.extra.get("final_action_downgraded") is True


def test_research_org_applies_trend_failure_exit_overlay(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    prices = [140 + (i * 0.4) for i in range(120)] + [185, 183, 179, 172, 166, 160, 154, 150, 147, 145]
    history = _custom_history(prices, as_of, volume=7_000_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD, confidence=0.61))
    position = PositionSnapshot(
        symbol="AAPL",
        quantity=100,
        avg_cost=130.0,
        market_price=float(history["Close"].iloc[-1]),
        market_value=100 * float(history["Close"].iloc[-1]),
        cost_basis=13000.0,
        unrealized_pnl=(float(history["Close"].iloc[-1]) - 130.0) * 100,
    )

    decision, _ = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=position, position_holding_days=11)

    assert decision.action == TradeAction.SELL
    assert decision.position_lifecycle_state in {OrderIntentType.EXIT, OrderIntentType.TRIM_PARTIAL}
    assert decision.source_metadata.extra.get("exit_type") == "trend_failure_exit"


def test_research_org_applies_time_stop_exit(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    prices = [100 + (i * 0.08) for i in range(180)]
    history = _custom_history(prices, as_of, volume=6_500_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD, confidence=0.58))
    close = float(history["Close"].iloc[-1])
    position = PositionSnapshot(
        symbol="AAPL",
        quantity=120,
        avg_cost=close * 0.995,
        market_price=close,
        market_value=120 * close,
        cost_basis=120 * close * 0.995,
        unrealized_pnl=(close - (close * 0.995)) * 120,
    )

    decision, _ = org.run("AAPL", as_of, _candidate(as_of), _regime(as_of), current_position=position, position_holding_days=15)

    assert decision.action == TradeAction.SELL
    assert decision.position_lifecycle_state == OrderIntentType.TRIM_PARTIAL
    assert decision.source_metadata.extra.get("exit_type") == "time_stop_exit"


def test_research_org_applies_full_time_stop_exit_outside_risk_on(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    prices = [100 + (i * 0.08) for i in range(180)]
    history = _custom_history(prices, as_of, volume=6_500_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.HOLD, confidence=0.58))
    close = float(history["Close"].iloc[-1])
    position = PositionSnapshot(
        symbol="AAPL",
        quantity=120,
        avg_cost=close * 0.995,
        market_price=close,
        market_value=120 * close,
        cost_basis=120 * close * 0.995,
        unrealized_pnl=(close - (close * 0.995)) * 120,
    )

    decision, _ = org.run(
        "AAPL",
        as_of,
        _candidate(as_of),
        _balanced_regime(as_of),
        current_position=position,
        position_holding_days=15,
    )

    assert decision.action == TradeAction.SELL
    assert decision.position_lifecycle_state == OrderIntentType.EXIT
    assert decision.source_metadata.extra.get("exit_type") == "time_stop_exit"


def test_research_org_trims_generic_sell_for_profitable_risk_on_position(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 15)
    base = [100 + (i * 0.25) for i in range(165)]
    prices = base + [base[-1] * 1.025, base[-1] * 1.035, base[-1] * 1.045, base[-1] * 1.055, base[-1] * 1.065]
    history = _custom_history(prices, as_of, volume=7_500_000)
    provider = FakeMarketDataProvider(symbols_with_same_history(["AAPL"], history))
    org = ResearchOrganization(settings=settings, provider=provider, adapter=StaticActionAdapter(TradeAction.SELL, confidence=0.62))
    close = float(history["Close"].iloc[-1])
    position = PositionSnapshot(
        symbol="AAPL",
        quantity=100,
        avg_cost=close / 1.08,
        market_price=close,
        market_value=100 * close,
        cost_basis=100 * close / 1.08,
        unrealized_pnl=(close - (close / 1.08)) * 100,
    )

    decision, _ = org.run(
        "AAPL",
        as_of,
        _candidate(as_of),
        _regime(as_of),
        current_position=position,
        position_holding_days=6,
    )

    assert decision.action == TradeAction.SELL
    assert decision.position_lifecycle_state == OrderIntentType.TRIM_PARTIAL
    assert decision.source_metadata.extra.get("full_exit_rejected_in_favor_of_trim") is True
    assert "partial trim rationale" in decision.thesis.lower()
