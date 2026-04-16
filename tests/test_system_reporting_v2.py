from __future__ import annotations

from datetime import date

from tradingagents.system.orchestration.reporting import generate_daily_report
from tradingagents.system.schemas import (
    AnalystMemo,
    BearCaseMemo,
    BullCaseMemo,
    CandidateAssessment,
    DailyRunSummary,
    DebateSummary,
    PortfolioSnapshot,
    RegimeLabel,
    RegimeSnapshot,
    ResearchBundle,
    ResearchDecision,
    RiskDecision,
    RunMode,
    SourceMetadata,
    TradeAction,
    ExecutionConstraints,
)
from tradingagents.system.universe import ScreenedAsset


def test_generate_daily_report_includes_v2_sections(tmp_path):
    as_of = date(2026, 4, 13)
    decision = ResearchDecision(
        symbol="AAPL",
        as_of_date=as_of,
        action=TradeAction.BUY,
        confidence=0.68,
        thesis="Test thesis",
        risk_flags=[],
        invalidation_conditions=["invalid"],
        time_horizon="1-4 weeks",
        desired_position_fraction=0.03,
        source_metadata=SourceMetadata(
            research_adapter="unit_test",
            llm_provider="none",
            llm_model="none",
            parser_mode="deterministic",
        ),
    )
    risk = RiskDecision(
        source_decision_id=decision.decision_id,
        symbol="AAPL",
        as_of_date=as_of,
        approved=True,
        approved_size_fraction=0.03,
        execution_constraints=ExecutionConstraints(),
    )
    candidate = CandidateAssessment(
        symbol="AAPL",
        as_of_date=as_of,
        name="Apple",
        asset_type="Equity",
        sector="Technology",
        eligible=True,
        close=100.0,
        avg_dollar_volume_20d=50_000_000,
        return_20d=0.05,
        return_60d=0.1,
        volatility_20d=0.2,
        ranking_score=0.8,
        shortlist_reason="ranked_candidate",
    )
    bundle = ResearchBundle(
        symbol="AAPL",
        as_of_date=as_of,
        candidate_id=candidate.candidate_id,
        analyst_memos=[
            AnalystMemo(
                symbol="AAPL",
                as_of_date=as_of,
                role="Technical Analyst",
                signal="bullish",
                confidence=0.6,
                summary="Uptrend",
            )
        ],
        bull_case=BullCaseMemo(symbol="AAPL", as_of_date=as_of, summary="Bull"),
        bear_case=BearCaseMemo(symbol="AAPL", as_of_date=as_of, summary="Bear"),
        debate_summary=DebateSummary(
            symbol="AAPL",
            as_of_date=as_of,
            adjudication="Bull wins",
            winning_side="bull",
            final_action=TradeAction.BUY,
            aligned_with_final_action=True,
        ),
        trader_note="Buy",
        final_decision_id=decision.decision_id,
    )
    regime = RegimeSnapshot(
        as_of_date=as_of,
        label=RegimeLabel.BALANCED,
        volatility_regime="normal",
        trend_regime="mixed",
        risk_on_score=0.05,
        risk_budget_multiplier=1.0,
        max_gross_exposure_fraction=0.30,
    )
    summary = DailyRunSummary(
        mode=RunMode.DRY_RUN,
        as_of_date=as_of,
        started_at=decision.timestamp,
        completed_at=decision.timestamp,
        status="completed",
        universe_size=100,
        eligible_universe_size=80,
        shortlisted_symbols=["AAPL"],
        research_action_counts={"buy": 1, "avoid": 0},
        block_reason_counts={"risk_limits": 0},
        upstream_retry_count=1,
        upstream_failure_counts={"ResourceExhausted": 1},
        flat_book_suppressed=False,
    )
    shortlist = [
        ScreenedAsset(
            symbol="AAPL",
            name="Apple",
            asset_type="Equity",
            sector="Technology",
            close=100.0,
            avg_dollar_volume_20d=50_000_000,
            return_20d=0.05,
            return_60d=0.10,
            volatility_20d=0.20,
            score=0.8,
            shortlist_reason="ranked_candidate",
        )
    ]
    portfolio = PortfolioSnapshot(as_of_date=as_of, cash=100_000, equity=100_000, gross_exposure=0.0, positions=[])

    report_path = generate_daily_report(
        report_root=tmp_path,
        as_of_date=as_of,
        summary=summary,
        shortlist=shortlist,
        research_decisions=[decision],
        risk_decisions=[risk],
        orders=[],
        fills=[],
        portfolio=portfolio,
        regime_snapshot=regime,
        candidate_assessments=[candidate],
        research_bundles=[bundle],
        portfolio_fits=[],
        execution_plans=[],
    )
    content = report_path.read_text(encoding="utf-8")
    assert "## Regime Summary" in content
    assert "## Research & Debate" in content
    assert "## Risk Committee" in content
    assert "## Diagnostics" in content
    assert "final_action=buy" in content.lower()
    assert (tmp_path / as_of.isoformat() / "summary.json").exists()
