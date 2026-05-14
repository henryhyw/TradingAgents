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
        entry_mode="breakout",
        entry_trigger_reason="breakout_confirmation_passed",
        extension_penalty=0.12,
        overheat_penalty=0.05,
        extension_metrics={"extension_over_ma20": 0.04, "rsi14": 66.0, "breakout_distance": 0.02},
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
        promoted_buy_count=1,
        promoted_buy_from_debate_count=1,
        blocked_buy_due_to_fallback_count=0,
        blocked_buy_due_to_thesis_inconsistency_count=0,
        action_thesis_mismatch_count=0,
        fallback_origin_decision_count=0,
        final_action_changed_count=1,
        fallback_buy_block_count=0,
        thesis_inconsistency_block_count=0,
        buy_rewrite_attempt_count=1,
        buy_rewrite_success_count=1,
        buy_rewrite_failure_count=0,
        final_action_downgrade_count=0,
        inconsistent_buy_prevented_count=0,
        entry_mode_counts={"breakout": 1, "pullback": 0, "none": 0},
        promoted_buy_after_validation_count=1,
        buy_blocked_due_to_extension_count=0,
        buy_blocked_due_to_overheat_count=0,
        buy_blocked_due_to_missing_pullback_confirmation_count=0,
        buy_blocked_due_to_missing_breakout_confirmation_count=0,
        buy_near_miss_count=1,
        buy_near_miss_due_to_breakout_confirmation=1,
        buy_near_miss_due_to_pullback_confirmation=0,
        risk_on_participation_bias_applied_count=1,
        trim_partial_count=0,
        reduce_to_core_count=0,
        trend_failure_exit_count=0,
        time_stop_exit_count=0,
        regime_exit_count=0,
        full_exit_due_to_risk_reduction_count=0,
        full_exit_rejected_in_favor_of_trim_count=0,
        full_exit_rejected_in_favor_of_reduce_to_core_count=0,
        starter_position_kept_due_to_regime_count=0,
        went_flat_in_risk_on_count=0,
        risk_on_flattening_justification_count=0,
        source_pool_counts={"industry_leader": 1},
        average_entry_extension_metrics={"avg_extension_over_ma20": 0.04, "avg_entry_rsi14": 66.0},
        realized_vs_unrealized_by_exit_type={"realized_total": 0.0, "unrealized_total": 0.0},
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
    assert "BUY promotion diagnostics" in content
    assert "Entry mode diagnostics" in content
    assert "Entry balance diagnostics" in content
    assert "Exit lifecycle diagnostics" in content
    assert "Risk-on exit balance" in content
    assert "Consistency diagnostics" in content
    assert "Semantic guardrails" in content
    assert (tmp_path / as_of.isoformat() / "summary.json").exists()
