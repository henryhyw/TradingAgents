from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from tradingagents.system.schemas import (
    CandidateAssessment,
    DailyRunSummary,
    ExecutionPlan,
    FillRecord,
    OrderRecord,
    PortfolioFitAssessment,
    PortfolioSnapshot,
    RegimeSnapshot,
    ResearchBundle,
    ResearchDecision,
    RiskDecision,
)
from tradingagents.system.universe import ScreenedAsset


def _section(lines: list[str], title: str) -> None:
    lines.extend(["", f"## {title}"])


def _format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def generate_daily_report(
    report_root: Path,
    as_of_date: date,
    summary: DailyRunSummary | None,
    shortlist: list[ScreenedAsset],
    research_decisions: list[ResearchDecision],
    risk_decisions: list[RiskDecision],
    orders: list[OrderRecord],
    fills: list[FillRecord],
    portfolio: PortfolioSnapshot,
    regime_snapshot: RegimeSnapshot | None = None,
    candidate_assessments: list[CandidateAssessment] | None = None,
    research_bundles: list[ResearchBundle] | None = None,
    portfolio_fits: list[PortfolioFitAssessment] | None = None,
    execution_plans: list[ExecutionPlan] | None = None,
) -> Path:
    report_dir = report_root / as_of_date.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_assessments = candidate_assessments or []
    research_bundles = research_bundles or []
    portfolio_fits = portfolio_fits or []
    execution_plans = execution_plans or []
    bundle_by_symbol = {bundle.symbol: bundle for bundle in research_bundles}

    payload = {
        "date": as_of_date.isoformat(),
        "summary": None if summary is None else summary.model_dump(mode="json"),
        "regime_snapshot": None if regime_snapshot is None else regime_snapshot.model_dump(mode="json"),
        "universe": {
            "shortlist": [asset.model_dump(mode="json") for asset in shortlist],
            "candidate_assessments": [candidate.model_dump(mode="json") for candidate in candidate_assessments],
        },
        "research_bundles": [bundle.model_dump(mode="json") for bundle in research_bundles],
        "research_decisions": [decision.model_dump(mode="json") for decision in research_decisions],
        "risk_decisions": [decision.model_dump(mode="json") for decision in risk_decisions],
        "portfolio_fit_assessments": [fit.model_dump(mode="json") for fit in portfolio_fits],
        "execution_plans": [plan.model_dump(mode="json") for plan in execution_plans],
        "orders": [order.model_dump(mode="json") for order in orders],
        "fills": [fill.model_dump(mode="json") for fill in fills],
        "portfolio": portfolio.model_dump(mode="json"),
    }
    (report_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines: list[str] = [f"# TradingAgents Daily Report ({as_of_date.isoformat()})"]
    if summary is not None:
        lines.append(
            f"Run mode: `{summary.mode.value}` | Status: `{summary.status}` | "
            f"Shortlist: {len(summary.shortlisted_symbols)} | Orders: {summary.orders_submitted} | Fills: {summary.fills_completed}"
        )

    _section(lines, "Regime Summary")
    if regime_snapshot is None:
        lines.append("- Regime snapshot unavailable.")
    else:
        lines.append(
            f"- Label: `{regime_snapshot.label.value}` | Trend: `{regime_snapshot.trend_regime}` | "
            f"Volatility: `{regime_snapshot.volatility_regime}`"
        )
        lines.append(
            f"- Risk-on score: {regime_snapshot.risk_on_score:+.2f} | "
            f"Risk budget multiplier: {regime_snapshot.risk_budget_multiplier:.2f} | "
            f"Gross cap: {_format_pct(regime_snapshot.max_gross_exposure_fraction)}"
        )
        for note in regime_snapshot.notes[:4]:
            lines.append(f"- Note: {note}")
        for warning in regime_snapshot.warnings[:4]:
            lines.append(f"- Warning: {warning}")

    _section(lines, "Universe & Discovery")
    if summary is not None:
        lines.append(
            f"- Universe size: {summary.universe_size} | Eligible: {summary.eligible_universe_size} | "
            f"Watchlist-only: {len(summary.watchlist_symbols)}"
        )
    if candidate_assessments:
        top_candidates = sorted(candidate_assessments, key=lambda item: item.ranking_score, reverse=True)[:10]
        for candidate in top_candidates:
            status = "watchlist" if candidate.watchlist_only else ("eligible" if candidate.eligible else "rejected")
            lines.append(
                f"- {candidate.symbol}: {status}, score={candidate.ranking_score:.3f}, "
                f"rel_strength20={candidate.relative_strength_20d:.2%}, regime_fit={candidate.regime_fit_score:.2f}, "
                f"pool={candidate.source_pool}, event_strength={candidate.event_strength_score:.2f}"
            )

    _section(lines, "Shortlist")
    for asset in shortlist:
        lines.append(
            f"- {asset.symbol}: score={asset.score:.3f}, close=${asset.close:.2f}, "
            f"20d={asset.return_20d:.2%}, 60d={asset.return_60d:.2%}, "
            f"ADV20=${asset.avg_dollar_volume_20d:,.0f}, pool={asset.source_pool}, "
            f"reason={asset.shortlist_reason or 'n/a'}"
        )

    _section(lines, "Research & Debate")
    for decision in research_decisions:
        bundle = bundle_by_symbol.get(decision.symbol)
        lines.append(
            f"- {decision.symbol}: {decision.action.value.upper()} ({decision.confidence:.2f}) horizon={decision.time_horizon}"
        )
        lines.append(
            f"  Entry mode: {decision.entry_mode.value} | trigger={decision.entry_trigger_reason or 'n/a'} | "
            f"extension_penalty={decision.extension_penalty:.2f} | overheat_penalty={decision.overheat_penalty:.2f}"
        )
        if decision.position_lifecycle_state is not None:
            lines.append(f"  Lifecycle state: {decision.position_lifecycle_state.value}")
        lines.append(f"  Thesis: {decision.thesis}")
        source_extra = decision.source_metadata.extra
        extension_metrics = source_extra.get("extension_metrics", decision.extension_metrics)
        if isinstance(extension_metrics, dict) and extension_metrics:
            lines.append(
                "  Entry metrics: "
                f"ext_ma20={float(extension_metrics.get('extension_over_ma20', 0.0)):.2%}, "
                f"rsi14={float(extension_metrics.get('rsi14', 50.0)):.1f}, "
                f"breakout_dist={float(extension_metrics.get('breakout_distance', 0.0)):.2%}"
            )
        if decision.source_metadata.parser_mode == "upstream_error_no_entry" or source_extra.get("fallback_origin"):
            lines.append("  Fallback: upstream failure/insufficient-research state (non-tradable no-entry).")
        if source_extra.get("buy_promotion_applied"):
            lines.append(
                f"  Promotion: BUY promoted via {source_extra.get('buy_promotion_source') or 'adjudication'} after validation."
            )
        if source_extra.get("entry_reject_class") and source_extra.get("entry_reject_class") != "none":
            lines.append(
                "  Entry reject class: "
                f"{source_extra.get('entry_reject_class')} | reasons={source_extra.get('entry_reject_reasons', [])}"
            )
        if source_extra.get("starter_entry_due_to_risk_on_bias"):
            lines.append(
                "  Starter entry: risk_on participation bias promoted a validated near-miss into a small starter position."
            )
        if source_extra.get("near_miss_not_promoted"):
            lines.append("  Near miss: not promoted because validation or risk conditions were insufficient.")
        if source_extra.get("hold_existing"):
            lines.append("  Position handling: hold_existing in current portfolio context.")
        if source_extra.get("buy_blocked_due_to_fallback"):
            lines.append("  Promotion Block: BUY blocked because research originated from upstream fallback.")
        if source_extra.get("buy_blocked_due_to_thesis_inconsistency"):
            lines.append("  Promotion Block: BUY blocked due to bearish/no-entry thesis semantics.")
        if source_extra.get("action_thesis_mismatch_detected"):
            lines.append("  Consistency: action/thesis mismatch detected and corrected.")
        if source_extra.get("final_action_downgraded"):
            lines.append("  Consistency: final action downgraded after semantic validation.")
        if source_extra.get("buy_rewrite_attempted"):
            lines.append(
                "  BUY rewrite: "
                f"attempted={bool(source_extra.get('buy_rewrite_attempted'))}, "
                f"success={bool(source_extra.get('buy_rewrite_success'))}, "
                f"failure={bool(source_extra.get('buy_rewrite_failure'))}"
            )
        if bundle is not None:
            lines.append(
                f"  Debate: winner={bundle.debate_summary.winning_side}, "
                f"balance={bundle.debate_summary.confidence_balance:.2f}, "
                f"final_action={bundle.debate_summary.final_action.value}, "
                f"aligned={bundle.debate_summary.aligned_with_final_action}"
            )
            if bundle.debate_summary.override_reason:
                lines.append(f"  Override: {bundle.debate_summary.override_reason}")
            lines.append(f"  Bull: {bundle.bull_case.summary}")
            lines.append(f"  Bear: {bundle.bear_case.summary}")

    _section(lines, "Risk Committee")
    for risk in risk_decisions:
        approval = "approved" if risk.approved else f"rejected ({risk.rejection_reason})"
        lines.append(f"- {risk.symbol}: {approval}, size={risk.approved_size_fraction:.3f}")
        for note in risk.committee_notes[:2]:
            lines.append(f"  Note: {note}")
        for warning in risk.warnings[:2]:
            lines.append(f"  Warning: {warning}")

    _section(lines, "Portfolio Manager")
    if not portfolio_fits:
        lines.append("- No portfolio fit assessments persisted for this run.")
    for fit in portfolio_fits:
        lines.append(
            f"- {fit.symbol}: action={fit.recommended_action.value}, fits={fit.fits_portfolio}, "
            f"current={fit.current_weight:.3f}, target={fit.target_weight:.3f}"
        )
        lines.append(f"  Rationale: {fit.rationale}")

    _section(lines, "Execution Planner")
    if not execution_plans:
        lines.append("- No execution plans persisted for this run.")
    for plan in execution_plans:
        lines.append(
            f"- {plan.symbol}: intent={plan.intent_type.value}, side={plan.side.value if plan.side else 'none'}, "
            f"qty={plan.quantity if plan.quantity is not None else 'n/a'}, target={plan.target_weight:.3f}"
        )
        for note in plan.notes[:2]:
            lines.append(f"  Note: {note}")

    _section(lines, "Orders & Fills")
    if not orders:
        lines.append("- No orders submitted.")
    for order in orders:
        lines.append(
            f"- {order.symbol}: {order.side.value.upper()} {order.quantity} status={order.status.value}, "
            f"fill={order.fill_price if order.fill_price is not None else 'n/a'}, commission={order.commission:.2f}"
        )
    if fills:
        for fill in fills:
            lines.append(
                f"  Fill {fill.fill_id}: {fill.symbol} {fill.side.value.upper()} {fill.quantity} @ {fill.price:.2f}, "
                f"realized={fill.realized_pnl if fill.realized_pnl is not None else 'n/a'}"
            )

    _section(lines, "Portfolio Snapshot")
    lines.append(f"- Cash: ${portfolio.cash:,.2f}")
    lines.append(f"- Equity: ${portfolio.equity:,.2f}")
    lines.append(f"- Gross Exposure: ${portfolio.gross_exposure:,.2f}")
    lines.append(f"- Daily Realized PnL: ${portfolio.daily_realized_pnl:,.2f}")
    lines.append(f"- Daily Unrealized PnL: ${portfolio.daily_unrealized_pnl:,.2f}")

    sector_map = {candidate.symbol: candidate.sector for candidate in candidate_assessments}
    sector_exposure: dict[str, float] = {}
    for position in portfolio.positions:
        sector = sector_map.get(position.symbol, "Unknown")
        sector_exposure[sector] = sector_exposure.get(sector, 0.0) + position.market_value
    if sector_exposure:
        _section(lines, "Concentration Summary")
        for sector, value in sorted(sector_exposure.items(), key=lambda item: item[1], reverse=True):
            fraction = value / portfolio.equity if portfolio.equity > 0 else 0.0
            lines.append(f"- {sector}: ${value:,.2f} ({_format_pct(fraction)})")

    warnings = []
    if summary is not None:
        warnings.extend(summary.warnings)
    warnings.extend([warning for bundle in research_bundles for warning in bundle.warnings])
    warnings.extend(
        [warning for candidate in candidate_assessments for warning in candidate.data_quality_warnings]
    )
    warnings = sorted(set(warnings))
    _section(lines, "Warnings & Data Quality")
    if not warnings:
        lines.append("- No material data-quality warnings recorded.")
    for warning in warnings[:20]:
        lines.append(f"- {warning}")

    _section(lines, "Diagnostics")
    if summary is None:
        lines.append("- Run summary unavailable; diagnostics omitted.")
    else:
        action_counts = summary.research_action_counts or {}
        if action_counts:
            lines.append(
                "- Research actions: "
                + ", ".join(f"{action.upper()}={count}" for action, count in sorted(action_counts.items()))
            )
        block_counts = summary.block_reason_counts or {}
        if block_counts:
            lines.append(
                "- Block reasons: "
                + ", ".join(f"{reason}={count}" for reason, count in sorted(block_counts.items()))
            )
        if summary.upstream_failure_counts:
            lines.append(
                "- Upstream failures: "
                + ", ".join(f"{error_type}={count}" for error_type, count in sorted(summary.upstream_failure_counts.items()))
            )
        lines.append(f"- Upstream retries: {summary.upstream_retry_count}")
        lines.append(
            "- BUY promotion diagnostics: "
            f"promoted={summary.promoted_buy_count}, "
            f"promoted_from_debate={summary.promoted_buy_from_debate_count}, "
            f"promoted_after_validation={summary.promoted_buy_after_validation_count}, "
            f"blocked_fallback={summary.blocked_buy_due_to_fallback_count}, "
            f"blocked_thesis={summary.blocked_buy_due_to_thesis_inconsistency_count}"
        )
        lines.append(
            "- Entry mode diagnostics: "
            + ", ".join(f"{mode}={count}" for mode, count in sorted((summary.entry_mode_counts or {}).items()))
        )
        lines.append(
            "- Entry block diagnostics: "
            f"extension={summary.buy_blocked_due_to_extension_count}, "
            f"overheat={summary.buy_blocked_due_to_overheat_count}, "
            f"missing_pullback={summary.buy_blocked_due_to_missing_pullback_confirmation_count}, "
            f"missing_breakout={summary.buy_blocked_due_to_missing_breakout_confirmation_count}"
        )
        lines.append(
            "- Entry balance diagnostics: "
            f"near_miss={summary.buy_near_miss_count}, "
            f"near_miss_breakout={summary.buy_near_miss_due_to_breakout_confirmation}, "
            f"near_miss_pullback={summary.buy_near_miss_due_to_pullback_confirmation}, "
            f"risk_on_participation_bias={summary.risk_on_participation_bias_applied_count}"
        )
        lines.append(
            "- Starter participation diagnostics: "
            f"starter_entry={summary.starter_entry_count}, "
            f"starter_bias={summary.starter_entry_due_to_risk_on_bias_count}, "
            f"starter_rejected={summary.starter_entry_rejected_count}, "
            f"near_miss_promoted={summary.near_miss_promoted_count}, "
            f"near_miss_not_promoted={summary.near_miss_not_promoted_count}"
        )
        lines.append(
            "- Reject classification diagnostics: "
            f"hard_reject={summary.hard_reject_count}, "
            f"soft_reject={summary.soft_reject_count}, "
            f"hold_existing={summary.hold_existing_count}, "
            f"starter_keep={summary.starter_keep_count}"
        )
        lines.append(
            "- Exit lifecycle diagnostics: "
            f"trim_partial={summary.trim_partial_count}, "
            f"reduce_to_core={summary.reduce_to_core_count}, "
            f"trend_failure={summary.trend_failure_exit_count}, "
            f"time_stop={summary.time_stop_exit_count}, "
            f"regime_exit={summary.regime_exit_count}"
        )
        lines.append(
            "- Risk-on exit balance: "
            f"full_exit_risk_reduction={summary.full_exit_due_to_risk_reduction_count}, "
            f"exit_to_trim={summary.full_exit_rejected_in_favor_of_trim_count}, "
            f"exit_to_reduce_core={summary.full_exit_rejected_in_favor_of_reduce_to_core_count}, "
            f"starter_kept={summary.starter_position_kept_due_to_regime_count}, "
            f"went_flat_in_risk_on={summary.went_flat_in_risk_on_count}, "
            f"flattening_justifications={summary.risk_on_flattening_justification_count}, "
            f"risk_on_no_trade={summary.repeated_risk_on_no_trade_count}, "
            f"risk_on_low_exposure={summary.repeated_risk_on_low_exposure_count}"
        )
        lines.append(
            "- Source pools: "
            + ", ".join(
                f"{pool}={count}" for pool, count in sorted((summary.source_pool_counts or {}).items())
            )
        )
        lines.append(
            "- Entry extension metrics: "
            f"avg_ext_ma20={summary.average_entry_extension_metrics.get('avg_extension_over_ma20', 0.0):.2%}, "
            f"avg_rsi14={summary.average_entry_extension_metrics.get('avg_entry_rsi14', 50.0):.1f}"
        )
        if summary.realized_vs_unrealized_by_exit_type:
            lines.append(
                "- Realized/Unrealized by exit type: "
                + ", ".join(
                    f"{name}={value:.2f}"
                    for name, value in sorted(summary.realized_vs_unrealized_by_exit_type.items())
                )
            )
        lines.append(
            "- Consistency diagnostics: "
            f"action_thesis_mismatch={summary.action_thesis_mismatch_count}, "
            f"fallback_origin_decisions={summary.fallback_origin_decision_count}, "
            f"final_action_changed={summary.final_action_changed_count}"
        )
        lines.append(
            "- Semantic guardrails: "
            f"fallback_buy_block={summary.fallback_buy_block_count}, "
            f"thesis_inconsistency_block={summary.thesis_inconsistency_block_count}, "
            f"buy_rewrite_attempt={summary.buy_rewrite_attempt_count}, "
            f"buy_rewrite_success={summary.buy_rewrite_success_count}, "
            f"buy_rewrite_failure={summary.buy_rewrite_failure_count}, "
            f"final_action_downgrade={summary.final_action_downgrade_count}, "
            f"inconsistent_buy_prevented={summary.inconsistent_buy_prevented_count}"
        )
        lines.append(f"- Flat-book suppressed: {summary.flat_book_suppressed}")

    report_path = report_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
