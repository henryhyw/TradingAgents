from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from tradingagents.system.config import SystemSettings
from tradingagents.system.context import RegimeAnalyzer
from tradingagents.system.data import YFinanceMarketDataProvider
from tradingagents.system.execution import PaperBroker
from tradingagents.system.monitoring.logging_utils import setup_logging
from tradingagents.system.orchestration.calendar_utils import default_as_of_date, next_market_days
from tradingagents.system.orchestration.reporting import generate_daily_report
from tradingagents.system.portfolio import PortfolioService
from tradingagents.system.research import (
    DeterministicResearchAdapter,
    ResearchAdapter,
    ResearchOrganization,
    TradingAgentsResearchAdapter,
)
from tradingagents.system.risk import RiskEngine
from tradingagents.system.schemas import (
    CandidateAssessment,
    DailyRunSummary,
    HealthCheckResult,
    OrderRecord,
    OrderStatus,
    PortfolioSnapshot,
    OrderIntentType,
    RegimeSnapshot,
    ResearchBundle,
    ResearchDecision,
    RiskDecision,
    RunMode,
    utc_now,
)
from tradingagents.system.storage.repository import TradingRepository
from tradingagents.system.universe import ScreenedAsset, UniverseSelector


logger = logging.getLogger(__name__)


class TradingSystemRunner:
    def __init__(
        self,
        settings: SystemSettings,
        deterministic_research: bool = False,
        verbose: bool = False,
        repository: TradingRepository | None = None,
        provider: YFinanceMarketDataProvider | None = None,
        research_adapter: ResearchAdapter | None = None,
    ):
        self.settings = settings
        setup_logging(settings.paths.logs_dir, verbose=verbose)
        self.repository = repository or TradingRepository(settings.paths.database_path)
        self.provider = provider or YFinanceMarketDataProvider(settings)
        self.selector = UniverseSelector(settings, self.provider)
        self.regime_analyzer = RegimeAnalyzer(settings, self.provider)
        self.risk_engine = RiskEngine(settings)
        self.portfolio_service = PortfolioService()
        self.broker = PaperBroker(settings, self.repository, self.provider)
        self.research: ResearchAdapter
        if research_adapter is not None:
            self.research = research_adapter
        elif deterministic_research:
            self.research = DeterministicResearchAdapter(self.provider, settings)
        else:
            self.research = TradingAgentsResearchAdapter(settings)
        self.research_org = ResearchOrganization(settings, self.provider, self.research)

    def resolve_as_of_date(self, as_of_date: date | None) -> date:
        return as_of_date or default_as_of_date(self.settings.run.market_timezone)

    def bootstrap(self, as_of_date: date | None = None) -> PortfolioSnapshot:
        resolved_date = self.resolve_as_of_date(as_of_date)
        return self.broker.bootstrap(resolved_date)

    def health_check(self, as_of_date: date | None = None) -> list[HealthCheckResult]:
        resolved_date = self.resolve_as_of_date(as_of_date)
        checks = [
            HealthCheckResult(
                name="database",
                status="ok",
                detail=str(self.settings.paths.database_path),
            ),
            HealthCheckResult(
                name="universe",
                status="ok",
                detail=f"{len(self.selector.load_universe())} symbols in curated universe",
            ),
            HealthCheckResult(
                name="default_model",
                status="ok" if self.settings.llm.model == "gpt-5.4-nano" else "warning",
                detail=f"default={self.settings.llm.model}",
            ),
        ]
        bar = self.provider.get_latest_bar("SPY", resolved_date)
        checks.append(
            HealthCheckResult(
                name="yfinance",
                status="ok" if bar is not None else "error",
                detail="Fetched SPY latest bar" if bar is not None else "Unable to fetch SPY latest bar",
            )
        )
        try:
            regime = self.regime_analyzer.analyze(resolved_date)
            checks.append(
                HealthCheckResult(
                    name="regime_model",
                    status="ok" if regime.data_quality == "ok" else "warning",
                    detail=f"{regime.label.value} (risk_on_score={regime.risk_on_score:+.2f})",
                )
            )
        except Exception as exc:  # pragma: no cover - defensive path
            checks.append(
                HealthCheckResult(
                    name="regime_model",
                    status="error",
                    detail=f"Unable to compute regime: {exc}",
                )
            )
        checks.append(
            HealthCheckResult(
                name="llm_credentials",
                status="ok" if self.settings.llm_ready() else "error",
                detail="OPENAI_API_KEY detected" if self.settings.llm_ready() else "OPENAI_API_KEY missing",
            )
        )
        return checks

    def _daily_pnl_fraction(self, as_of_date: date, current_snapshot: PortfolioSnapshot) -> float:
        baseline = self.repository.get_first_portfolio_snapshot_for_date(as_of_date)
        if baseline is None or baseline.equity <= 0:
            return 0.0
        return (current_snapshot.equity - baseline.equity) / baseline.equity

    @staticmethod
    def _symbol_position(portfolio: PortfolioSnapshot, symbol: str):
        for position in portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    @staticmethod
    def _candidate_from_asset(asset: ScreenedAsset, as_of_date: date) -> CandidateAssessment:
        return CandidateAssessment(
            symbol=asset.symbol,
            as_of_date=as_of_date,
            name=asset.name,
            asset_type=asset.asset_type,
            sector=asset.sector,
            style_tags=asset.style_tags,
            benchmark_symbol=asset.benchmark_symbol,
            peer_group=asset.peer_group,
            eligible=not asset.rejection_reasons,
            watchlist_only=asset.watchlist_only,
            eligibility_reasons=asset.rejection_reasons,
            close=asset.close,
            avg_dollar_volume_20d=asset.avg_dollar_volume_20d,
            return_20d=asset.return_20d,
            return_60d=asset.return_60d,
            volatility_20d=asset.volatility_20d,
            relative_strength_20d=asset.relative_strength_20d,
            regime_fit_score=asset.regime_fit_score,
            ranking_score=asset.score,
            ranking_breakdown=asset.ranking_breakdown,
            shortlist_reason=asset.shortlist_reason,
            data_quality_warnings=asset.quality_warnings,
        )

    def _sector_exposure_fraction(
        self,
        portfolio: PortfolioSnapshot,
        sector: str,
        sector_by_symbol: dict[str, str],
    ) -> float:
        if portfolio.equity <= 0:
            return 0.0
        exposure = 0.0
        for position in portfolio.positions:
            if sector_by_symbol.get(position.symbol, "Unknown") == sector:
                exposure += position.market_value
        return exposure / portfolio.equity

    def _max_correlation_to_book(self, symbol: str, portfolio: PortfolioSnapshot, as_of_date: date) -> float | None:
        symbols = [position.symbol for position in portfolio.positions if position.quantity > 0 and position.symbol != symbol]
        if not symbols:
            return None
        lookback = self.settings.data.correlation_lookback_days
        histories = self.provider.batch_get_history([symbol] + symbols, as_of_date, lookback + 20)
        base_history = histories.get(symbol)
        if base_history is None or base_history.empty:
            return None
        base_returns = (
            base_history.assign(Date=pd.to_datetime(base_history["Date"]).dt.tz_localize(None))
            .set_index("Date")["Close"]
            .pct_change()
            .dropna()
        )
        if base_returns.empty:
            return None
        max_corr: float | None = None
        for peer in symbols:
            peer_history = histories.get(peer)
            if peer_history is None or peer_history.empty:
                continue
            peer_returns = (
                peer_history.assign(Date=pd.to_datetime(peer_history["Date"]).dt.tz_localize(None))
                .set_index("Date")["Close"]
                .pct_change()
                .dropna()
            )
            aligned = pd.concat([base_returns, peer_returns], axis=1, join="inner").dropna()
            if len(aligned) < 20:
                continue
            corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            if pd.isna(corr):
                continue
            corr = max(0.0, corr)
            max_corr = corr if max_corr is None else max(max_corr, corr)
        return max_corr

    def _cooldown_active(self, symbol: str, as_of_date: date) -> bool:
        recent_loss = self.repository.has_recent_losing_exit(
            symbol=symbol,
            as_of_date=as_of_date,
            lookback_days=self.settings.risk.cooldown_days_after_loss,
        )
        recent_rejection = self.repository.has_recent_rejection(
            symbol=symbol,
            as_of_date=as_of_date,
            lookback_days=self.settings.risk.cooldown_days_after_rejection,
        )
        return recent_loss or recent_rejection

    def _shortlist_with_context(
        self,
        run_date: date,
        shortlist_size: int,
        symbols: list[str] | None,
        include_symbols: list[str],
        regime: RegimeSnapshot | None,
    ) -> tuple[list[ScreenedAsset], list[ScreenedAsset]]:
        if symbols:
            shortlist = self.selector.screen_symbols(symbols, run_date, regime=regime)
            return shortlist, shortlist
        screened = self.selector.screen_universe(run_date, regime=regime)
        shortlist = self.selector.build_shortlist_from_screened(
            screened=screened,
            shortlist_size=shortlist_size,
            include_symbols=include_symbols,
        )
        return screened, shortlist

    def run_once(
        self,
        as_of_date: date | None = None,
        mode: RunMode = RunMode.DRY_RUN,
        shortlist_size: int | None = None,
        execute: bool = False,
        symbols: list[str] | None = None,
    ) -> DailyRunSummary:
        run_date = self.resolve_as_of_date(as_of_date)
        started_at = utc_now()
        warnings: list[str] = []
        baseline_portfolio = self.bootstrap(run_date)
        existing_symbols = [position.symbol for position in baseline_portfolio.positions]

        regime: RegimeSnapshot | None = None
        try:
            regime = self.regime_analyzer.analyze(run_date)
            self.repository.save_regime_snapshot(regime)
        except Exception as exc:  # pragma: no cover - defensive path
            warnings.append(f"regime_unavailable:{type(exc).__name__}")
            logger.warning("Regime analysis failed for %s: %s", run_date, exc)

        screened, shortlist = self._shortlist_with_context(
            run_date=run_date,
            shortlist_size=shortlist_size or self.settings.run.default_shortlist_size,
            symbols=symbols,
            include_symbols=existing_symbols,
            regime=regime,
        )

        candidates = [self._candidate_from_asset(asset, run_date) for asset in screened]
        for candidate in candidates:
            self.repository.save_candidate_assessment(candidate)
        candidate_by_symbol = {candidate.symbol: candidate for candidate in candidates}
        sector_by_symbol = {candidate.symbol: candidate.sector for candidate in candidates}

        research_decisions: list[ResearchDecision] = []
        research_bundles: list[ResearchBundle] = []
        risk_decisions: list[RiskDecision] = []
        portfolio_fits = []
        execution_plans = []
        orders: list[OrderRecord] = []
        fills = []
        rejected_symbols: dict[str, str] = {}

        for asset in shortlist:
            logger.info("Researching %s", asset.symbol)
            candidate = candidate_by_symbol.get(asset.symbol) or self._candidate_from_asset(asset, run_date)
            decision, bundle = self.research_org.run(asset.symbol, run_date, candidate, regime)
            self.repository.save_research_decision(decision)
            research_decisions.append(decision)

            current_portfolio = self.broker.get_portfolio_snapshot(run_date)
            current_position = self._symbol_position(current_portfolio, asset.symbol)
            market_bar = self.provider.get_latest_bar(asset.symbol, run_date)
            earnings_event = self.provider.get_earnings_event(asset.symbol, run_date)
            sector_exposure_fraction = self._sector_exposure_fraction(
                current_portfolio,
                candidate.sector,
                sector_by_symbol=sector_by_symbol,
            )
            max_corr = self._max_correlation_to_book(asset.symbol, current_portfolio, run_date)
            cooldown_active = self._cooldown_active(asset.symbol, run_date)

            risk_decision = self.risk_engine.evaluate(
                decision=decision,
                portfolio=current_portfolio,
                current_position=current_position,
                market_bar=market_bar,
                avg_dollar_volume_20d=asset.avg_dollar_volume_20d,
                earnings_event=earnings_event,
                daily_pnl_fraction=self._daily_pnl_fraction(run_date, current_portfolio),
                opening_trades_today=self.repository.count_opening_orders_for_symbol(asset.symbol, run_date),
                losing_exits_today=self.repository.count_losing_exits(run_date),
                as_of_date=run_date,
                candidate=candidate,
                regime=regime,
                sector_exposure_fraction=sector_exposure_fraction,
                max_correlation_to_book=max_corr,
                cooldown_active=cooldown_active,
            )
            self.repository.save_risk_decision(risk_decision)
            risk_decisions.append(risk_decision)

            fit = self.portfolio_service.assess_portfolio_fit(
                decision=decision,
                risk_decision=risk_decision,
                portfolio=current_portfolio,
                current_position=current_position,
                market_bar=market_bar,
                candidate=candidate,
                regime=regime,
                max_correlation_to_book=max_corr,
            )
            self.repository.save_portfolio_fit_assessment(fit)
            portfolio_fits.append(fit)

            plan = self.portfolio_service.build_execution_plan(
                fit=fit,
                decision=decision,
                portfolio=current_portfolio,
                market_bar=market_bar,
                current_position=current_position,
            )
            self.repository.save_execution_plan(plan)
            execution_plans.append(plan)

            bundle = bundle.model_copy(
                update={
                    "portfolio_fit_id": fit.fit_id,
                    "execution_plan_id": plan.plan_id,
                    "risk_committee_note": "; ".join(risk_decision.committee_notes[:3]) if risk_decision.committee_notes else None,
                }
            )
            self.repository.save_research_bundle(bundle)
            research_bundles.append(bundle)

            order_intent = self.portfolio_service.build_order_intent_from_plan(plan, decision, risk_decision)
            if order_intent is None:
                if not risk_decision.approved:
                    rejected_symbols[asset.symbol] = risk_decision.rejection_reason or "risk_rejected"
                elif fit.recommended_action == OrderIntentType.HOLD:
                    rejected_symbols.setdefault(asset.symbol, "portfolio_hold")
                continue

            intent_status = OrderStatus.PENDING if not execute else OrderStatus.NEW
            self.repository.save_order_intent(order_intent, intent_status)

            if execute:
                order, fill = self.broker.submit_order(order_intent, run_date)
                orders.append(order)
                if fill is not None:
                    fills.append(fill)
                if order.status == OrderStatus.REJECTED:
                    rejected_symbols[asset.symbol] = "; ".join(order.notes)
            else:
                rejected_symbols.setdefault(asset.symbol, "dry_run_not_executed")

        final_portfolio = self.broker.get_portfolio_snapshot(run_date)
        summary = DailyRunSummary(
            mode=mode,
            as_of_date=run_date,
            started_at=started_at,
            completed_at=utc_now(),
            status="completed",
            universe_size=len(screened),
            eligible_universe_size=len([candidate for candidate in candidates if candidate.eligible]),
            regime_label=None if regime is None else regime.label.value,
            regime_risk_budget=None if regime is None else regime.risk_budget_multiplier,
            shortlisted_symbols=[asset.symbol for asset in shortlist],
            watchlist_symbols=[candidate.symbol for candidate in candidates if candidate.watchlist_only],
            approved_symbols=[decision.symbol for decision in risk_decisions if decision.approved],
            rejected_symbols=rejected_symbols,
            orders_submitted=len(orders),
            fills_completed=len(fills),
            notes=[],
            warnings=warnings,
        )
        report_path = generate_daily_report(
            report_root=self.settings.paths.reports_dir,
            as_of_date=run_date,
            summary=summary,
            shortlist=shortlist,
            research_decisions=research_decisions,
            risk_decisions=risk_decisions,
            orders=orders,
            fills=fills,
            portfolio=final_portfolio,
            regime_snapshot=regime,
            candidate_assessments=candidates,
            research_bundles=research_bundles,
            portfolio_fits=portfolio_fits,
            execution_plans=execution_plans,
        )
        summary = summary.model_copy(update={"completed_at": utc_now(), "report_path": str(report_path)})
        self.repository.save_daily_run_summary(summary)
        logger.info("Completed %s run for %s", mode.value, run_date)
        return summary

    def replay(
        self,
        start_date: date,
        end_date: date,
        execute: bool = False,
        shortlist_size: int | None = None,
    ) -> list[DailyRunSummary]:
        summaries: list[DailyRunSummary] = []
        for market_day in next_market_days(start_date, end_date):
            summaries.append(
                self.run_once(
                    as_of_date=market_day,
                    mode=RunMode.REPLAY,
                    shortlist_size=shortlist_size,
                    execute=execute,
                )
            )
        return summaries

    def generate_report_from_storage(self, as_of_date: date) -> str | None:
        def _dedupe_latest_by_symbol(items):
            deduped = {}
            for item in sorted(items, key=lambda candidate: candidate.timestamp):
                deduped[item.symbol] = item
            return list(deduped.values())

        research_decisions = _dedupe_latest_by_symbol(self.repository.list_research_decisions_for_date(as_of_date))
        research_bundles = _dedupe_latest_by_symbol(self.repository.list_research_bundles_for_date(as_of_date))
        risk_decisions = _dedupe_latest_by_symbol(self.repository.list_risk_decisions_for_date(as_of_date))
        portfolio_fits = _dedupe_latest_by_symbol(self.repository.list_portfolio_fit_assessments_for_date(as_of_date))
        execution_plans = _dedupe_latest_by_symbol(self.repository.list_execution_plans_for_date(as_of_date))
        candidates = self.repository.list_candidate_assessments_for_date(as_of_date)
        regime = self.repository.get_regime_snapshot_for_date(as_of_date)
        summary = self.repository.get_run_summary_for_date(as_of_date)
        shortlist_symbols: list[str] = []
        if summary is not None:
            shortlist_symbols = summary.shortlisted_symbols
        if not shortlist_symbols:
            shortlist_symbols = [decision.symbol for decision in research_decisions]

        orders = [order for order in self.repository.list_recent_orders(400) if order.as_of_date == as_of_date]
        fills = self.repository.list_fills_for_date(as_of_date)
        portfolio = self.broker.get_portfolio_snapshot(as_of_date)
        relevant_symbols = set(shortlist_symbols)
        relevant_symbols.update(position.symbol for position in portfolio.positions)
        relevant_symbols.update(order.symbol for order in orders)
        if relevant_symbols:
            research_decisions = [decision for decision in research_decisions if decision.symbol in relevant_symbols]
            research_bundles = [bundle for bundle in research_bundles if bundle.symbol in relevant_symbols]
            risk_decisions = [decision for decision in risk_decisions if decision.symbol in relevant_symbols]
            portfolio_fits = [fit for fit in portfolio_fits if fit.symbol in relevant_symbols]
            execution_plans = [plan for plan in execution_plans if plan.symbol in relevant_symbols]
            candidates = [candidate for candidate in candidates if candidate.symbol in relevant_symbols]

        candidate_map = {candidate.symbol: candidate for candidate in candidates}
        shortlist: list[ScreenedAsset] = []
        for symbol in shortlist_symbols:
            candidate = candidate_map.get(symbol)
            bar = self.provider.get_latest_bar(symbol, as_of_date)
            close = 0.0 if bar is None else bar.close
            if candidate is not None:
                shortlist.append(
                    ScreenedAsset(
                        symbol=symbol,
                        name=candidate.name,
                        asset_type=candidate.asset_type,
                        sector=candidate.sector,
                        style_tags=candidate.style_tags,
                        benchmark_symbol=candidate.benchmark_symbol,
                        peer_group=candidate.peer_group,
                        close=close,
                        avg_dollar_volume_20d=candidate.avg_dollar_volume_20d,
                        return_20d=candidate.return_20d,
                        return_60d=candidate.return_60d,
                        volatility_20d=candidate.volatility_20d,
                        relative_strength_20d=candidate.relative_strength_20d,
                        regime_fit_score=candidate.regime_fit_score,
                        score=candidate.ranking_score,
                        ranking_breakdown=candidate.ranking_breakdown,
                        rejection_reasons=[] if candidate.eligible else candidate.eligibility_reasons,
                        quality_warnings=candidate.data_quality_warnings,
                        watchlist_only=candidate.watchlist_only,
                        shortlist_reason=candidate.shortlist_reason,
                    )
                )
            else:
                shortlist.append(
                    ScreenedAsset(
                        symbol=symbol,
                        name=symbol,
                        asset_type="Equity",
                        sector="Unknown",
                        close=close,
                        avg_dollar_volume_20d=0.0,
                        return_20d=0.0,
                        return_60d=0.0,
                        volatility_20d=0.0,
                        score=0.0,
                    )
                )

        report_path = generate_daily_report(
            report_root=self.settings.paths.reports_dir,
            as_of_date=as_of_date,
            summary=summary,
            shortlist=shortlist,
            research_decisions=research_decisions,
            risk_decisions=risk_decisions,
            orders=orders,
            fills=fills,
            portfolio=portfolio,
            regime_snapshot=regime,
            candidate_assessments=candidates,
            research_bundles=research_bundles,
            portfolio_fits=portfolio_fits,
            execution_plans=execution_plans,
        )
        return str(report_path)
