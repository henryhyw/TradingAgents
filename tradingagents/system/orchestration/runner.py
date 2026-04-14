from __future__ import annotations

import logging
from datetime import date

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import YFinanceMarketDataProvider
from tradingagents.system.execution import PaperBroker
from tradingagents.system.monitoring.logging_utils import setup_logging
from tradingagents.system.orchestration.calendar_utils import default_as_of_date, next_market_days
from tradingagents.system.orchestration.reporting import generate_daily_report
from tradingagents.system.portfolio import PortfolioService
from tradingagents.system.research import DeterministicResearchAdapter, ResearchAdapter, TradingAgentsResearchAdapter
from tradingagents.system.risk import RiskEngine
from tradingagents.system.schemas import DailyRunSummary, HealthCheckResult, OrderRecord, OrderStatus, PortfolioSnapshot, ResearchDecision, RiskDecision, RunMode, utc_now
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
                detail=f"{len(self.selector.load_universe())} symbols in curated phase-1 universe",
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

    def _symbol_position(self, portfolio: PortfolioSnapshot, symbol: str):
        for position in portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

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
        baseline_portfolio = self.bootstrap(run_date)
        existing_symbols = [position.symbol for position in baseline_portfolio.positions]

        if symbols:
            shortlist = self.selector.screen_symbols(symbols, run_date)
        else:
            shortlist = self.selector.build_shortlist(
                run_date,
                shortlist_size or self.settings.run.default_shortlist_size,
                include_symbols=existing_symbols,
            )

        research_decisions: list[ResearchDecision] = []
        risk_decisions: list[RiskDecision] = []
        orders: list[OrderRecord] = []
        fills = []
        rejected_symbols: dict[str, str] = {}

        for asset in shortlist:
            logger.info("Researching %s", asset.symbol)
            decision = self.research.research(asset.symbol, run_date)
            self.repository.save_research_decision(decision)
            research_decisions.append(decision)

            current_portfolio = self.broker.get_portfolio_snapshot(run_date)
            current_position = self._symbol_position(current_portfolio, asset.symbol)
            market_bar = self.provider.get_latest_bar(asset.symbol, run_date)
            earnings_event = self.provider.get_earnings_event(asset.symbol, run_date)
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
            )
            self.repository.save_risk_decision(risk_decision)
            risk_decisions.append(risk_decision)

            order_intent = self.portfolio_service.build_order_intent(
                decision=decision,
                risk_decision=risk_decision,
                portfolio=current_portfolio,
                current_position=current_position,
                market_bar=market_bar,
            )

            if order_intent is None:
                if not risk_decision.approved:
                    rejected_symbols[asset.symbol] = risk_decision.rejection_reason or "risk_rejected"
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
        preliminary_summary = DailyRunSummary(
            mode=mode,
            as_of_date=run_date,
            started_at=started_at,
            completed_at=utc_now(),
            status="completed",
            shortlisted_symbols=[asset.symbol for asset in shortlist],
            approved_symbols=[decision.symbol for decision in risk_decisions if decision.approved],
            rejected_symbols=rejected_symbols,
            orders_submitted=len(orders),
            fills_completed=len(fills),
            notes=[],
        )
        report_path = generate_daily_report(
            report_root=self.settings.paths.reports_dir,
            as_of_date=run_date,
            summary=preliminary_summary,
            shortlist=shortlist,
            research_decisions=research_decisions,
            risk_decisions=risk_decisions,
            orders=orders,
            fills=fills,
            portfolio=final_portfolio,
        )
        summary = DailyRunSummary(
            mode=mode,
            as_of_date=run_date,
            started_at=started_at,
            completed_at=utc_now(),
            status="completed",
            shortlisted_symbols=[asset.symbol for asset in shortlist],
            approved_symbols=[decision.symbol for decision in risk_decisions if decision.approved],
            rejected_symbols=rejected_symbols,
            orders_submitted=len(orders),
            fills_completed=len(fills),
            report_path=str(report_path),
            notes=[],
        )
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
        shortlist = [ScreenedAsset.model_validate(asset.model_dump()) for asset in self.selector.build_shortlist(as_of_date, self.settings.run.default_shortlist_size)]
        research_decisions = self.repository.list_research_decisions_for_date(as_of_date)
        risk_decisions = self.repository.list_risk_decisions_for_date(as_of_date)
        orders = [order for order in self.repository.list_recent_orders(200) if order.as_of_date == as_of_date]
        fills = self.repository.list_fills_for_date(as_of_date)
        portfolio = self.broker.get_portfolio_snapshot(as_of_date)
        summary = self.repository.get_run_summary_for_date(as_of_date)
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
        )
        return str(report_path)
