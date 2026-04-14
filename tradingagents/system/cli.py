from __future__ import annotations

from datetime import date
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from tradingagents.system.config import load_settings
from tradingagents.system.orchestration.runner import TradingSystemRunner
from tradingagents.system.orchestration.scheduler import DailyScheduler
from tradingagents.system.schemas import RunMode


app = typer.Typer(help="Professional local paper-trading system for US equities.")
console = Console()


def _parse_date(value: str | None) -> date | None:
    return None if value is None else date.fromisoformat(value)


def _runner(config: str | None, deterministic_research: bool, verbose: bool) -> TradingSystemRunner:
    settings = load_settings(config)
    return TradingSystemRunner(settings, deterministic_research=deterministic_research, verbose=verbose)


@app.command("bootstrap")
def bootstrap(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Bootstrap date in YYYY-MM-DD.")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    snapshot = runner.bootstrap(_parse_date(as_of))
    console.print(f"Database: {runner.settings.paths.database_path}")
    console.print(f"Reports: {runner.settings.paths.reports_dir}")
    console.print(f"Artifacts: {runner.settings.paths.artifacts_dir}")
    console.print(f"Initial equity: ${snapshot.equity:,.2f}")


@app.command("setup")
def setup_cmd(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Bootstrap date in YYYY-MM-DD.")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    bootstrap(config=config, as_of=as_of, verbose=verbose)


@app.command("health-check")
def health_check(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Check date in YYYY-MM-DD.")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    checks = runner.health_check(_parse_date(as_of))
    table = Table(title="System Health")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for check in checks:
        table.add_row(check.name, check.status, check.detail)
    console.print(table)
    if any(check.status == "error" for check in checks):
        raise typer.Exit(code=1)


@app.command("show-config")
def show_config(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
) -> None:
    settings = load_settings(config)
    table = Table(title="System Configuration")
    table.add_column("Key")
    table.add_column("Value")
    rows = [
        ("repo_root", str(settings.repo_root)),
        ("database", str(settings.paths.database_path)),
        ("reports_dir", str(settings.paths.reports_dir)),
        ("model", settings.llm.model),
        ("data_provider", settings.data.provider),
        ("shortlist_size", str(settings.run.default_shortlist_size)),
        ("max_per_sector", str(settings.run.max_shortlist_per_sector)),
        ("max_position_size", f"{settings.risk.max_position_size_fraction:.2%}"),
        ("max_gross_exposure", f"{settings.risk.max_gross_exposure_fraction:.2%}"),
        ("max_sector_exposure", f"{settings.risk.max_sector_exposure_fraction:.2%}"),
    ]
    for key, value in rows:
        table.add_row(key, value)
    console.print(table)


@app.command("run-once")
def run_once(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Run date in YYYY-MM-DD.")] = None,
    shortlist_size: Annotated[int | None, typer.Option(help="Shortlist size override.")] = None,
    execute: Annotated[bool, typer.Option("--execute/--no-execute", help="Submit paper trades after risk approval.")] = True,
    symbols: Annotated[str | None, typer.Option(help="Comma-separated symbol override.")] = None,
    deterministic_research: Annotated[bool, typer.Option(hidden=True)] = False,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=deterministic_research, verbose=verbose)
    try:
        summary = runner.run_once(
            as_of_date=_parse_date(as_of),
            mode=RunMode.PAPER if execute else RunMode.DRY_RUN,
            shortlist_size=shortlist_size,
            execute=execute,
            symbols=None if not symbols else [item.strip().upper() for item in symbols.split(",") if item.strip()],
        )
    except Exception as exc:
        console.print(f"Run failed: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Run complete: {summary.mode.value} {summary.as_of_date.isoformat()}")
    if summary.regime_label:
        console.print(f"Regime: {summary.regime_label} (risk budget {summary.regime_risk_budget:.2f})")
    console.print(f"Report: {summary.report_path}")
    console.print(f"Orders submitted: {summary.orders_submitted}")
    console.print(f"Fills completed: {summary.fills_completed}")


@app.command("dry-run")
def dry_run(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Run date in YYYY-MM-DD.")] = None,
    shortlist_size: Annotated[int | None, typer.Option(help="Shortlist size override.")] = None,
    symbols: Annotated[str | None, typer.Option(help="Comma-separated symbol override.")] = None,
    deterministic_research: Annotated[bool, typer.Option(hidden=True)] = False,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    run_once(
        config=config,
        as_of=as_of,
        shortlist_size=shortlist_size,
        execute=False,
        symbols=symbols,
        deterministic_research=deterministic_research,
        verbose=verbose,
    )


@app.command("run-daily")
def run_daily(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    run_at: Annotated[str, typer.Option(help="Daily NYSE-local trigger time HH:MM.")] = "15:45",
    shortlist_size: Annotated[int | None, typer.Option(help="Shortlist size override.")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    scheduler = DailyScheduler(runner)
    scheduler.run_forever(run_at=run_at, shortlist_size=shortlist_size, execute=True)


@app.command("replay")
def replay(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    start: Annotated[str, typer.Option(help="Start date YYYY-MM-DD.")] = ...,
    end: Annotated[str, typer.Option(help="End date YYYY-MM-DD.")] = ...,
    shortlist_size: Annotated[int | None, typer.Option(help="Shortlist size override.")] = None,
    execute: Annotated[bool, typer.Option("--execute/--no-execute", help="Submit simulated paper trades during replay.")] = False,
    deterministic_research: Annotated[bool, typer.Option(hidden=True)] = False,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=deterministic_research, verbose=verbose)
    try:
        summaries = runner.replay(date.fromisoformat(start), date.fromisoformat(end), execute=execute, shortlist_size=shortlist_size)
    except Exception as exc:
        console.print(f"Replay failed: {exc}")
        raise typer.Exit(code=1) from exc
    console.print(f"Replay completed for {len(summaries)} market sessions.")


@app.command("show-positions")
def show_positions(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Valuation date in YYYY-MM-DD.")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    snapshot = runner.broker.get_portfolio_snapshot(runner.resolve_as_of_date(_parse_date(as_of)))
    table = Table(title="Positions")
    table.add_column("Symbol")
    table.add_column("Quantity", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Market Price", justify="right")
    table.add_column("Market Value", justify="right")
    table.add_column("Unrealized PnL", justify="right")
    for position in snapshot.positions:
        table.add_row(
            position.symbol,
            str(position.quantity),
            f"${position.avg_cost:,.2f}",
            f"${position.market_price:,.2f}",
            f"${position.market_value:,.2f}",
            f"${position.unrealized_pnl:,.2f}",
        )
    console.print(table)
    console.print(f"Cash: ${snapshot.cash:,.2f} | Equity: ${snapshot.equity:,.2f}")


@app.command("show-recent-orders")
def show_recent_orders(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum number of orders.")] = 20,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    table = Table(title="Recent Orders")
    table.add_column("Timestamp")
    table.add_column("Symbol")
    table.add_column("Side")
    table.add_column("Qty", justify="right")
    table.add_column("Status")
    table.add_column("Fill Price", justify="right")
    for order in runner.repository.list_recent_orders(limit):
        table.add_row(
            order.timestamp.isoformat(),
            order.symbol,
            order.side.value.upper(),
            str(order.quantity),
            order.status.value,
            "n/a" if order.fill_price is None else f"${order.fill_price:,.2f}",
        )
    console.print(table)


@app.command("show-regime")
def show_regime(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Date in YYYY-MM-DD.")] = None,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    run_date = runner.resolve_as_of_date(_parse_date(as_of))
    regime = runner.repository.get_regime_snapshot_for_date(run_date)
    if regime is None:
        regime = runner.regime_analyzer.analyze(run_date)
    table = Table(title=f"Regime Snapshot ({run_date.isoformat()})")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("label", regime.label.value)
    table.add_row("trend_regime", regime.trend_regime)
    table.add_row("volatility_regime", regime.volatility_regime)
    table.add_row("risk_on_score", f"{regime.risk_on_score:+.2f}")
    table.add_row("risk_budget_multiplier", f"{regime.risk_budget_multiplier:.2f}")
    table.add_row("max_gross_exposure", f"{regime.max_gross_exposure_fraction:.2%}")
    console.print(table)


@app.command("show-candidates")
def show_candidates(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str | None, typer.Option(help="Date in YYYY-MM-DD.")] = None,
    limit: Annotated[int, typer.Option(help="Maximum rows to display.")] = 20,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    run_date = runner.resolve_as_of_date(_parse_date(as_of))
    candidates = runner.repository.list_candidate_assessments_for_date(run_date)
    table = Table(title=f"Candidates ({run_date.isoformat()})")
    table.add_column("Symbol")
    table.add_column("Sector")
    table.add_column("Eligible")
    table.add_column("Watchlist")
    table.add_column("Score", justify="right")
    table.add_column("Reason")
    for candidate in sorted(candidates, key=lambda item: item.ranking_score, reverse=True)[:limit]:
        table.add_row(
            candidate.symbol,
            candidate.sector,
            "yes" if candidate.eligible else "no",
            "yes" if candidate.watchlist_only else "no",
            f"{candidate.ranking_score:.3f}",
            candidate.shortlist_reason or ",".join(candidate.eligibility_reasons[:1]) or "-",
        )
    console.print(table)


@app.command("generate-daily-report")
def generate_daily_report_cmd(
    config: Annotated[str | None, typer.Option(help="Optional TOML config path.")] = None,
    as_of: Annotated[str, typer.Option(help="Report date in YYYY-MM-DD.")] = ...,
    verbose: Annotated[bool, typer.Option(help="Enable verbose logging.")] = False,
) -> None:
    runner = _runner(config, deterministic_research=False, verbose=verbose)
    report_path = runner.generate_report_from_storage(date.fromisoformat(as_of))
    console.print(f"Report written to {report_path}")
