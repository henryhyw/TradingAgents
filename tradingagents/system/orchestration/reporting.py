from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from tradingagents.system.schemas import DailyRunSummary, FillRecord, OrderRecord, PortfolioSnapshot, ResearchDecision, RiskDecision
from tradingagents.system.universe import ScreenedAsset


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
) -> Path:
    report_dir = report_root / as_of_date.isoformat()
    report_dir.mkdir(parents=True, exist_ok=True)

    risk_by_symbol = {decision.symbol: decision for decision in risk_decisions}
    payload = {
        "date": as_of_date.isoformat(),
        "summary": None if summary is None else summary.model_dump(mode="json"),
        "shortlist": [asset.model_dump(mode="json") for asset in shortlist],
        "research_decisions": [decision.model_dump(mode="json") for decision in research_decisions],
        "risk_decisions": [decision.model_dump(mode="json") for decision in risk_decisions],
        "orders": [order.model_dump(mode="json") for order in orders],
        "fills": [fill.model_dump(mode="json") for fill in fills],
        "portfolio": portfolio.model_dump(mode="json"),
    }
    (report_dir / "summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines: list[str] = [
        f"# TradingAgents Daily Report ({as_of_date.isoformat()})",
        "",
        "## Portfolio",
        f"- Cash: ${portfolio.cash:,.2f}",
        f"- Equity: ${portfolio.equity:,.2f}",
        f"- Gross Exposure: ${portfolio.gross_exposure:,.2f}",
        f"- Daily Realized PnL: ${portfolio.daily_realized_pnl:,.2f}",
        f"- Daily Unrealized PnL: ${portfolio.daily_unrealized_pnl:,.2f}",
        "",
        "## Shortlist",
    ]
    for asset in shortlist:
        lines.append(
            f"- {asset.symbol}: score={asset.score:.3f}, close=${asset.close:.2f}, "
            f"20d return={asset.return_20d:.2%}, 60d return={asset.return_60d:.2%}, "
            f"20d ADTV=${asset.avg_dollar_volume_20d:,.0f}"
        )

    lines.extend(["", "## Decisions"])
    for decision in research_decisions:
        risk = risk_by_symbol.get(decision.symbol)
        approval = "approved" if risk and risk.approved else f"rejected ({risk.rejection_reason if risk else 'n/a'})"
        lines.append(f"- {decision.symbol}: {decision.action.value.upper()} at {decision.confidence:.2f}, {approval}")
        lines.append(f"  Thesis: {decision.thesis}")

    if orders:
        lines.extend(["", "## Orders"])
        for order in orders:
            lines.append(
                f"- {order.symbol}: {order.side.value.upper()} {order.quantity} status={order.status.value} "
                f"fill_price={order.fill_price if order.fill_price is not None else 'n/a'}"
            )

    if portfolio.positions:
        lines.extend(["", "## Positions"])
        for position in portfolio.positions:
            lines.append(
                f"- {position.symbol}: qty={position.quantity}, avg_cost=${position.avg_cost:.2f}, "
                f"market_price=${position.market_price:.2f}, unrealized=${position.unrealized_pnl:,.2f}"
            )

    report_path = report_dir / "summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path
