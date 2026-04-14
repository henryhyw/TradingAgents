from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from tradingagents.system.config import SystemSettings
from tradingagents.system.schemas import DailyRunSummary, PortfolioSnapshot
from tradingagents.system.storage.repository import TradingRepository


@dataclass(frozen=True)
class ArtifactExportResult:
    local_root: Path
    report_markdown: Path
    report_json: Path
    latest_positions: Path
    latest_orders: Path
    latest_run_summary: Path
    latest_regime: Path | None = None
    latest_candidates: Path | None = None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def export_publishable_artifacts(
    *,
    settings: SystemSettings,
    repository: TradingRepository,
    as_of_date: date,
    portfolio_snapshot: PortfolioSnapshot,
    summary: DailyRunSummary | None = None,
) -> ArtifactExportResult:
    """
    Export a publishable local artifact tree with stable paths:
    - reports/YYYY-MM-DD/summary.md
    - reports/YYYY-MM-DD/summary.json
    - snapshots/latest_positions.json
    - snapshots/latest_orders.json
    - snapshots/latest_run_summary.json
    - snapshots/latest_regime.json (optional)
    - snapshots/latest_candidates.json (optional)
    """
    local_root = settings.paths.artifacts_dir / "publish"
    report_prefix = settings.gcp.reports_prefix.strip("/") or "reports"
    snapshots_prefix = settings.gcp.snapshots_prefix.strip("/") or "snapshots"

    report_source_dir = settings.paths.reports_dir / as_of_date.isoformat()
    summary_md_source = report_source_dir / "summary.md"
    summary_json_source = report_source_dir / "summary.json"
    if not summary_md_source.exists() or not summary_json_source.exists():
        raise FileNotFoundError(
            f"Daily report files are missing for {as_of_date.isoformat()} "
            f"at {report_source_dir}"
        )

    report_target_dir = local_root / report_prefix / as_of_date.isoformat()
    report_target_dir.mkdir(parents=True, exist_ok=True)
    report_md_target = report_target_dir / "summary.md"
    report_json_target = report_target_dir / "summary.json"
    shutil.copy2(summary_md_source, report_md_target)
    shutil.copy2(summary_json_source, report_json_target)

    snapshots_dir = local_root / snapshots_prefix
    latest_positions_path = snapshots_dir / "latest_positions.json"
    latest_orders_path = snapshots_dir / "latest_orders.json"
    latest_run_summary_path = snapshots_dir / "latest_run_summary.json"
    latest_regime_path = snapshots_dir / "latest_regime.json"
    latest_candidates_path = snapshots_dir / "latest_candidates.json"

    _write_json(
        latest_positions_path,
        {
            "as_of_date": as_of_date.isoformat(),
            "timestamp": portfolio_snapshot.timestamp.isoformat(),
            "cash": portfolio_snapshot.cash,
            "equity": portfolio_snapshot.equity,
            "gross_exposure": portfolio_snapshot.gross_exposure,
            "positions": [position.model_dump(mode="json") for position in portfolio_snapshot.positions],
        },
    )
    recent_orders = [order.model_dump(mode="json") for order in repository.list_recent_orders(limit=200)]
    _write_json(
        latest_orders_path,
        {
            "as_of_date": as_of_date.isoformat(),
            "orders": recent_orders,
        },
    )
    latest_summary = summary or repository.get_run_summary_for_date(as_of_date) or repository.latest_run_summary()
    _write_json(
        latest_run_summary_path,
        None if latest_summary is None else latest_summary.model_dump(mode="json"),
    )

    regime = repository.get_regime_snapshot_for_date(as_of_date)
    if regime is not None:
        _write_json(latest_regime_path, regime.model_dump(mode="json"))
    else:
        latest_regime_path = None

    candidates = repository.list_candidate_assessments_for_date(as_of_date)
    if candidates:
        _write_json(
            latest_candidates_path,
            {
                "as_of_date": as_of_date.isoformat(),
                "candidates": [item.model_dump(mode="json") for item in candidates],
            },
        )
    else:
        latest_candidates_path = None

    return ArtifactExportResult(
        local_root=local_root,
        report_markdown=report_md_target,
        report_json=report_json_target,
        latest_positions=latest_positions_path,
        latest_orders=latest_orders_path,
        latest_run_summary=latest_run_summary_path,
        latest_regime=latest_regime_path,
        latest_candidates=latest_candidates_path,
    )
