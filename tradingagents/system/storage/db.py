from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS regime_snapshots (
        regime_snapshot_id TEXT PRIMARY KEY,
        as_of_date TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        label TEXT NOT NULL,
        risk_budget_multiplier REAL NOT NULL,
        max_gross_exposure_fraction REAL NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS candidate_assessments (
        candidate_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        eligible INTEGER NOT NULL,
        watchlist_only INTEGER NOT NULL,
        ranking_score REAL NOT NULL,
        timestamp TEXT NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_decisions (
        decision_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        action TEXT NOT NULL,
        confidence REAL NOT NULL,
        thesis TEXT NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS research_bundles (
        bundle_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        final_decision_id TEXT NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS risk_decisions (
        risk_decision_id TEXT PRIMARY KEY,
        source_decision_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        approved INTEGER NOT NULL,
        approved_size_fraction REAL NOT NULL,
        rejection_reason TEXT,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_fit_assessments (
        fit_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        fits_portfolio INTEGER NOT NULL,
        recommended_action TEXT NOT NULL,
        target_weight REAL NOT NULL,
        timestamp TEXT NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS execution_plans (
        plan_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        intent_type TEXT NOT NULL,
        side TEXT,
        target_weight REAL NOT NULL,
        timestamp TEXT NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS order_intents (
        intent_id TEXT PRIMARY KEY,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        source_decision_id TEXT NOT NULL,
        source_risk_decision_id TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS orders (
        order_id TEXT PRIMARY KEY,
        intent_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        status TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        fill_timestamp TEXT,
        fill_price REAL,
        commission REAL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fills (
        fill_id TEXT PRIMARY KEY,
        order_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        as_of_date TEXT NOT NULL,
        side TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        price REAL NOT NULL,
        commission REAL NOT NULL,
        slippage_bps REAL NOT NULL,
        timestamp TEXT NOT NULL,
        realized_pnl REAL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS positions (
        symbol TEXT PRIMARY KEY,
        timestamp TEXT NOT NULL,
        quantity INTEGER NOT NULL,
        avg_cost REAL NOT NULL,
        market_price REAL NOT NULL,
        market_value REAL NOT NULL,
        cost_basis REAL NOT NULL,
        unrealized_pnl REAL NOT NULL,
        realized_pnl_day REAL NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        snapshot_id TEXT PRIMARY KEY,
        as_of_date TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        cash REAL NOT NULL,
        equity REAL NOT NULL,
        gross_exposure REAL NOT NULL,
        json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS daily_run_summaries (
        run_id TEXT PRIMARY KEY,
        as_of_date TEXT NOT NULL,
        mode TEXT NOT NULL,
        status TEXT NOT NULL,
        started_at TEXT NOT NULL,
        completed_at TEXT NOT NULL,
        report_path TEXT,
        json TEXT NOT NULL
    )
    """,
]


def connect(database_path: Path) -> sqlite3.Connection:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(database_path: Path) -> None:
    with connect(database_path) as connection:
        for statement in SCHEMA_STATEMENTS:
            connection.execute(statement)
        connection.commit()
