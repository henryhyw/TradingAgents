# TradingAgents Local Paper Trading System (US Equities, Phase 1)

This repository now contains a complete local-first paper-trading system built around the existing TradingAgents multi-agent research graph.

Phase 1 goals delivered:
- Local macOS operation (Python and Docker)
- Default model: `gpt-5.4-nano`
- Default data source: `yfinance`
- Only required secret: `OPENAI_API_KEY`
- Structured decisions, hard risk overrides, paper execution, SQLite persistence, logs, and reports
- Clean broker abstraction for future Futu/OpenD migration

## What This System Does

On each run, the system:
1. Loads a curated US universe (liquid stocks + major ETFs)
2. Applies hard market-quality filters and ranking
3. Runs multi-agent research on shortlisted symbols
4. Parses research into validated `ResearchDecision` objects
5. Applies hard risk controls (`RiskDecision`) that override AI output
6. Builds validated order intents
7. Simulates paper execution (optional)
8. Persists decisions, orders, fills, positions, and run summaries in SQLite
9. Emits report artifacts and structured logs

## Architecture

The new production-shaped implementation lives in `tradingagents/system`:

- `config.py`: strongly typed settings + env/config loading
- `schemas.py`: typed contracts (`ResearchDecision`, `RiskDecision`, `OrderIntent`, `OrderRecord`, `FillRecord`, `PositionSnapshot`, `DailyRunSummary`, etc.)
- `universe/`: curated universe + screening + shortlist logic
- `data/`: market data provider interface + yfinance implementation
- `research/`: adapter around upstream TradingAgents graph + structured parser + deterministic fallback adapter
- `risk/`: hard risk engine
- `portfolio/`: intent sizing and portfolio translation logic
- `execution/`: broker interface + fully working `PaperBroker` + production-shaped `FutuBroker` fail-safe adapter
- `storage/`: SQLite schema + repository
- `orchestration/`: run loop, scheduler, report generation
- `cli.py`: operator CLI

## Upstream Reuse vs New System Code

Reused from upstream TradingAgents:
- Multi-agent research graph and agent role structure (`tradingagents/graph/*`, `tradingagents/agents/*`)
- LLM client stack (`tradingagents/llm_clients/*`)
- Existing yfinance/market tooling where relevant

Wrapped/refactored/added for this system:
- Stable research adapter boundary (`tradingagents/system/research/adapter.py`)
- Full structured contracts and persistence
- Hard risk engine and sizing constraints
- Real paper broker simulation and state handling
- Universe construction/screening and shortlist gating
- New operator CLI and daily scheduler
- Reporting, logging, tests, and runbook-quality docs

## Repository Layout (Key Paths)

```text
tradingagents/
  system/
    assets/defaults.toml
    cli.py
    config.py
    schemas.py
    data/
    execution/
    monitoring/
    orchestration/
    portfolio/
    research/
    risk/
    storage/
    universe/us_equities_phase1.csv
tests/
  test_system_*.py
  system_helpers.py
```

## Prerequisites (macOS)

- Python 3.10+ (3.12 recommended)
- Docker Desktop (for Docker mode)
- Internet access for yfinance + OpenAI API

## Setup (Local Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`:

```bash
OPENAI_API_KEY=your_key_here
```

Initialize local state:

```bash
tradingagents setup
```

## First Run Commands

First dry-run (research + risk, no execution):

```bash
tradingagents dry-run --as-of 2026-04-13
```

First paper-trading run (simulated execution enabled):

```bash
tradingagents run-once --as-of 2026-04-13 --execute
```

Health check:

```bash
tradingagents health-check
```

## Operator CLI

Available commands:

- `tradingagents setup` (alias of `bootstrap`)
- `tradingagents bootstrap`
- `tradingagents health-check`
- `tradingagents run-once`
- `tradingagents dry-run`
- `tradingagents run-daily --run-at 15:45` (America/New_York clock)
- `tradingagents replay --start YYYY-MM-DD --end YYYY-MM-DD`
- `tradingagents show-positions`
- `tradingagents show-recent-orders`
- `tradingagents generate-daily-report --as-of YYYY-MM-DD`

Legacy upstream interactive CLI remains available:

```bash
tradingagents-legacy
```

## Data, DB, Logs, Reports

Default storage root:

- `~/.tradingagents/`

Important files/directories:

- SQLite DB: `~/.tradingagents/db/tradingagents.db`
- Market cache: `~/.tradingagents/cache/`
- Structured logs: `~/.tradingagents/logs/tradingagents-system.log`
- Reports: `~/.tradingagents/reports/<YYYY-MM-DD>/summary.md` and `summary.json`
- Upstream graph artifacts: `~/.tradingagents/artifacts/`

## Docker Usage

Build:

```bash
docker compose build
```

Health check in container:

```bash
docker compose run --rm tradingagents health-check
```

Dry-run:

```bash
docker compose run --rm tradingagents dry-run --as-of 2026-04-13
```

Paper run:

```bash
docker compose run --rm tradingagents run-once --as-of 2026-04-13 --execute
```

Docker persistence:
- Compose mounts volume `tradingagents_data` to `/home/appuser/.tradingagents` in the container.

## Configuration

Default settings live in:

- `tradingagents/system/assets/defaults.toml`

Runtime overrides are supported via environment variables, including:

- `TRADINGAGENTS_LLM_MODEL`
- `TRADINGAGENTS_LLM_DEEP_MODEL`
- `TRADINGAGENTS_LLM_QUICK_MODEL`
- `TRADINGAGENTS_SHORTLIST_SIZE`
- `TRADINGAGENTS_STARTING_CASH`
- `TRADINGAGENTS_MAX_POSITION_SIZE`
- `TRADINGAGENTS_MAX_GROSS_EXPOSURE`
- `TRADINGAGENTS_DAILY_LOSS_LIMIT`

## Risk Rules (Phase 1 Defaults)

- Long-only
- Max position size: 5% of equity
- Max gross exposure: 30% of equity
- Min price: $10
- Min liquidity: 20-day average dollar volume threshold
- Max one new opening trade per symbol per day
- Stop opening new positions after 3 losing exits in the same day
- Daily loss guardrail: 2%
- Earnings blackout attempt (best-effort from yfinance events)
- AI decisions cannot bypass these rules

## Execution Model (Paper Broker)

- Fill model: `same_bar_close`
- Slippage: configurable (`slippage_bps`)
- Commission: configurable (`commission_per_order`)
- Cash/positions/orders/fills persisted in SQLite
- Deterministic bar-based simulation assumptions (documented in config)

## Testing and Quality

Run tests:

```bash
pytest -q
```

Run lint:

```bash
ruff check tradingagents/system tests
```

Test coverage includes:
- Schema validation
- Risk engine behavior
- Paper broker behavior
- Research parser and deterministic adapter behavior
- End-to-end dry-run smoke orchestration

## Current Limitations

- yfinance can intermittently fail or throttle; code now degrades safely, but data completeness can vary.
- Earnings blackout is best-effort with no paid event feed.
- Paper fills are simulation, not broker market microstructure.
- Strategy is phase-1 long-only swing/daily style (no options/HFT/live broker routing).

## Futu/OpenD Phase-2 Path

`FutuBroker` interface and config shape are already present in `tradingagents/system/execution/futu.py`.

What is intentionally disabled in phase 1:
- Live Futu connectivity
- Credentialed live order routing
- Live account startup checks

When phase-2 credentials/connectivity are available, swap `PaperBroker` with `FutuBroker` behind the existing broker interface and keep the same orchestration/risk/storage layers.
