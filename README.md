# TradingAgents Local Research & Paper Trading System (V2)

This repository contains a local-first US equities research + risk + portfolio + paper execution system built on top of the upstream TradingAgents multi-agent graph.

It is designed as a professional internal prototype:
- Local macOS operation (Python and Docker workflow)
- Default model: `gpt-5.4-nano`
- Default data source: `yfinance`
- Phase-1 required secret: `OPENAI_API_KEY` only
- Structured objects, hard deterministic risk controls, auditable SQLite artifacts, and paper execution

## V2 Upgrade Summary

Compared with the prior phase-1 baseline, v2 adds:
- Regime layer (`risk_on` / `balanced` / `risk_off` / `high_volatility`) from liquid proxy instruments
- Multi-stage universe screening with richer ranking and shortlist reasons
- Explicit multi-role research organization outputs:
  - Universe Scout
  - Market Regime Analyst
  - Macro Proxy Analyst
  - Technical Analyst
  - Fundamental Analyst
  - News/Event Analyst
  - Sentiment/Narrative Analyst
  - Bull Researcher
  - Bear Researcher
  - Debate / Adjudication Layer
  - Trader
- Structured research bundle persistence (not just top-level buy/sell/hold)
- Upgraded risk committee logic:
  - regime-aware risk budgeting
  - sector concentration checks
  - correlation-aware checks
  - volatility-aware sizing
  - symbol cooldown logic
- Portfolio fit assessment + execution planning (new/add/trim/exit/hold)
- Richer daily reports (markdown + JSON) with regime, discovery, debate, risk, portfolio fit, and execution sections

## What Is Reused vs Added

Reused from upstream TradingAgents:
- Upstream multi-agent graph and debate/risk discussion backbone (`tradingagents/graph/*`, `tradingagents/agents/*`)
- Upstream LLM client integration
- Existing dataflow tooling used by upstream graph

Added/extended in this local system:
- `tradingagents/system/*` architecture (config, schemas, storage, orchestration, CLI)
- Stable adapter around upstream graph for controlled production-like flows
- Regime, universe ranking, risk committee, portfolio manager, execution planner, reporting, and tests

## Architecture (V2)

`tradingagents/system`:

- `config.py`: typed config loading from defaults + env
- `schemas.py`: strongly typed contracts
  - Existing: `ResearchDecision`, `RiskDecision`, `OrderIntent`, `OrderRecord`, `FillRecord`, `PortfolioSnapshot`, `DailyRunSummary`
  - V2 additions: `RegimeSnapshot`, `CandidateAssessment`, `AnalystMemo`, `BullCaseMemo`, `BearCaseMemo`, `DebateSummary`, `ResearchBundle`, `PortfolioFitAssessment`, `ExecutionPlan`
- `context/regime.py`: market regime model from liquid proxies
- `universe/selector.py`: screening, ranking, shortlist generation with explainability
- `data/yfinance_provider.py`: default market/fundamentals/news/events provider
- `research/adapter.py`: upstream graph adapter and safe fallbacks
- `research/organization.py`: multi-role research orchestration + structured bundle
- `risk/engine.py`: hard deterministic risk committee logic
- `portfolio/service.py`: portfolio fit + target-weight translation + execution plan
- `execution/paper.py`: paper broker with persisted state and fill assumptions
- `execution/futu.py`: production-shaped fail-safe live broker stub (disabled in phase-1)
- `storage/db.py` + `storage/repository.py`: SQLite schema and persistence APIs
- `orchestration/runner.py`: run loop, replay, storage-backed report generation
- `orchestration/reporting.py`: rich markdown/json report generation
- `cli.py`: operator commands

## Data and Execution Constraints

Hard constraints retained:
- No Alpha Vantage requirement
- No paid market data requirement
- No Futu credentials required
- No live broker required
- Paper trading is the required execution mode

## Data Reliability Guardrails (Unattended Safety)

The yfinance provider now uses a reliability-first policy:
- Single-symbol history (`get_history` / `get_latest_bar`) uses `yf.Ticker(symbol).history(...)` as the primary path.
- Batch history keeps `yf.download(...)` for speed, but automatically falls back per symbol to `Ticker.history(...)` when batch output is empty/missing.
- Per-symbol fallback includes retries with small backoff and source-path logging.

Run safety policy:
- Regime proxy completeness is measured each run.
- Shortlist critical history completeness is measured before research.
- In live LLM mode, runs abort before research if data completeness falls below configured thresholds (to avoid burning tokens on impaired market data).

Relevant knobs:
- `TRADINGAGENTS_MIN_REGIME_PROXY_COVERAGE`
- `TRADINGAGENTS_MIN_SHORTLIST_DATA_COVERAGE`
- `TRADINGAGENTS_HISTORY_RETRY_ATTEMPTS`
- `TRADINGAGENTS_FAIL_LIVE_RUN_ON_DATA_IMPAIRMENT`

## Prerequisites

- Python 3.10+ (3.12 recommended)
- Docker Desktop (optional runtime path)
- Internet access (yfinance and OpenAI APIs)

## Setup (Local Python)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

Set:

```bash
OPENAI_API_KEY=your_key_here
```

Bootstrap local state:

```bash
tradingagents setup
```

## Exact Commands

Health check:

```bash
tradingagents health-check
```

Inspect loaded config:

```bash
tradingagents show-config
```

One-shot dry run:

```bash
tradingagents dry-run --as-of 2026-04-13
```

One-shot paper run:

```bash
tradingagents run-once --as-of 2026-04-13 --execute
```

Daily scheduler:

```bash
tradingagents run-daily --run-at 15:45
```

Replay/backfill:

```bash
tradingagents replay --start 2026-04-01 --end 2026-04-10 --no-execute
```

Portfolio and orders:

```bash
tradingagents show-positions
tradingagents show-recent-orders --limit 20
```

Regime and candidate inspection:

```bash
tradingagents show-regime --as-of 2026-04-13
tradingagents show-candidates --as-of 2026-04-13 --limit 20
```

Regenerate report from persisted artifacts:

```bash
tradingagents generate-daily-report --as-of 2026-04-13
```

Legacy upstream CLI (kept):

```bash
tradingagents-legacy
```

## Docker Commands

Build:

```bash
docker compose build
```

Health check:

```bash
docker compose run --rm tradingagents health-check
```

Dry run:

```bash
docker compose run --rm tradingagents dry-run --as-of 2026-04-13
```

Paper run:

```bash
docker compose run --rm tradingagents run-once --as-of 2026-04-13 --execute
```

Persistence:
- Volume `tradingagents_data` mounts to `/home/appuser/.tradingagents`

## Storage Layout

Default root: `~/.tradingagents`

- SQLite: `~/.tradingagents/db/tradingagents.db`
- Cache: `~/.tradingagents/cache`
- Logs: `~/.tradingagents/logs/tradingagents-system.log`
- Reports: `~/.tradingagents/reports/<YYYY-MM-DD>/summary.md` and `summary.json`
- Upstream artifacts: `~/.tradingagents/artifacts`

## V2 Research Pipeline

Per candidate, v2 persists:
- `CandidateAssessment` (screening + ranking evidence)
- multi-role `AnalystMemo` set
- `BullCaseMemo`
- `BearCaseMemo`
- `DebateSummary`
- final `ResearchDecision`
- `ResearchBundle` tying the above together

## V2 Risk and Portfolio Logic

Risk committee is deterministic and non-bypassable:
- Existing hard rules remain (long-only, max position, max gross, daily loss guardrail, liquidity floor, earnings blackout best effort)
- Added regime-aware gross budgeting
- Added sector concentration checks
- Added correlation-aware controls
- Added volatility-aware sizing
- Added symbol cooldown checks

Portfolio manager adds:
- fit assessment (`PortfolioFitAssessment`)
- target weight translation
- action type (`new_entry`, `add`, `trim`, `exit`, `hold`)
- execution shaping (`ExecutionPlan`)

## Reporting

Each daily report includes:
- regime summary
- universe/discovery summary
- shortlist with reasons
- research + bull/bear + debate summary
- risk committee decisions
- portfolio fit and execution planner outputs
- orders/fills
- end-of-day portfolio snapshot
- concentration summary
- warnings/data quality notes

Outputs:
- human markdown: `summary.md`
- machine JSON: `summary.json`

## Quality Commands

Lint:

```bash
ruff check tradingagents/system tests
```

Tests:

```bash
pytest -q
```

## Limitations (No Extra Vendor)

Because phase-1 data is yfinance-only:
- Data quality can be inconsistent (missing fundamentals/news/events, transient API/TLS/network issues)
- Earnings/event coverage is best effort, not institutional-grade event data
- Correlation and regime models are approximations from public proxies
- Paper fills are bar-based simulation, not broker microstructure

The system degrades safely and records warnings when evidence is weak.

## Deferred to Phase-2 (Intentional)

Still intentionally disabled:
- Live Futu OpenD connectivity
- Real broker order routing
- Live account startup checks and credentialed live safeguards

The `BrokerAdapter` abstraction and `FutuBroker` shape are present so phase-2 can swap execution backends without replacing research/risk/storage/reporting layers.
