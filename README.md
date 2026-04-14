# TradingAgents Local Research + Paper Trading System (V2, Vertex/GCP)

This repository is a local-first US equities research, risk, portfolio, and paper-execution system built on top of the upstream TradingAgents graph, with:

- Default live LLM provider: **Google Vertex AI**
- Default live model: **`gemini-2.5-flash`**
- Default data source: **`yfinance`**
- Execution mode: **local paper trading**
- Persistence: **SQLite + report artifacts**
- Optional artifact publishing to **Google Cloud Storage (GCS)**
- Compute Engine VM deployment scripts for unattended daily runs

The local trading and paper broker flow is preserved. Cloud deployment is additive.

## What Changed In This Upgrade

- Added Vertex AI Gemini provider path (ADC/service-account based).
- Switched default live model/provider to `vertex` + `gemini-2.5-flash`.
- Kept OpenAI and other provider paths optional.
- Added publishable artifact export tree:
  - `reports/YYYY-MM-DD/summary.md`
  - `reports/YYYY-MM-DD/summary.json`
  - `snapshots/latest_positions.json`
  - `snapshots/latest_orders.json`
  - `snapshots/latest_run_summary.json`
  - optional: `snapshots/latest_regime.json`, `snapshots/latest_candidates.json`
- Added GCS upload utility.
- Added Compute Engine deployment scripts and cron-based daily execution wrapper.
- Preserved yfinance reliability guardrails and fail-fast data impairment checks.

## Architecture (System Layer)

Primary local system modules are under `tradingagents/system`:

- `config.py`: typed settings, env/config merge, provider readiness checks
- `data/yfinance_provider.py`: robust yfinance access with batch+fallback/retries
- `context/regime.py`: structured regime model from liquid proxy instruments
- `universe/selector.py`: screening, ranking, shortlist pipeline
- `research/adapter.py` + `research/organization.py`: upstream adapter + multi-role research artifacts
- `risk/engine.py`: deterministic hard risk committee logic
- `portfolio/service.py`: fit + target-weight + execution plan logic
- `execution/paper.py`: paper broker with persisted positions/orders/fills
- `storage/db.py` + `storage/repository.py`: SQLite schema and repository APIs
- `orchestration/runner.py`: run lifecycle, guardrails, report generation, export/publish hook
- `orchestration/reporting.py`: markdown/json daily report
- `orchestration/artifacts.py`: publishable snapshot export
- `cloud/gcs_publisher.py`: GCS uploader
- `cli.py`: operator CLI

## LLM Provider Migration: OpenAI -> Vertex AI Gemini

Default config now uses:

- `TRADINGAGENTS_LLM_PROVIDER=vertex`
- `TRADINGAGENTS_LLM_MODEL=gemini-2.5-flash`
- `TRADINGAGENTS_LLM_DEEP_MODEL=gemini-2.5-flash`
- `TRADINGAGENTS_LLM_QUICK_MODEL=gemini-2.5-flash`

Vertex auth is via ADC (local `gcloud auth application-default login`) or attached VM service account.

OpenAI remains optional:

- set `TRADINGAGENTS_LLM_PROVIDER=openai`
- set `OPENAI_API_KEY=...`

## Prerequisites

Local/macOS:

- Python 3.10+ (3.12 recommended)
- `gcloud` installed and authenticated
- ADC configured (`gcloud auth application-default login`) for Vertex/GCS if running live LLM/publish locally
- Docker Desktop optional

GCP target assumptions for deployment scripts:

- Project: `ta-henry-2026`
- Region: `us-central1`
- Zone: `us-central1-a`
- VM: `ta-runner-01`
- Bucket: `gs://ta-artifacts-ta-henry-2026`
- APIs enabled: `compute`, `aiplatform`, `storage`, `iam`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Environment Configuration

See `.env.example`. Core defaults are:

```bash
TRADINGAGENTS_LLM_PROVIDER=vertex
TRADINGAGENTS_LLM_MODEL=gemini-2.5-flash
TRADINGAGENTS_VERTEX_PROJECT=ta-henry-2026
TRADINGAGENTS_VERTEX_REGION=us-central1
TRADINGAGENTS_GCS_BUCKET=ta-artifacts-ta-henry-2026
TRADINGAGENTS_PUBLISH_ON_RUN=false
```

No static JSON key file is required. ADC/service-account is used.

## Core CLI Commands

Setup/bootstrap:

```bash
tradingagents setup
```

Health check:

```bash
tradingagents health-check
```

Dry run:

```bash
tradingagents dry-run --as-of 2026-04-13
```

Run once (paper execute):

```bash
tradingagents run-once --as-of 2026-04-13 --execute
```

Run daily local loop (non-VM mode):

```bash
tradingagents run-daily --run-at 15:45
```

Replay:

```bash
tradingagents replay --start 2026-04-01 --end 2026-04-10 --no-execute
```

Inspect state:

```bash
tradingagents show-positions
tradingagents show-recent-orders --limit 20
tradingagents show-regime --as-of 2026-04-13
tradingagents show-candidates --as-of 2026-04-13 --limit 20
tradingagents generate-daily-report --as-of 2026-04-13
```

Export/publish artifacts:

```bash
tradingagents export-artifacts --as-of 2026-04-13
tradingagents publish-artifacts --as-of 2026-04-13
```

## Docker Commands

```bash
docker compose build
docker compose run --rm tradingagents health-check
docker compose run --rm tradingagents dry-run --as-of 2026-04-13
docker compose run --rm tradingagents run-once --as-of 2026-04-13 --execute
```

## Local Storage Paths

Default root: `~/.tradingagents`

- DB: `~/.tradingagents/db/tradingagents.db`
- Logs: `~/.tradingagents/logs/tradingagents-system.log`
- Reports: `~/.tradingagents/reports/YYYY-MM-DD/summary.md|summary.json`
- Export staging: `~/.tradingagents/artifacts/publish/...`

## Google Cloud Deployment (Compute Engine VM)

Deployment scripts are in `scripts/gcp`.

### 1. Create/Update Infra

```bash
./scripts/gcp/create_infra.sh
```

This script:

- ensures service account `ta-runner-sa@ta-henry-2026.iam.gserviceaccount.com`
- grants:
  - `roles/aiplatform.user` (Vertex inference)
  - bucket-level `roles/storage.objectAdmin` on `gs://ta-artifacts-ta-henry-2026` (artifact writes/overwrites)
- creates VM `ta-runner-01` if missing, with:
  - `e2-micro`
  - 20GB `pd-standard`
  - Ubuntu 22.04 LTS
  - attached service account + cloud-platform scope

### 2. Deploy/Update Code To VM

```bash
./scripts/gcp/deploy_or_update.sh
```

### 3. Bootstrap Runtime On VM

```bash
./scripts/gcp/bootstrap_vm.sh
```

This installs Python runtime, creates venv, installs project, and writes `/opt/tradingagents/runtime.env`.

### 4. Install Cron Job

```bash
./scripts/gcp/install_cron.sh
```

Cron behavior:

- `CRON_TZ=America/New_York`
- default schedule: `45 15 * * 1-5`
- executes `scripts/gcp/vm_daily_run.sh`
- wrapper is idempotent for a given market date (skips if already completed)

### 5. Verify VM

```bash
./scripts/gcp/health_check_vm.sh
```

### 6. Trigger Manual Run On VM

```bash
./scripts/gcp/run_remote_daily.sh
```

## VM Runtime Behavior (Daily)

`scripts/gcp/vm_daily_run.sh` executes:

1. activate venv + load runtime env
2. resolve market session date
3. skip if run already completed for that date
4. `tradingagents run-once --as-of <date> --execute`
5. `tradingagents export-artifacts --as-of <date>`
6. `tradingagents publish-artifacts --as-of <date>`

Manual VM scripts:

- `scripts/gcp/vm_run_once.sh [YYYY-MM-DD]`
- `scripts/gcp/vm_publish_once.sh [YYYY-MM-DD]`

## VM File Locations

On VM:

- Repo: `/opt/tradingagents/TradingAgents`
- Runtime env: `/opt/tradingagents/runtime.env`
- Venv: `/opt/tradingagents/venv`
- Local app home: `/opt/tradingagents/.tradingagents`
- Daily log: `/opt/tradingagents/.tradingagents/logs/daily-run.log`
- Cron log: `/opt/tradingagents/.tradingagents/logs/cron.log`
- SQLite: `/opt/tradingagents/.tradingagents/db/tradingagents.db`

## GCS Artifact Layout

After each publish:

- `gs://ta-artifacts-ta-henry-2026/reports/YYYY-MM-DD/summary.md`
- `gs://ta-artifacts-ta-henry-2026/reports/YYYY-MM-DD/summary.json`
- `gs://ta-artifacts-ta-henry-2026/snapshots/latest_positions.json`
- `gs://ta-artifacts-ta-henry-2026/snapshots/latest_orders.json`
- `gs://ta-artifacts-ta-henry-2026/snapshots/latest_run_summary.json`
- optional:
  - `.../snapshots/latest_regime.json`
  - `.../snapshots/latest_candidates.json`

## Remote Inspection Commands

List recent published artifacts:

```bash
gcloud storage ls gs://ta-artifacts-ta-henry-2026/reports/
gcloud storage ls gs://ta-artifacts-ta-henry-2026/snapshots/
```

Inspect latest run summary:

```bash
gcloud storage cat gs://ta-artifacts-ta-henry-2026/snapshots/latest_run_summary.json
```

SSH when needed:

```bash
gcloud compute ssh ta-runner-01 --zone us-central1-a --project ta-henry-2026
```

## Quality / Validation

```bash
ruff check tradingagents/system tests
pytest -q
```

## Data Reliability Guardrails

yfinance safety behavior remains active:

- single-symbol history uses `Ticker.history(...)`
- batch path uses `yf.download(...)` with per-symbol fallback + retries
- run aborts before live LLM research when regime/shortlist data coverage is below thresholds
- symbols with critical missing history are skipped before research

This prevents token burn on materially impaired market data.

## Current Limitations (No Paid Data Vendor)

- yfinance coverage can be noisy/incomplete for some fields
- news/events/fundamentals are best-effort, not institutional-grade
- paper fills are low-frequency bar-based simulation

## Deferred To Future Futu/Live Phase

Intentionally deferred:

- live Futu OpenD connectivity
- real broker order routing
- live account credential and startup safety gates

The broker abstraction remains in place so live execution can be added without replacing research/risk/storage/reporting layers.
