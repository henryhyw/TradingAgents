from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, PositiveInt

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULTS_PATH = REPO_ROOT / "tradingagents" / "system" / "assets" / "defaults.toml"


class BaseSettingsModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LLMSettings(BaseSettingsModel):
    provider: str = "openai"
    model: str = "gpt-5.4-nano"
    deep_model: str = "gpt-5.4-nano"
    quick_model: str = "gpt-5.4-nano"
    reasoning_effort: str | None = "low"
    temperature: float = 0.0
    backend_url: str = "https://api.openai.com/v1"


class RunSettings(BaseSettingsModel):
    default_shortlist_size: PositiveInt = 8
    market_timezone: str = "America/New_York"
    report_timezone: str = "America/New_York"
    research_analysts: list[str] = Field(default_factory=lambda: ["market", "news", "fundamentals"])
    loop_sleep_seconds: PositiveInt = 60


class DataSettings(BaseSettingsModel):
    provider: str = "yfinance"
    history_lookback_days: PositiveInt = 260
    screen_lookback_days: PositiveInt = 90
    shortlist_min_history_days: PositiveInt = 60
    min_price: float = 10.0
    min_avg_dollar_volume: float = 20_000_000.0
    max_news_items: PositiveInt = 8
    cache_ttl_hours: PositiveInt = 12
    earnings_blackout_days: PositiveInt = 3


class RiskSettings(BaseSettingsModel):
    long_only: bool = True
    max_position_size_fraction: float = 0.05
    max_gross_exposure_fraction: float = 0.30
    daily_loss_limit_fraction: float = 0.02
    max_new_opening_trades_per_symbol_per_day: PositiveInt = 1
    stop_opening_after_losing_exits: PositiveInt = 3
    minimum_cash_buffer_fraction: float = 0.05


class PaperSettings(BaseSettingsModel):
    starting_cash: float = 100_000.0
    fill_model: str = "same_bar_close"
    slippage_bps: float = 10.0
    commission_per_order: float = 1.0
    allow_fractional_shares: bool = False


class StorageSettings(BaseSettingsModel):
    home_subdir: str = ".tradingagents"
    database_name: str = "tradingagents.db"


class PathSettings(BaseSettingsModel):
    home: Path
    database_path: Path
    cache_dir: Path
    logs_dir: Path
    reports_dir: Path
    artifacts_dir: Path


class SystemSettings(BaseSettingsModel):
    repo_root: Path = REPO_ROOT
    llm: LLMSettings = Field(default_factory=LLMSettings)
    run: RunSettings = Field(default_factory=RunSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    paper: PaperSettings = Field(default_factory=PaperSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    paths: PathSettings

    def ensure_directories(self) -> None:
        for path in (
            self.paths.home,
            self.paths.cache_dir,
            self.paths.logs_dir,
            self.paths.reports_dir,
            self.paths.artifacts_dir,
            self.paths.database_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    def llm_ready(self) -> bool:
        if self.llm.provider.lower() != "openai":
            return False
        return bool(self.openai_api_key())

    def as_tradingagents_config(self) -> dict[str, Any]:
        return {
            "project_dir": str(self.repo_root / "tradingagents"),
            "results_dir": str(self.paths.artifacts_dir),
            "data_cache_dir": str(self.paths.cache_dir / "upstream"),
            "llm_provider": self.llm.provider,
            "deep_think_llm": self.llm.deep_model,
            "quick_think_llm": self.llm.quick_model,
            "backend_url": self.llm.backend_url,
            "openai_reasoning_effort": self.llm.reasoning_effort,
            "output_language": "English",
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "data_vendors": {
                "core_stock_apis": "yfinance",
                "technical_indicators": "yfinance",
                "fundamental_data": "yfinance",
                "news_data": "yfinance",
            },
            "tool_vendors": {},
        }


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_defaults(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULTS_PATH
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _env_float(name: str) -> float | None:
    value = os.getenv(name)
    return None if value is None else float(value)


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    return None if value is None else int(value)


def _env_bool(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _paths_from_storage(repo_root: Path, storage: StorageSettings) -> PathSettings:
    home = Path(os.getenv("TRADINGAGENTS_HOME", Path.home() / storage.home_subdir)).expanduser()
    return PathSettings(
        home=home,
        database_path=home / "db" / storage.database_name,
        cache_dir=home / "cache",
        logs_dir=home / "logs",
        reports_dir=home / "reports",
        artifacts_dir=home / "artifacts",
    )


def load_settings(config_path: str | Path | None = None) -> SystemSettings:
    load_dotenv()
    defaults = _load_defaults(Path(config_path) if config_path else None)
    env_overrides: dict[str, Any] = {
        "llm": {
            "provider": os.getenv("TRADINGAGENTS_LLM_PROVIDER"),
            "model": os.getenv("TRADINGAGENTS_LLM_MODEL"),
            "deep_model": os.getenv("TRADINGAGENTS_LLM_DEEP_MODEL"),
            "quick_model": os.getenv("TRADINGAGENTS_LLM_QUICK_MODEL"),
            "reasoning_effort": os.getenv("TRADINGAGENTS_OPENAI_REASONING_EFFORT"),
            "backend_url": os.getenv("TRADINGAGENTS_BACKEND_URL"),
        },
        "run": {
            "default_shortlist_size": _env_int("TRADINGAGENTS_SHORTLIST_SIZE"),
            "market_timezone": os.getenv("TRADINGAGENTS_MARKET_TIMEZONE"),
            "report_timezone": os.getenv("TRADINGAGENTS_REPORT_TIMEZONE"),
            "loop_sleep_seconds": _env_int("TRADINGAGENTS_LOOP_SLEEP_SECONDS"),
        },
        "data": {
            "min_price": _env_float("TRADINGAGENTS_MIN_PRICE"),
            "min_avg_dollar_volume": _env_float("TRADINGAGENTS_MIN_ADTV"),
            "history_lookback_days": _env_int("TRADINGAGENTS_HISTORY_LOOKBACK_DAYS"),
            "screen_lookback_days": _env_int("TRADINGAGENTS_SCREEN_LOOKBACK_DAYS"),
            "max_news_items": _env_int("TRADINGAGENTS_MAX_NEWS_ITEMS"),
            "earnings_blackout_days": _env_int("TRADINGAGENTS_EARNINGS_BLACKOUT_DAYS"),
        },
        "risk": {
            "max_position_size_fraction": _env_float("TRADINGAGENTS_MAX_POSITION_SIZE"),
            "max_gross_exposure_fraction": _env_float("TRADINGAGENTS_MAX_GROSS_EXPOSURE"),
            "daily_loss_limit_fraction": _env_float("TRADINGAGENTS_DAILY_LOSS_LIMIT"),
            "stop_opening_after_losing_exits": _env_int("TRADINGAGENTS_MAX_LOSING_EXITS_PER_DAY"),
            "minimum_cash_buffer_fraction": _env_float("TRADINGAGENTS_MIN_CASH_BUFFER"),
        },
        "paper": {
            "starting_cash": _env_float("TRADINGAGENTS_STARTING_CASH"),
            "fill_model": os.getenv("TRADINGAGENTS_FILL_MODEL"),
            "slippage_bps": _env_float("TRADINGAGENTS_SLIPPAGE_BPS"),
            "commission_per_order": _env_float("TRADINGAGENTS_COMMISSION_PER_ORDER"),
            "allow_fractional_shares": _env_bool("TRADINGAGENTS_ALLOW_FRACTIONAL_SHARES"),
        },
    }

    def _drop_none(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {key: _drop_none(value) for key, value in obj.items() if value is not None}
        return obj

    merged = _deep_merge(defaults, _drop_none(env_overrides))
    storage = StorageSettings.model_validate(merged.get("storage", {}))
    settings = SystemSettings.model_validate(
        {
            "repo_root": REPO_ROOT,
            "llm": merged.get("llm", {}),
            "run": merged.get("run", {}),
            "data": merged.get("data", {}),
            "risk": merged.get("risk", {}),
            "paper": merged.get("paper", {}),
            "storage": storage.model_dump(),
            "paths": _paths_from_storage(REPO_ROOT, storage).model_dump(),
        }
    )
    settings.ensure_directories()
    return settings
