from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

import pandas as pd

from pydantic import Field

from tradingagents.system.schemas import StrictModel


class NewsItem(StrictModel):
    symbol: str
    title: str
    publisher: str
    summary: str = ""
    link: str | None = None
    published_at: datetime | None = None


class EarningsEvent(StrictModel):
    symbol: str
    earnings_date: date | None = None
    source: str = "unknown"
    reliable: bool = False


class FundamentalSnapshot(StrictModel):
    symbol: str
    sector: str | None = None
    industry: str | None = None
    market_cap: float | None = None
    beta: float | None = None
    trailing_pe: float | None = None
    forward_pe: float | None = None
    price_to_book: float | None = None
    fifty_two_week_high: float | None = None
    fifty_two_week_low: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MarketBar(StrictModel):
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataProvider(ABC):
    @abstractmethod
    def get_history(self, symbol: str, as_of_date: date, lookback_days: int) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def get_latest_bar(self, symbol: str, as_of_date: date) -> MarketBar | None:
        raise NotImplementedError

    @abstractmethod
    def batch_get_history(self, symbols: list[str], as_of_date: date, lookback_days: int) -> dict[str, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def get_news(self, symbol: str, as_of_date: date, limit: int) -> list[NewsItem]:
        raise NotImplementedError

    @abstractmethod
    def get_fundamentals(self, symbol: str) -> FundamentalSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_earnings_event(self, symbol: str, as_of_date: date) -> EarningsEvent:
        raise NotImplementedError
