from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

from tradingagents.system.data import EarningsEvent, FundamentalSnapshot, MarketBar, MarketDataProvider, NewsItem


def make_price_history(
    as_of_date: date,
    periods: int = 120,
    start_price: float = 100.0,
    step: float = 0.5,
    volume: float = 2_000_000.0,
) -> pd.DataFrame:
    dates = pd.bdate_range(end=as_of_date, periods=periods)
    records = []
    close = start_price
    for ts in dates:
        open_price = close - 0.2
        high = close + 0.4
        low = close - 0.6
        records.append(
            {
                "Date": ts.tz_localize(None),
                "Open": open_price,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
        close += step
    return pd.DataFrame(records)


class FakeMarketDataProvider(MarketDataProvider):
    def __init__(
        self,
        histories: dict[str, pd.DataFrame],
        earnings_dates: dict[str, date | None] | None = None,
    ):
        self.histories = histories
        self.earnings_dates = earnings_dates or {}

    def get_history(self, symbol: str, as_of_date: date, lookback_days: int) -> pd.DataFrame:
        history = self.histories.get(symbol)
        if history is None:
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        filtered = history[history["Date"] <= pd.Timestamp(as_of_date)].copy()
        return filtered.tail(lookback_days).reset_index(drop=True)

    def get_latest_bar(self, symbol: str, as_of_date: date) -> MarketBar | None:
        history = self.get_history(symbol, as_of_date, 5)
        if history.empty:
            return None
        row = history.iloc[-1]
        return MarketBar(
            symbol=symbol,
            date=row["Date"].date(),
            open=float(row["Open"]),
            high=float(row["High"]),
            low=float(row["Low"]),
            close=float(row["Close"]),
            volume=float(row["Volume"]),
        )

    def batch_get_history(self, symbols: list[str], as_of_date: date, lookback_days: int) -> dict[str, pd.DataFrame]:
        return {symbol: self.get_history(symbol, as_of_date, lookback_days) for symbol in symbols}

    def get_news(self, symbol: str, as_of_date: date, limit: int) -> list[NewsItem]:
        return []

    def get_fundamentals(self, symbol: str) -> FundamentalSnapshot:
        return FundamentalSnapshot(symbol=symbol)

    def get_earnings_event(self, symbol: str, as_of_date: date) -> EarningsEvent:
        return EarningsEvent(
            symbol=symbol,
            earnings_date=self.earnings_dates.get(symbol),
            source="fake",
            reliable=self.earnings_dates.get(symbol) is not None,
        )


def symbols_with_same_history(symbols: Iterable[str], history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {symbol: history.copy() for symbol in symbols}
