from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from tradingagents.dataflows.stockstats_utils import yf_retry
from tradingagents.system.config import SystemSettings

from .base import EarningsEvent, FundamentalSnapshot, MarketBar, MarketDataProvider, NewsItem


logger = logging.getLogger(__name__)


class YFinanceMarketDataProvider(MarketDataProvider):
    def __init__(self, settings: SystemSettings):
        self.settings = settings
        self.history_cache_dir = settings.paths.cache_dir / "yfinance" / "history"
        self.meta_cache_dir = settings.paths.cache_dir / "yfinance" / "meta"
        self.history_cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _empty_history() -> pd.DataFrame:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume"])

    @staticmethod
    def _history_columns() -> list[str]:
        return ["Date", "Open", "High", "Low", "Close", "Volume"]

    def _history_cache_path(self, symbol: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.history_cache_dir / f"{safe_symbol}.csv"

    def _meta_cache_path(self, symbol: str, suffix: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.meta_cache_dir / f"{safe_symbol}.{suffix}.json"

    def _cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        max_age = timedelta(hours=self.settings.data.cache_ttl_hours)
        return datetime.now() - datetime.fromtimestamp(path.stat().st_mtime) <= max_age

    def _history_window(self, as_of_date: date, lookback_days: int) -> tuple[date, date]:
        start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=max(lookback_days * 3, 365))).date()
        end = (pd.Timestamp(as_of_date) + pd.Timedelta(days=1)).date()
        return start, end

    @staticmethod
    def _coerce_event_timestamp(value: Any) -> pd.Timestamp | None:
        if value is None:
            return None
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes, pd.Timestamp, datetime, date)):
            return YFinanceMarketDataProvider._coerce_event_timestamp(value.tolist())
        if isinstance(value, (list, tuple, set)):
            if not value:
                return None
            return YFinanceMarketDataProvider._coerce_event_timestamp(next(iter(value)))
        try:
            return pd.Timestamp(value).tz_localize(None)
        except Exception:
            return None

    @staticmethod
    def _normalize_history(frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return YFinanceMarketDataProvider._empty_history()
        normalized = frame.reset_index().copy()
        if isinstance(normalized.columns, pd.MultiIndex):
            normalized.columns = [
                next((str(part) for part in column if part not in ("", None)), "")
                for column in normalized.columns
            ]
        if "Date" not in normalized.columns and "Datetime" in normalized.columns:
            normalized = normalized.rename(columns={"Datetime": "Date"})
        if "Date" not in normalized.columns and "index" in normalized.columns:
            normalized = normalized.rename(columns={"index": "Date"})
        if "Date" not in normalized.columns:
            return YFinanceMarketDataProvider._empty_history()
        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.tz_localize(None)
        normalized = normalized.dropna(subset=["Date"])
        for column in ("Open", "High", "Low", "Close", "Volume"):
            if column in normalized.columns:
                normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        normalized = normalized.dropna(subset=["Close", "Volume"])
        return normalized.sort_values("Date").reset_index(drop=True)

    def _filter_history(self, frame: pd.DataFrame, as_of_date: date, lookback_days: int) -> pd.DataFrame:
        if frame.empty:
            return frame
        cutoff = pd.Timestamp(as_of_date)
        filtered = frame[frame["Date"] <= cutoff].copy()
        if filtered.empty:
            return filtered
        return filtered.tail(lookback_days).reset_index(drop=True)

    def _load_cached_history(self, symbol: str) -> pd.DataFrame:
        cache_path = self._history_cache_path(symbol)
        if not cache_path.exists():
            return self._empty_history()
        try:
            frame = pd.read_csv(cache_path, parse_dates=["Date"])
            frame["Date"] = frame["Date"].dt.tz_localize(None)
            return self._normalize_history(frame)
        except Exception as exc:
            logger.warning("Unable to read cached history for %s: %s", symbol, exc)
            return self._empty_history()

    def _write_history_cache(self, symbol: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        try:
            frame.to_csv(self._history_cache_path(symbol), index=False)
        except Exception as exc:
            logger.warning("Unable to write history cache for %s: %s", symbol, exc)

    def _fetch_symbol_history_ticker(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        retries = self.settings.data.history_retry_attempts
        backoff = self.settings.data.history_retry_backoff_seconds
        for attempt in range(1, retries + 2):
            try:
                ticker = yf.Ticker(symbol)
                raw = yf_retry(
                    lambda ticker=ticker, start=start, end=end: ticker.history(
                        start=start.isoformat(),
                        end=end.isoformat(),
                        interval="1d",
                        auto_adjust=True,
                        actions=False,
                    )
                )
                normalized = self._normalize_history(raw)
                if normalized.empty:
                    raw_period = yf_retry(
                        lambda ticker=ticker: ticker.history(
                            period="3y",
                            interval="1d",
                            auto_adjust=True,
                            actions=False,
                        )
                    )
                    normalized = self._normalize_history(raw_period)
                if not normalized.empty:
                    logger.info(
                        "History source for %s: ticker_history (attempt %s/%s)",
                        symbol,
                        attempt,
                        retries + 1,
                    )
                    return normalized
                logger.warning(
                    "Empty ticker.history for %s on attempt %s/%s",
                    symbol,
                    attempt,
                    retries + 1,
                )
            except Exception as exc:
                logger.warning(
                    "ticker.history failed for %s on attempt %s/%s: %s",
                    symbol,
                    attempt,
                    retries + 1,
                    exc,
                )
            if attempt <= retries:
                sleep_for = backoff * attempt
                logger.info("Retrying %s ticker.history in %.2fs", symbol, sleep_for)
                time.sleep(sleep_for)
        return self._empty_history()

    def get_history(self, symbol: str, as_of_date: date, lookback_days: int) -> pd.DataFrame:
        cache_path = self._history_cache_path(symbol)
        frame = self._empty_history()
        used_source = "none"
        try:
            if self._cache_fresh(cache_path):
                frame = self._load_cached_history(symbol)
                used_source = "cache"
                logger.info("History source for %s: cache", symbol)
            if frame.empty:
                start, end = self._history_window(as_of_date, lookback_days)
                frame = self._fetch_symbol_history_ticker(symbol, start=start, end=end)
                used_source = "ticker_history"
                if not frame.empty:
                    self._write_history_cache(symbol, frame)
            if frame.empty:
                stale = self._load_cached_history(symbol)
                if not stale.empty:
                    frame = stale
                    used_source = "stale_cache_fallback"
                    logger.warning("History source for %s: stale_cache_fallback", symbol)
        except Exception as exc:
            logger.warning("Unable to load history for %s: %s", symbol, exc)
            return self._empty_history()
        if frame.empty:
            logger.warning("No usable history for %s after source=%s", symbol, used_source)
            return frame
        return self._filter_history(frame, as_of_date, lookback_days)

    def batch_get_history(self, symbols: list[str], as_of_date: date, lookback_days: int) -> dict[str, pd.DataFrame]:
        if not symbols:
            return {}
        start_date, end_date = self._history_window(as_of_date, lookback_days)
        histories: dict[str, pd.DataFrame] = {}
        symbols = [symbol.upper() for symbol in symbols]
        chunk_size = 25
        for chunk_start in range(0, len(symbols), chunk_size):
            chunk = list(dict.fromkeys(symbols[chunk_start : chunk_start + chunk_size]))
            missing_in_chunk: set[str] = set(chunk)
            try:
                raw = yf_retry(
                    lambda chunk_symbols=chunk: yf.download(
                        " ".join(chunk_symbols),
                        start=start_date.isoformat(),
                        end=end_date.isoformat(),
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        group_by="ticker",
                        threads=True,
                        timeout=20,
                    )
                )
            except Exception as exc:
                logger.warning("Batch yfinance history failed for %s: %s", ",".join(chunk), exc)
                raw = pd.DataFrame()

            if not raw.empty:
                multi_symbol = isinstance(raw.columns, pd.MultiIndex)
                for symbol in chunk:
                    if multi_symbol:
                        if symbol not in raw.columns.get_level_values(0):
                            continue
                        symbol_frame = raw[symbol]
                    else:
                        symbol_frame = raw
                    normalized = self._normalize_history(symbol_frame)
                    filtered = self._filter_history(normalized, as_of_date, lookback_days)
                    if filtered.empty:
                        continue
                    histories[symbol] = filtered
                    missing_in_chunk.discard(symbol)
                    self._write_history_cache(symbol, normalized)
                    logger.info("History source for %s: batch_download", symbol)

            if missing_in_chunk:
                logger.warning(
                    "Batch missing %s/%s symbols (%s). Falling back to ticker.history.",
                    len(missing_in_chunk),
                    len(chunk),
                    ",".join(sorted(missing_in_chunk)),
                )
                for symbol in sorted(missing_in_chunk):
                    normalized = self._fetch_symbol_history_ticker(symbol, start=start_date, end=end_date)
                    filtered = self._filter_history(normalized, as_of_date, lookback_days)
                    if filtered.empty:
                        logger.warning("No usable history for %s after batch+fallback paths", symbol)
                        continue
                    histories[symbol] = filtered
                    self._write_history_cache(symbol, normalized)
                    logger.info("History source for %s: ticker_fallback_after_batch", symbol)
        return histories

    def get_latest_bar(self, symbol: str, as_of_date: date) -> MarketBar | None:
        history = self.get_history(symbol, as_of_date, max(self.settings.data.shortlist_min_history_days, 10))
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

    def _load_meta_cache(self, path: Path) -> dict[str, Any] | None:
        if not self._cache_fresh(path):
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _write_meta_cache(self, path: Path, payload: dict[str, Any]) -> None:
        path.write_text(json.dumps(payload, default=str), encoding="utf-8")

    def get_fundamentals(self, symbol: str) -> FundamentalSnapshot:
        cache_path = self._meta_cache_path(symbol, "info")
        cached = self._load_meta_cache(cache_path)
        if cached is None:
            ticker = yf.Ticker(symbol)
            try:
                info = yf_retry(lambda: ticker.info) or {}
            except Exception as exc:
                logger.warning("Unable to load fundamentals for %s: %s", symbol, exc)
                info = {}
            payload = {
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "beta": info.get("beta"),
                "trailing_pe": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "price_to_book": info.get("priceToBook"),
                "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                "extra": {
                    "long_name": info.get("longName"),
                    "exchange": info.get("exchange"),
                    "currency": info.get("currency"),
                },
            }
            self._write_meta_cache(cache_path, payload)
        else:
            payload = cached
        return FundamentalSnapshot(symbol=symbol, **payload)

    def get_news(self, symbol: str, as_of_date: date, limit: int) -> list[NewsItem]:
        ticker = yf.Ticker(symbol)
        try:
            raw_items = yf_retry(lambda: ticker.get_news(count=limit)) or []
        except Exception as exc:
            logger.warning("Unable to load news for %s: %s", symbol, exc)
            raw_items = []
        items: list[NewsItem] = []
        for item in raw_items:
            content = item.get("content", item)
            provider = content.get("provider", {})
            published_at = None
            if content.get("pubDate"):
                try:
                    published_at = datetime.fromisoformat(content["pubDate"].replace("Z", "+00:00"))
                except ValueError:
                    published_at = None
            if published_at and published_at.date() > as_of_date:
                continue
            link_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
            items.append(
                NewsItem(
                    symbol=symbol,
                    title=content.get("title", "Untitled"),
                    publisher=provider.get("displayName", content.get("publisher", "Unknown")),
                    summary=content.get("summary", ""),
                    link=link_obj.get("url"),
                    published_at=published_at,
                )
            )
        return items[:limit]

    def get_earnings_event(self, symbol: str, as_of_date: date) -> EarningsEvent:
        cache_path = self._meta_cache_path(symbol, "earnings")
        cached = self._load_meta_cache(cache_path)
        if cached is not None:
            earnings_date = cached.get("earnings_date")
            return EarningsEvent(
                symbol=symbol,
                earnings_date=date.fromisoformat(earnings_date) if earnings_date else None,
                source=cached.get("source", "cache"),
                reliable=bool(cached.get("reliable", False)),
            )

        ticker = yf.Ticker(symbol)
        event = EarningsEvent(symbol=symbol)
        try:
            calendar = yf_retry(lambda: ticker.calendar)
        except Exception as exc:  # pragma: no cover - yfinance schema drift
            logger.warning("Unable to load earnings calendar for %s: %s", symbol, exc)
            calendar = None

        earnings_value = None
        if isinstance(calendar, pd.DataFrame) and not calendar.empty:
            if "Earnings Date" in calendar.index:
                earnings_value = calendar.loc["Earnings Date"].iloc[0]
            elif "Earnings Date" in calendar.columns:
                earnings_value = calendar["Earnings Date"].iloc[0]
        elif isinstance(calendar, dict):
            earnings_value = calendar.get("Earnings Date")

        if earnings_value is None:
            try:
                earnings_dates = yf_retry(lambda: ticker.get_earnings_dates(limit=8))
            except Exception as exc:  # pragma: no cover - yfinance schema drift
                logger.warning("Unable to load earnings dates for %s: %s", symbol, exc)
                earnings_dates = None
            if earnings_dates is not None and not earnings_dates.empty:
                for idx in earnings_dates.index:
                    idx_ts = pd.Timestamp(idx).tz_localize(None)
                    if idx_ts.date() >= as_of_date:
                        earnings_value = idx_ts
                        event = EarningsEvent(symbol=symbol, earnings_date=idx_ts.date(), source="earnings_dates", reliable=True)
                        break

        if earnings_value is not None and event.earnings_date is None:
            parsed = self._coerce_event_timestamp(earnings_value)
            if parsed is not None:
                event = EarningsEvent(symbol=symbol, earnings_date=parsed.date(), source="calendar", reliable=True)

        self._write_meta_cache(
            cache_path,
            {
                "earnings_date": event.earnings_date.isoformat() if event.earnings_date else None,
                "source": event.source,
                "reliable": event.reliable,
            },
        )
        return event
