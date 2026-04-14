from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import Field

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import MarketDataProvider
from tradingagents.system.schemas import StrictModel


class ScreenedAsset(StrictModel):
    symbol: str
    name: str
    asset_type: str
    sector: str
    close: float
    avg_dollar_volume_20d: float
    return_20d: float
    return_60d: float
    volatility_20d: float
    score: float
    rejection_reasons: list[str] = Field(default_factory=list)


class UniverseSelector:
    def __init__(self, settings: SystemSettings, provider: MarketDataProvider, universe_path: Path | None = None):
        self.settings = settings
        self.provider = provider
        self.universe_path = universe_path or settings.repo_root / "tradingagents" / "system" / "universe" / "us_equities_phase1.csv"

    def load_universe(self) -> list[dict[str, str]]:
        with self.universe_path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _rank_series(values: pd.Series) -> pd.Series:
        return values.rank(pct=True, method="average").fillna(0.0)

    def screen_universe(self, as_of_date: date) -> list[ScreenedAsset]:
        universe = self.load_universe()
        symbols = [row["symbol"] for row in universe]
        histories = self.provider.batch_get_history(symbols, as_of_date, self.settings.data.history_lookback_days)
        assets: list[ScreenedAsset] = []
        rows_for_ranking: list[dict[str, float | str]] = []

        for row in universe:
            symbol = row["symbol"]
            history = histories.get(symbol)
            rejection_reasons: list[str] = []
            if history is None or history.empty:
                rejection_reasons.append("missing_history")
            elif len(history) < self.settings.data.shortlist_min_history_days:
                rejection_reasons.append("insufficient_history")

            if rejection_reasons:
                assets.append(
                    ScreenedAsset(
                        symbol=symbol,
                        name=row["name"],
                        asset_type=row["asset_type"],
                        sector=row["sector"],
                        close=0.0,
                        avg_dollar_volume_20d=0.0,
                        return_20d=0.0,
                        return_60d=0.0,
                        volatility_20d=0.0,
                        score=0.0,
                        rejection_reasons=rejection_reasons,
                    )
                )
                continue

            close = float(history["Close"].iloc[-1])
            avg_dollar_volume_20d = float((history["Close"].tail(20) * history["Volume"].tail(20)).mean())
            returns = history["Close"].pct_change().dropna()
            return_20d = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1) if len(history) > 21 else 0.0
            return_60d = float(history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1) if len(history) > 61 else return_20d
            volatility_20d = float(returns.tail(20).std() * (252 ** 0.5)) if len(returns) >= 20 else 0.0

            if close < self.settings.data.min_price:
                rejection_reasons.append("price_below_minimum")
            if avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
                rejection_reasons.append("liquidity_below_minimum")

            assets.append(
                ScreenedAsset(
                    symbol=symbol,
                    name=row["name"],
                    asset_type=row["asset_type"],
                    sector=row["sector"],
                    close=close,
                    avg_dollar_volume_20d=avg_dollar_volume_20d,
                    return_20d=return_20d,
                    return_60d=return_60d,
                    volatility_20d=volatility_20d,
                    score=0.0,
                    rejection_reasons=rejection_reasons,
                )
            )
            if not rejection_reasons:
                rows_for_ranking.append(
                    {
                        "symbol": symbol,
                        "avg_dollar_volume_20d": avg_dollar_volume_20d,
                        "return_20d": return_20d,
                        "return_60d": return_60d,
                        "volatility_20d": volatility_20d,
                    }
                )

        if rows_for_ranking:
            rank_frame = pd.DataFrame(rows_for_ranking).set_index("symbol")
            rank_frame["score"] = (
                0.40 * self._rank_series(rank_frame["return_20d"])
                + 0.35 * self._rank_series(rank_frame["return_60d"])
                + 0.20 * self._rank_series(rank_frame["avg_dollar_volume_20d"])
                + 0.05 * (1.0 - self._rank_series(rank_frame["volatility_20d"]))
            )
            scores = rank_frame["score"].to_dict()
            for asset in assets:
                asset.score = float(scores.get(asset.symbol, 0.0))

        return sorted(assets, key=lambda item: (-item.score, item.symbol))

    def build_shortlist(self, as_of_date: date, shortlist_size: int, include_symbols: list[str] | None = None) -> list[ScreenedAsset]:
        screened = self.screen_universe(as_of_date)
        include_set = set(include_symbols or [])
        approved = [asset for asset in screened if not asset.rejection_reasons]
        shortlisted: list[ScreenedAsset] = []
        seen: set[str] = set()

        for asset in approved:
            if asset.symbol in include_set:
                shortlisted.append(asset)
                seen.add(asset.symbol)

        for asset in approved:
            if asset.symbol in seen:
                continue
            shortlisted.append(asset)
            seen.add(asset.symbol)
            if len(shortlisted) >= shortlist_size:
                break

        return shortlisted

    def screen_symbols(self, symbols: list[str], as_of_date: date) -> list[ScreenedAsset]:
        universe_map = {row["symbol"]: row for row in self.load_universe()}
        histories = self.provider.batch_get_history(symbols, as_of_date, self.settings.data.history_lookback_days)
        assets: list[ScreenedAsset] = []
        for symbol in symbols:
            metadata = universe_map.get(
                symbol,
                {
                    "symbol": symbol,
                    "name": symbol,
                    "asset_type": "Equity",
                    "sector": "Unknown",
                },
            )
            history = histories.get(symbol)
            rejection_reasons: list[str] = []
            if history is None or history.empty:
                rejection_reasons.append("missing_history")
                assets.append(
                    ScreenedAsset(
                        symbol=symbol,
                        name=metadata["name"],
                        asset_type=metadata["asset_type"],
                        sector=metadata["sector"],
                        close=0.0,
                        avg_dollar_volume_20d=0.0,
                        return_20d=0.0,
                        return_60d=0.0,
                        volatility_20d=0.0,
                        score=0.0,
                        rejection_reasons=rejection_reasons,
                    )
                )
                continue

            close = float(history["Close"].iloc[-1])
            avg_dollar_volume_20d = float((history["Close"].tail(20) * history["Volume"].tail(20)).mean())
            returns = history["Close"].pct_change().dropna()
            return_20d = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1) if len(history) > 21 else 0.0
            return_60d = float(history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1) if len(history) > 61 else return_20d
            volatility_20d = float(returns.tail(20).std() * (252 ** 0.5)) if len(returns) >= 20 else 0.0
            if close < self.settings.data.min_price:
                rejection_reasons.append("price_below_minimum")
            if avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
                rejection_reasons.append("liquidity_below_minimum")
            assets.append(
                ScreenedAsset(
                    symbol=symbol,
                    name=metadata["name"],
                    asset_type=metadata["asset_type"],
                    sector=metadata["sector"],
                    close=close,
                    avg_dollar_volume_20d=avg_dollar_volume_20d,
                    return_20d=return_20d,
                    return_60d=return_60d,
                    volatility_20d=volatility_20d,
                    score=1.0,
                    rejection_reasons=rejection_reasons,
                )
            )
        return assets
