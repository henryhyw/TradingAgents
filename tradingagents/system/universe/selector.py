from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pandas as pd
from pydantic import Field

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import MarketDataProvider
from tradingagents.system.schemas import RegimeLabel, RegimeSnapshot, StrictModel


class ScreenedAsset(StrictModel):
    symbol: str
    name: str
    asset_type: str
    sector: str
    style_tags: list[str] = Field(default_factory=list)
    benchmark_symbol: str | None = None
    peer_group: str | None = None
    close: float
    avg_dollar_volume_20d: float
    return_20d: float
    return_60d: float
    volatility_20d: float
    relative_strength_20d: float = 0.0
    regime_fit_score: float = 0.0
    score: float
    ranking_breakdown: dict[str, float] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    watchlist_only: bool = False
    shortlist_reason: str | None = None


class UniverseSelector:
    def __init__(self, settings: SystemSettings, provider: MarketDataProvider, universe_path: Path | None = None):
        self.settings = settings
        self.provider = provider
        self.universe_path = universe_path or settings.repo_root / "tradingagents" / "system" / "universe" / "us_equities_phase1.csv"
        self.metadata_path = settings.repo_root / "tradingagents" / "system" / "universe" / "universe_metadata_overrides.csv"
        self._metadata_overrides: dict[str, dict[str, str]] | None = None

    def _load_metadata_overrides(self) -> dict[str, dict[str, str]]:
        if self._metadata_overrides is not None:
            return self._metadata_overrides
        if not self.metadata_path.exists():
            self._metadata_overrides = {}
            return self._metadata_overrides
        with self.metadata_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        overrides: dict[str, dict[str, str]] = {}
        for row in rows:
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            overrides[symbol] = {
                "style_tags": (row.get("style_tags") or "").strip(),
                "benchmark_symbol": (row.get("benchmark_symbol") or "").strip(),
                "peer_group": (row.get("peer_group") or "").strip(),
            }
        self._metadata_overrides = overrides
        return overrides

    def load_universe(self) -> list[dict[str, str]]:
        with self.universe_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        overrides = self._load_metadata_overrides()
        normalized: list[dict[str, str]] = []
        for row in rows:
            symbol = (row.get("symbol") or "").strip().upper()
            if not symbol:
                continue
            symbol_overrides = overrides.get(symbol, {})
            normalized.append(
                {
                    "symbol": symbol,
                    "name": (row.get("name") or symbol).strip(),
                    "asset_type": (row.get("asset_type") or "Equity").strip(),
                    "sector": (row.get("sector") or "Unknown").strip(),
                    "style_tags": symbol_overrides.get("style_tags") or (row.get("style_tags") or "").strip(),
                    "benchmark_symbol": symbol_overrides.get("benchmark_symbol") or (row.get("benchmark_symbol") or "").strip(),
                    "peer_group": symbol_overrides.get("peer_group") or (row.get("peer_group") or "").strip(),
                }
            )
        return normalized

    @staticmethod
    def _rank_series(values: pd.Series) -> pd.Series:
        return values.rank(pct=True, method="average").fillna(0.0)

    @staticmethod
    def _split_tags(raw: str) -> list[str]:
        tags = [part.strip().lower().replace(" ", "_") for part in raw.split("|") if part.strip()]
        return sorted(set(tags))

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        return max(lower, min(upper, value))

    def _infer_benchmark(self, row: dict[str, str]) -> str:
        if row.get("benchmark_symbol"):
            return row["benchmark_symbol"]
        if row["asset_type"].lower() == "etf":
            return "SPY"
        sector = row["sector"].lower()
        mapping = {
            "technology": "XLK",
            "financials": "XLF",
            "health care": "XLV",
            "industrials": "XLI",
            "consumer discretionary": "XLY",
            "consumer staples": "XLP",
            "energy": "XLE",
            "utilities": "XLU",
            "materials": "XLB",
            "real estate": "XLRE",
            "communication services": "XLC",
        }
        return mapping.get(sector, "SPY")

    def _regime_fit_score(self, row: dict[str, str], regime: RegimeSnapshot | None) -> float:
        if regime is None:
            return 0.5
        sector = row["sector"].lower()
        risk_on_friendly = {
            "technology",
            "consumer discretionary",
            "communication services",
            "industrials",
            "financials",
            "small cap",
        }
        defensive = {"utilities", "consumer staples", "health care", "real estate"}
        if regime.label == RegimeLabel.RISK_ON:
            if sector in risk_on_friendly:
                return 0.8
            if sector in defensive:
                return 0.4
            return 0.6
        if regime.label == RegimeLabel.RISK_OFF:
            if sector in defensive:
                return 0.75
            if sector in risk_on_friendly:
                return 0.35
            return 0.5
        if regime.label == RegimeLabel.HIGH_VOLATILITY:
            return 0.4 if row["asset_type"].lower() == "equity" else 0.55
        return 0.55

    def screen_universe(self, as_of_date: date, regime: RegimeSnapshot | None = None) -> list[ScreenedAsset]:
        universe = self.load_universe()
        symbols = [row["symbol"] for row in universe]
        all_symbols = sorted(set(symbols + ["SPY"]))
        histories = self.provider.batch_get_history(all_symbols, as_of_date, self.settings.data.history_lookback_days)
        spy_history = histories.get("SPY")
        spy_return_20d = 0.0
        if spy_history is not None and len(spy_history) > 21:
            spy_return_20d = float(spy_history["Close"].iloc[-1] / spy_history["Close"].iloc[-21] - 1)

        assets: list[ScreenedAsset] = []
        rows_for_ranking: list[dict[str, float | str]] = []

        for row in universe:
            symbol = row["symbol"]
            history = histories.get(symbol)
            rejection_reasons: list[str] = []
            quality_warnings: list[str] = []

            if history is None or history.empty:
                rejection_reasons.append("missing_history")
            elif len(history) < self.settings.data.universe_min_observations:
                rejection_reasons.append("insufficient_history")

            if rejection_reasons:
                assets.append(
                    ScreenedAsset(
                        symbol=symbol,
                        name=row["name"],
                        asset_type=row["asset_type"],
                        sector=row["sector"],
                        style_tags=self._split_tags(row["style_tags"]),
                        benchmark_symbol=self._infer_benchmark(row),
                        peer_group=row["peer_group"] or row["sector"],
                        close=0.0,
                        avg_dollar_volume_20d=0.0,
                        return_20d=0.0,
                        return_60d=0.0,
                        volatility_20d=0.0,
                        relative_strength_20d=0.0,
                        regime_fit_score=self._regime_fit_score(row, regime),
                        score=0.0,
                        ranking_breakdown={},
                        rejection_reasons=rejection_reasons,
                        quality_warnings=quality_warnings,
                    )
                )
                continue

            close = float(history["Close"].iloc[-1])
            avg_dollar_volume_20d = float((history["Close"].tail(20) * history["Volume"].tail(20)).mean())
            returns = history["Close"].pct_change().dropna()
            return_20d = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1) if len(history) > 21 else 0.0
            return_60d = float(history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1) if len(history) > 61 else return_20d
            volatility_20d = float(returns.tail(20).std() * (252 ** 0.5)) if len(returns) >= 20 else 0.0
            relative_strength_20d = return_20d - spy_return_20d
            regime_fit = self._regime_fit_score(row, regime)

            if close < self.settings.data.min_price:
                rejection_reasons.append("price_below_minimum")
            if avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
                rejection_reasons.append("liquidity_below_minimum")
            if volatility_20d <= 0:
                quality_warnings.append("volatility_missing")
            if volatility_20d > self.settings.risk.volatility_ceiling_annual:
                quality_warnings.append("high_volatility")

            watchlist_only = bool(
                "high_volatility" in quality_warnings
                and volatility_20d > (self.settings.risk.volatility_ceiling_annual * 1.25)
            )
            if watchlist_only and not rejection_reasons:
                quality_warnings.append("watchlist_due_to_volatility")

            asset = ScreenedAsset(
                symbol=symbol,
                name=row["name"],
                asset_type=row["asset_type"],
                sector=row["sector"],
                style_tags=self._split_tags(row["style_tags"]),
                benchmark_symbol=self._infer_benchmark(row),
                peer_group=row["peer_group"] or row["sector"],
                close=close,
                avg_dollar_volume_20d=avg_dollar_volume_20d,
                return_20d=return_20d,
                return_60d=return_60d,
                volatility_20d=volatility_20d,
                relative_strength_20d=relative_strength_20d,
                regime_fit_score=regime_fit,
                score=0.0,
                ranking_breakdown={},
                rejection_reasons=rejection_reasons,
                quality_warnings=quality_warnings,
                watchlist_only=watchlist_only,
            )
            assets.append(asset)
            if not rejection_reasons:
                rows_for_ranking.append(
                    {
                        "symbol": symbol,
                        "avg_dollar_volume_20d": avg_dollar_volume_20d,
                        "return_20d": return_20d,
                        "return_60d": return_60d,
                        "volatility_20d": volatility_20d,
                        "relative_strength_20d": relative_strength_20d,
                        "regime_fit_score": regime_fit,
                    }
                )

        if rows_for_ranking:
            rank_frame = pd.DataFrame(rows_for_ranking).set_index("symbol")
            momentum_score = (
                0.55 * self._rank_series(rank_frame["return_20d"])
                + 0.45 * self._rank_series(rank_frame["return_60d"])
            )
            liquidity_score = self._rank_series(rank_frame["avg_dollar_volume_20d"])
            stability_score = 1.0 - self._rank_series(rank_frame["volatility_20d"])
            relative_strength_score = self._rank_series(rank_frame["relative_strength_20d"])
            regime_fit_score = rank_frame["regime_fit_score"].apply(self._clamp)

            rank_frame["score"] = (
                0.30 * momentum_score
                + 0.20 * liquidity_score
                + 0.15 * stability_score
                + 0.20 * relative_strength_score
                + 0.15 * regime_fit_score
            )
            for asset in assets:
                if asset.symbol not in rank_frame.index:
                    continue
                asset.score = float(rank_frame.loc[asset.symbol, "score"])
                asset.ranking_breakdown = {
                    "momentum": float(momentum_score[asset.symbol]),
                    "liquidity": float(liquidity_score[asset.symbol]),
                    "stability": float(stability_score[asset.symbol]),
                    "relative_strength": float(relative_strength_score[asset.symbol]),
                    "regime_fit": float(regime_fit_score[asset.symbol]),
                }

        return sorted(assets, key=lambda item: (-item.score, item.symbol))

    def build_shortlist_from_screened(
        self,
        screened: list[ScreenedAsset],
        shortlist_size: int,
        include_symbols: list[str] | None = None,
    ) -> list[ScreenedAsset]:
        include_set = set(include_symbols or [])
        shortlist: list[ScreenedAsset] = []
        sector_counts: dict[str, int] = {}
        seen: set[str] = set()
        max_per_sector = self.settings.run.max_shortlist_per_sector

        # Keep existing holdings in the shortlist for rebalance/exit logic.
        for asset in screened:
            if asset.symbol not in include_set:
                continue
            asset.shortlist_reason = "existing_position_coverage"
            shortlist.append(asset)
            seen.add(asset.symbol)
            sector_counts[asset.sector] = sector_counts.get(asset.sector, 0) + 1

        approved = [asset for asset in screened if not asset.rejection_reasons and not asset.watchlist_only]
        watchlist = [asset for asset in screened if not asset.rejection_reasons and asset.watchlist_only]

        for asset in approved:
            if len(shortlist) >= shortlist_size:
                break
            if asset.symbol in seen:
                continue
            if sector_counts.get(asset.sector, 0) >= max_per_sector:
                continue
            asset.shortlist_reason = (
                f"ranked_candidate(score={asset.score:.3f}; "
                f"momentum={asset.ranking_breakdown.get('momentum', 0.0):.2f}; "
                f"regime_fit={asset.ranking_breakdown.get('regime_fit', 0.0):.2f})"
            )
            shortlist.append(asset)
            seen.add(asset.symbol)
            sector_counts[asset.sector] = sector_counts.get(asset.sector, 0) + 1

        for asset in watchlist:
            if len(shortlist) >= shortlist_size:
                break
            if asset.symbol in seen:
                continue
            asset.shortlist_reason = "watchlist_candidate_data_quality"
            shortlist.append(asset)
            seen.add(asset.symbol)

        return shortlist

    def build_shortlist(
        self,
        as_of_date: date,
        shortlist_size: int,
        include_symbols: list[str] | None = None,
        regime: RegimeSnapshot | None = None,
    ) -> list[ScreenedAsset]:
        screened = self.screen_universe(as_of_date, regime=regime)
        return self.build_shortlist_from_screened(
            screened=screened,
            shortlist_size=shortlist_size,
            include_symbols=include_symbols,
        )

    def screen_symbols(
        self,
        symbols: list[str],
        as_of_date: date,
        regime: RegimeSnapshot | None = None,
    ) -> list[ScreenedAsset]:
        universe_map = {row["symbol"]: row for row in self.load_universe()}
        histories = self.provider.batch_get_history(symbols, as_of_date, self.settings.data.history_lookback_days)
        spy_history = self.provider.get_history("SPY", as_of_date, self.settings.data.history_lookback_days)
        spy_return_20d = 0.0
        if not spy_history.empty and len(spy_history) > 21:
            spy_return_20d = float(spy_history["Close"].iloc[-1] / spy_history["Close"].iloc[-21] - 1)

        assets: list[ScreenedAsset] = []
        for symbol in symbols:
            metadata = universe_map.get(
                symbol,
                {
                    "symbol": symbol,
                    "name": symbol,
                    "asset_type": "Equity",
                    "sector": "Unknown",
                    "style_tags": "",
                    "benchmark_symbol": "",
                    "peer_group": "",
                },
            )
            history = histories.get(symbol)
            rejection_reasons: list[str] = []
            quality_warnings: list[str] = []
            if history is None or history.empty:
                rejection_reasons.append("missing_history")
                assets.append(
                    ScreenedAsset(
                        symbol=symbol,
                        name=metadata["name"],
                        asset_type=metadata["asset_type"],
                        sector=metadata["sector"],
                        style_tags=self._split_tags(metadata.get("style_tags", "")),
                        benchmark_symbol=self._infer_benchmark(metadata),
                        peer_group=metadata.get("peer_group") or metadata["sector"],
                        close=0.0,
                        avg_dollar_volume_20d=0.0,
                        return_20d=0.0,
                        return_60d=0.0,
                        volatility_20d=0.0,
                        relative_strength_20d=0.0,
                        regime_fit_score=self._regime_fit_score(metadata, regime),
                        score=0.0,
                        ranking_breakdown={},
                        rejection_reasons=rejection_reasons,
                        quality_warnings=quality_warnings,
                        shortlist_reason="manual_symbol_override_missing_data",
                    )
                )
                continue

            close = float(history["Close"].iloc[-1])
            avg_dollar_volume_20d = float((history["Close"].tail(20) * history["Volume"].tail(20)).mean())
            returns = history["Close"].pct_change().dropna()
            return_20d = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1) if len(history) > 21 else 0.0
            return_60d = float(history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1) if len(history) > 61 else return_20d
            volatility_20d = float(returns.tail(20).std() * (252 ** 0.5)) if len(returns) >= 20 else 0.0
            relative_strength = return_20d - spy_return_20d

            if close < self.settings.data.min_price:
                rejection_reasons.append("price_below_minimum")
            if avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
                rejection_reasons.append("liquidity_below_minimum")

            score = 1.0 if not rejection_reasons else 0.0
            assets.append(
                ScreenedAsset(
                    symbol=symbol,
                    name=metadata["name"],
                    asset_type=metadata["asset_type"],
                    sector=metadata["sector"],
                    style_tags=self._split_tags(metadata.get("style_tags", "")),
                    benchmark_symbol=self._infer_benchmark(metadata),
                    peer_group=metadata.get("peer_group") or metadata["sector"],
                    close=close,
                    avg_dollar_volume_20d=avg_dollar_volume_20d,
                    return_20d=return_20d,
                    return_60d=return_60d,
                    volatility_20d=volatility_20d,
                    relative_strength_20d=relative_strength,
                    regime_fit_score=self._regime_fit_score(metadata, regime),
                    score=score,
                    ranking_breakdown={
                        "manual_override": 1.0 if not rejection_reasons else 0.0,
                        "relative_strength": relative_strength,
                    },
                    rejection_reasons=rejection_reasons,
                    quality_warnings=quality_warnings,
                    shortlist_reason="manual_symbol_override",
                )
            )
        return assets
