from __future__ import annotations

from datetime import date
import re
from typing import Any

import pandas as pd

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import FundamentalSnapshot, MarketDataProvider, NewsItem
from tradingagents.system.schemas import (
    AnalystMemo,
    BearCaseMemo,
    BullCaseMemo,
    CandidateAssessment,
    DebateSummary,
    EntryMode,
    OrderIntentType,
    PositionSnapshot,
    RegimeSnapshot,
    ResearchBundle,
    ResearchDecision,
    TradeAction,
)

from .adapter import ResearchAdapter


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


_BULLISH_THESIS_HINTS = (
    "buy",
    "accumulate",
    "entry",
    "initiate",
    "long",
    "upside",
    "bullish",
    "breakout",
    "momentum",
    "outperform",
    "relative strength",
    "supportive",
    "favorable setup",
)
_BEARISH_THESIS_HINTS = (
    "sell",
    "trim",
    "exit",
    "underweight",
    "bearish",
    "downside",
    "avoid new entries",
    "unfavorable risk/reward",
    "wait for pullback",
    "de-risk",
    "reduce exposure",
    "not a buy",
    "no entry",
    "no-entry",
)
_NO_ENTRY_THESIS_HINTS = (
    "avoid",
    "no entry",
    "no-entry",
    "not a buy",
    "defer",
    "wait for pullback",
    "insufficient research",
    "fallback",
    "hold off",
)
_BUY_INCONSISTENT_PHRASES = (
    "sell",
    "avoid new entries",
    "trim existing positions",
    "reduce exposure",
    "wait for pullback",
    "unfavorable risk/reward",
    "exit current positions",
    "underweight",
    "capital preservation over chasing rally",
    "no margin of safety",
    "do not initiate new position",
    "tactical wait",
    "hold existing but defer new capital",
    "no-entry",
    "no entry",
    "not a buy",
    "defer new capital",
)
_HARD_ENTRY_BLOCKERS = {
    "insufficient_history_for_entry_mode",
    "watchlist_only_candidate",
    "candidate_not_eligible",
    "liquidity_below_minimum",
    "technical_signal_not_bullish",
    "extension_overheat_block",
    "rsi_overheat_block",
}
_SOFT_ENTRY_BLOCKERS = {
    "missing_breakout_confirmation",
    "missing_pullback_confirmation",
    "technical_confidence_too_low",
    "relative_strength_not_positive",
    "short_term_return_not_positive",
}


class ResearchOrganization:
    def __init__(self, settings: SystemSettings, provider: MarketDataProvider, adapter: ResearchAdapter):
        self.settings = settings
        self.provider = provider
        self.adapter = adapter

    @staticmethod
    def _compute_rsi(history: pd.DataFrame, period: int = 14) -> float | None:
        if history.empty or len(history) <= period + 1:
            return None
        delta = history["Close"].diff()
        gains = delta.clip(lower=0).rolling(period).mean()
        losses = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gains.iloc[-1] / losses.iloc[-1] if losses.iloc[-1] > 0 else float("inf")
        rsi = 100 - (100 / (1 + rs))
        if pd.isna(rsi):
            return None
        return float(rsi)

    def _technical_memo(self, symbol: str, as_of_date: date, history: pd.DataFrame) -> AnalystMemo:
        warnings: list[str] = []
        technical_state = self._technical_state(history)
        if technical_state["insufficient_history"]:
            return AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Technical Analyst",
                signal="neutral",
                confidence=0.2,
                summary="Insufficient history for robust technical analysis.",
                evidence=[],
                warnings=["insufficient_price_history"],
            )

        close = float(technical_state["close"])
        sma_20 = float(technical_state["sma20"])
        sma_50 = float(technical_state["sma50"])
        ret_20 = float(technical_state["ret_20d"])
        ret_60 = float(technical_state["ret_60d"])
        vol_20 = float(technical_state["vol_20d"])
        drawdown = float(technical_state["drawdown_120d"])
        rsi = float(technical_state["rsi14"])

        bullish_votes = 0
        bearish_votes = 0
        if close > sma_20:
            bullish_votes += 1
        else:
            bearish_votes += 1
        if sma_20 > sma_50:
            bullish_votes += 1
        else:
            bearish_votes += 1
        if ret_20 > 0:
            bullish_votes += 1
        else:
            bearish_votes += 1
        if drawdown < -0.15:
            bearish_votes += 1

        if bullish_votes > bearish_votes:
            signal = "bullish"
        elif bearish_votes > bullish_votes:
            signal = "bearish"
        else:
            signal = "neutral"

        confidence = _clamp(0.35 + abs(ret_20) * 2.5 + abs(ret_60) * 1.5, 0.25, 0.85)
        summary = (
            f"Close={close:.2f}, SMA20={sma_20:.2f}, SMA50={sma_50:.2f}, "
            f"20d return={ret_20:.2%}, 60d return={ret_60:.2%}, "
            f"20d vol={vol_20:.2%}, drawdown={drawdown:.2%}, RSI14={rsi:.1f}."
        )
        evidence = [
            "trend_above_sma20" if close > sma_20 else "trend_below_sma20",
            "sma20_above_sma50" if sma_20 > sma_50 else "sma20_below_sma50",
            "positive_20d_return" if ret_20 > 0 else "negative_20d_return",
            "rsi_overbought" if rsi >= 70 else ("rsi_oversold" if rsi <= 30 else "rsi_neutral"),
            "breakout_confirmed" if technical_state["breakout_confirmed"] else "breakout_unconfirmed",
            "pullback_confirmed" if technical_state["pullback_confirmed"] else "pullback_unconfirmed",
        ]
        return AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="Technical Analyst",
            signal=signal,
            confidence=confidence,
            summary=summary,
            evidence=evidence,
            warnings=warnings,
        )

    def _fundamental_memo(
        self,
        symbol: str,
        as_of_date: date,
        fundamentals: FundamentalSnapshot,
    ) -> AnalystMemo:
        warnings: list[str] = []
        evidence: list[str] = []
        positives = 0
        negatives = 0

        if fundamentals.market_cap is None:
            warnings.append("missing_market_cap")
        else:
            evidence.append(f"market_cap={fundamentals.market_cap:,.0f}")

        if fundamentals.trailing_pe is not None:
            if fundamentals.trailing_pe <= 22:
                positives += 1
                evidence.append("valuation_reasonable_trailing_pe")
            elif fundamentals.trailing_pe >= 35:
                negatives += 1
                evidence.append("valuation_expensive_trailing_pe")
        else:
            warnings.append("missing_trailing_pe")

        if fundamentals.forward_pe is not None:
            if fundamentals.trailing_pe and fundamentals.forward_pe < fundamentals.trailing_pe:
                positives += 1
                evidence.append("forward_pe_improves")
            elif fundamentals.trailing_pe and fundamentals.forward_pe > fundamentals.trailing_pe:
                negatives += 1
                evidence.append("forward_pe_worsens")
        else:
            warnings.append("missing_forward_pe")

        if fundamentals.beta is not None and fundamentals.beta > 1.6:
            negatives += 1
            evidence.append("high_beta_risk")
        if fundamentals.price_to_book is not None and fundamentals.price_to_book > 8:
            negatives += 1
            evidence.append("high_price_to_book")

        if positives > negatives:
            signal = "bullish"
        elif negatives > positives:
            signal = "bearish"
        else:
            signal = "neutral"
        confidence = _clamp(0.30 + (positives + negatives) * 0.08, 0.25, 0.75)

        summary = (
            f"Sector={fundamentals.sector or 'unknown'}, "
            f"trailing PE={fundamentals.trailing_pe if fundamentals.trailing_pe is not None else 'n/a'}, "
            f"forward PE={fundamentals.forward_pe if fundamentals.forward_pe is not None else 'n/a'}, "
            f"price/book={fundamentals.price_to_book if fundamentals.price_to_book is not None else 'n/a'}, "
            f"beta={fundamentals.beta if fundamentals.beta is not None else 'n/a'}."
        )
        return AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="Fundamental Analyst",
            signal=signal,
            confidence=confidence,
            summary=summary,
            evidence=evidence,
            warnings=warnings,
        )

    @staticmethod
    def _news_tone(items: list[NewsItem]) -> tuple[int, int]:
        positive_words = {"beat", "growth", "upgrade", "record", "strong", "expansion", "surge", "partnership"}
        negative_words = {"miss", "downgrade", "lawsuit", "probe", "decline", "cut", "weak", "layoff"}
        pos = 0
        neg = 0
        for item in items:
            text = f"{item.title} {item.summary}".lower()
            if any(word in text for word in positive_words):
                pos += 1
            if any(word in text for word in negative_words):
                neg += 1
        return pos, neg

    def _news_memos(self, symbol: str, as_of_date: date, items: list[NewsItem]) -> tuple[AnalystMemo, AnalystMemo]:
        if not items:
            news_summary = "No recent news items were returned by the default yfinance feed."
            news_memo = AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="News/Event Analyst",
                signal="neutral",
                confidence=0.2,
                summary=news_summary,
                evidence=[],
                warnings=["news_sparse"],
            )
            sentiment_memo = AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Sentiment/Narrative Analyst",
                signal="neutral",
                confidence=0.2,
                summary="Narrative confidence is low because news coverage is sparse.",
                evidence=[],
                warnings=["sentiment_low_evidence"],
            )
            return news_memo, sentiment_memo

        top_titles = [item.title for item in items[:3]]
        evidence = [f"title:{title[:120]}" for title in top_titles]
        pos, neg = self._news_tone(items)
        if pos > neg:
            signal = "bullish"
        elif neg > pos:
            signal = "bearish"
        else:
            signal = "neutral"
        confidence = _clamp(0.30 + (pos + neg) * 0.06, 0.25, 0.75)
        news_memo = AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="News/Event Analyst",
            signal=signal,
            confidence=confidence,
            summary=f"Reviewed {len(items)} recent items; top headlines captured for event context.",
            evidence=evidence,
            warnings=[],
        )
        narrative = (
            f"Narrative tone counts from available headlines: positive={pos}, negative={neg}, neutral={max(len(items) - pos - neg, 0)}."
        )
        sentiment_memo = AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="Sentiment/Narrative Analyst",
            signal=signal if (pos != neg) else "mixed",
            confidence=_clamp(0.25 + abs(pos - neg) * 0.08, 0.2, 0.7),
            summary=narrative,
            evidence=[f"headline_count={len(items)}", f"positive={pos}", f"negative={neg}"],
            warnings=[],
        )
        return news_memo, sentiment_memo

    def _regime_memos(
        self,
        symbol: str,
        as_of_date: date,
        regime: RegimeSnapshot | None,
    ) -> tuple[AnalystMemo, AnalystMemo]:
        if regime is None:
            neutral = AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Market Regime Analyst",
                signal="neutral",
                confidence=0.2,
                summary="Regime snapshot unavailable; operating under neutral assumptions.",
                evidence=[],
                warnings=["regime_unavailable"],
            )
            return neutral, neutral.model_copy(update={"role": "Macro Proxy Analyst"})

        regime_signal = "bullish" if regime.label == "risk_on" else ("bearish" if regime.label in {"risk_off", "high_volatility"} else "neutral")
        regime_memo = AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="Market Regime Analyst",
            signal=regime_signal,
            confidence=_clamp(0.45 + abs(regime.risk_on_score) * 0.35, 0.3, 0.85),
            summary=(
                f"Regime={regime.label.value}, trend={regime.trend_regime}, volatility={regime.volatility_regime}, "
                f"risk_on_score={regime.risk_on_score:+.2f}, risk_budget_multiplier={regime.risk_budget_multiplier:.2f}."
            ),
            evidence=[f"{name}={value:.4f}" for name, value in sorted(regime.signals.items())[:6]],
            warnings=regime.warnings,
        )
        macro_evidence = []
        for key in ("qqq_relative_strength_20d", "iwm_relative_strength_20d", "duration_return_20d", "dollar_return_20d", "vix_level"):
            if key in regime.signals:
                macro_evidence.append(f"{key}={regime.signals[key]:.4f}")
        macro_memo = AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="Macro Proxy Analyst",
            signal=regime_signal,
            confidence=_clamp(regime_memo.confidence - 0.05, 0.2, 0.8),
            summary="Macro proxy panel built from liquid ETFs/indexes; no paid macro feed assumptions.",
            evidence=macro_evidence,
            warnings=regime.warnings,
        )
        return regime_memo, macro_memo

    @staticmethod
    def _has_inventory(current_position: PositionSnapshot | None) -> bool:
        return current_position is not None and current_position.quantity > 0

    def _technical_state(self, history: pd.DataFrame) -> dict[str, Any]:
        state: dict[str, Any] = {
            "close": 0.0,
            "sma20": 0.0,
            "sma50": 0.0,
            "ret_3d": 0.0,
            "ret_20d": 0.0,
            "ret_60d": 0.0,
            "vol_20d": 0.0,
            "drawdown_120d": 0.0,
            "rsi14": 50.0,
            "prior_high": 0.0,
            "extension_over_ma20": 0.0,
            "extension_over_ma50": 0.0,
            "breakout_distance": 0.0,
            "pullback_distance_ma20": 1.0,
            "pullback_distance_ma50": 1.0,
            "volume_surge_5d": 1.0,
            "trend_alignment": False,
            "breakout_confirmed": False,
            "pullback_confirmed": False,
            "reacceleration_confirmed": False,
            "insufficient_history": True,
        }
        if history.empty or len(history) < 30:
            return state
        close = float(history["Close"].iloc[-1])
        sma20 = float(history["Close"].tail(20).mean())
        sma50 = float(history["Close"].tail(50).mean()) if len(history) >= 50 else sma20
        ret_3d = float(history["Close"].iloc[-1] / history["Close"].iloc[-4] - 1.0) if len(history) >= 4 else 0.0
        ret_20d = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1.0) if len(history) > 21 else 0.0
        ret_60d = float(history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1.0) if len(history) > 61 else ret_20d
        vol_20d = float(history["Close"].pct_change().tail(20).std() * (252 ** 0.5))
        rolling_peak = float(history["Close"].tail(120).max())
        drawdown = 0.0 if rolling_peak <= 0 else (close / rolling_peak) - 1.0
        rsi = self._compute_rsi(history)
        rsi = 50.0 if rsi is None else float(rsi)
        prior_high = float(history["Close"].iloc[-self.settings.data.breakout_lookback_days - 1 : -1].max())
        extension_over_ma20 = 0.0 if sma20 <= 0 else (close / sma20) - 1.0
        extension_over_ma50 = 0.0 if sma50 <= 0 else (close / sma50) - 1.0
        breakout_distance = 0.0 if prior_high <= 0 else (close / prior_high) - 1.0
        pullback_distance_ma20 = 1.0 if sma20 <= 0 else abs((close / sma20) - 1.0)
        pullback_distance_ma50 = 1.0 if sma50 <= 0 else abs((close / sma50) - 1.0)
        vol_5 = float(history["Volume"].tail(5).mean()) if len(history) >= 5 else 0.0
        vol_20 = float(history["Volume"].tail(20).mean()) if len(history) >= 20 else 0.0
        volume_surge = 1.0 if vol_20 <= 0 else vol_5 / vol_20
        trend_alignment = close > sma20 and sma20 >= (sma50 * 0.995) and ret_60d > -0.01
        breakout_confirmed = breakout_distance >= self.settings.data.breakout_confirmation_buffer_fraction and trend_alignment
        pullback_confirmed = (
            trend_alignment
            and (
                pullback_distance_ma20 <= self.settings.data.pullback_max_distance_fraction
                or pullback_distance_ma50 <= (self.settings.data.pullback_max_distance_fraction * 0.85)
            )
        )
        reacceleration_confirmed = ret_3d >= self.settings.data.pullback_reacceleration_min_return_3d
        return {
            "close": close,
            "sma20": sma20,
            "sma50": sma50,
            "ret_3d": ret_3d,
            "ret_20d": ret_20d,
            "ret_60d": ret_60d,
            "vol_20d": vol_20d,
            "drawdown_120d": drawdown,
            "rsi14": rsi,
            "prior_high": prior_high,
            "extension_over_ma20": extension_over_ma20,
            "extension_over_ma50": extension_over_ma50,
            "breakout_distance": breakout_distance,
            "pullback_distance_ma20": pullback_distance_ma20,
            "pullback_distance_ma50": pullback_distance_ma50,
            "volume_surge_5d": volume_surge,
            "trend_alignment": trend_alignment,
            "breakout_confirmed": breakout_confirmed,
            "pullback_confirmed": pullback_confirmed,
            "reacceleration_confirmed": reacceleration_confirmed,
            "insufficient_history": False,
        }

    def _entry_mode_and_gates(
        self,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
        technical_state: dict[str, Any],
    ) -> tuple[EntryMode, str, list[str], float, float]:
        blockers: list[str] = []
        if technical_state.get("insufficient_history", False):
            blockers.append("insufficient_history_for_entry_mode")
            return EntryMode.NONE, "insufficient_history_for_entry_mode", blockers, 0.0, 0.0
        if candidate is not None and candidate.watchlist_only:
            blockers.append("watchlist_only_candidate")
        if candidate is not None and not candidate.eligible:
            blockers.append("candidate_not_eligible")
        if candidate is not None and candidate.avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
            blockers.append("liquidity_below_minimum")
        if regime is not None and regime.label.value in {"risk_off", "high_volatility"}:
            blockers.append(f"regime_{regime.label.value}_blocks_new_entries")
        if technical_memo.signal != "bullish":
            blockers.append("technical_signal_not_bullish")
        risk_on_contained = self._is_risk_on_contained(regime)
        entry_confidence_min = self.settings.data.entry_confidence_min
        if risk_on_contained:
            entry_confidence_min = max(0.50, entry_confidence_min - self.settings.data.risk_on_entry_confidence_relief)
        if technical_memo.confidence < entry_confidence_min:
            blockers.append("technical_confidence_too_low")
        if candidate is not None and candidate.relative_strength_20d <= 0:
            blockers.append("relative_strength_not_positive")
        if candidate is not None and candidate.return_20d <= 0:
            blockers.append("short_term_return_not_positive")

        extension_over_ma20 = float(technical_state.get("extension_over_ma20", 0.0))
        rsi14 = float(technical_state.get("rsi14", 50.0))
        extension_penalty = _clamp(
            max(0.0, extension_over_ma20 - self.settings.data.max_extension_over_ma20_fraction)
            / max(self.settings.data.max_extension_over_ma20_fraction, 1e-6),
            0.0,
            1.0,
        )
        extension_factor = _clamp(
            extension_over_ma20 / max(self.settings.data.max_extension_over_ma20_fraction, 1e-6),
            0.0,
            1.0,
        )
        overheat_penalty = _clamp(
            (max(0.0, rsi14 - self.settings.data.overheat_rsi_threshold) / 20.0) * extension_factor
            + (max(0.0, extension_over_ma20 - self.settings.data.overheat_extension_fraction) / 0.10),
            0.0,
            1.0,
        )
        if extension_over_ma20 > self.settings.data.overheat_extension_fraction:
            blockers.append("extension_overheat_block")
        if (
            rsi14 > self.settings.data.overheat_rsi_threshold
            and extension_over_ma20 > (self.settings.data.max_extension_over_ma20_fraction * 0.85)
        ):
            blockers.append("rsi_overheat_block")

        trend_alignment = bool(technical_state.get("trend_alignment", False))
        ret_3d = float(technical_state.get("ret_3d", 0.0))
        ret_20d = float(technical_state.get("ret_20d", 0.0))
        breakout_distance = float(technical_state.get("breakout_distance", 0.0))
        pullback_distance_ma20 = float(technical_state.get("pullback_distance_ma20", 1.0))
        pullback_distance_ma50 = float(technical_state.get("pullback_distance_ma50", 1.0))
        candidate_relative_strength = 0.0 if candidate is None else candidate.relative_strength_20d
        candidate_avg_dollar_volume = 0.0 if candidate is None else candidate.avg_dollar_volume_20d
        candidate_return_20d = ret_20d if candidate is None else candidate.return_20d

        breakout_buffer = self.settings.data.breakout_confirmation_buffer_fraction
        pullback_distance_limit = self.settings.data.pullback_max_distance_fraction
        reacceleration_min = self.settings.data.pullback_reacceleration_min_return_3d
        if risk_on_contained:
            breakout_buffer *= self.settings.data.risk_on_breakout_buffer_multiplier
            pullback_distance_limit *= self.settings.data.risk_on_pullback_distance_multiplier
            reacceleration_min = min(reacceleration_min, self.settings.data.risk_on_pullback_reacceleration_min_return_3d)

        breakout_confirmed = breakout_distance >= breakout_buffer and trend_alignment
        strong_risk_on_trend = (
            risk_on_contained
            and trend_alignment
            and candidate_return_20d >= self.settings.data.risk_on_min_strong_trend_return_20d
            and candidate_relative_strength >= self.settings.data.risk_on_min_relative_strength_20d
            and candidate_avg_dollar_volume >= (self.settings.data.min_avg_dollar_volume * 2.0)
            and extension_over_ma20 <= (self.settings.data.max_extension_over_ma20_fraction * 1.20)
            and overheat_penalty < 0.65
        )
        risk_on_relaxed_breakout = (
            strong_risk_on_trend and breakout_distance >= self.settings.data.risk_on_near_breakout_floor
        )
        risk_on_near_miss_structure = (
            risk_on_contained
            and trend_alignment
            and candidate_return_20d >= (self.settings.data.risk_on_min_strong_trend_return_20d * 0.75)
            and candidate_relative_strength >= -0.005
            and candidate_avg_dollar_volume >= (self.settings.data.min_avg_dollar_volume * 1.5)
            and extension_over_ma20 <= (self.settings.data.max_extension_over_ma20_fraction * 1.25)
            and extension_penalty < self.settings.data.risk_on_near_miss_max_extension_penalty
            and overheat_penalty < self.settings.data.risk_on_near_miss_max_overheat_penalty
        )
        pullback_confirmed = trend_alignment and (
            pullback_distance_ma20 <= pullback_distance_limit
            or pullback_distance_ma50 <= (pullback_distance_limit * 0.85)
        )
        reacceleration_confirmed = ret_3d >= reacceleration_min
        breakout_mode_ok = breakout_confirmed or risk_on_relaxed_breakout
        pullback_mode_ok = pullback_confirmed and reacceleration_confirmed

        if regime is not None and regime.label.value == "balanced":
            # Balanced regime prefers pullback entries unless breakout quality is exceptional.
            breakout_mode_ok = breakout_mode_ok and breakout_distance >= 0.015 and rsi14 < 76.0

        if blockers:
            if not breakout_mode_ok:
                blockers.append("missing_breakout_confirmation")
            if not pullback_mode_ok:
                blockers.append("missing_pullback_confirmation")
            if (
                risk_on_near_miss_structure
                and not breakout_mode_ok
                and breakout_distance >= (self.settings.data.risk_on_near_breakout_floor - 0.006)
            ):
                blockers.append("near_miss_breakout_confirmation")
            if (
                risk_on_near_miss_structure
                and not pullback_mode_ok
                and min(pullback_distance_ma20, pullback_distance_ma50) <= (pullback_distance_limit * 1.20)
                and ret_3d >= 0.0
            ):
                blockers.append("near_miss_pullback_confirmation")
            return EntryMode.NONE, blockers[0], list(dict.fromkeys(blockers)), extension_penalty, overheat_penalty
        true_breakout_ok = breakout_confirmed
        if regime is not None and regime.label.value == "balanced":
            true_breakout_ok = true_breakout_ok and breakout_distance >= 0.015 and rsi14 < 76.0
        if true_breakout_ok:
            return EntryMode.BREAKOUT, "breakout_confirmation_passed", [], extension_penalty, overheat_penalty
        if pullback_mode_ok:
            if pullback_distance_ma20 <= pullback_distance_limit and breakout_distance < 0.04:
                reason = "risk_on_relaxed_pullback_confirmation" if risk_on_contained else "pullback_confirmation_passed"
                return EntryMode.PULLBACK, reason, [], extension_penalty, overheat_penalty
        if breakout_mode_ok:
            reason = "risk_on_relaxed_breakout_confirmation" if risk_on_relaxed_breakout and not breakout_confirmed else "breakout_confirmation_passed"
            return EntryMode.BREAKOUT, reason, [], extension_penalty, overheat_penalty
        if pullback_mode_ok:
            reason = "risk_on_relaxed_pullback_confirmation" if risk_on_contained else "pullback_confirmation_passed"
            return EntryMode.PULLBACK, reason, [], extension_penalty, overheat_penalty
        blockers.extend(["missing_breakout_confirmation", "missing_pullback_confirmation"])
        if risk_on_near_miss_structure and breakout_distance >= (self.settings.data.risk_on_near_breakout_floor - 0.006):
            blockers.append("near_miss_breakout_confirmation")
        if (
            risk_on_near_miss_structure
            and min(pullback_distance_ma20, pullback_distance_ma50) <= (pullback_distance_limit * 1.20)
            and ret_3d >= 0.0
        ):
            blockers.append("near_miss_pullback_confirmation")
        return EntryMode.NONE, "entry_confirmation_missing", blockers, extension_penalty, overheat_penalty

    def _entry_gate_satisfied(
        self,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
        technical_state: dict[str, Any],
    ) -> tuple[bool, str]:
        entry_mode, reason, _, _, _ = self._entry_mode_and_gates(
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            technical_state=technical_state,
        )
        if entry_mode == EntryMode.NONE:
            return False, reason
        return True, reason

    def _classify_entry_reject(
        self,
        *,
        decision: ResearchDecision,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
        debate: DebateSummary,
        entry_mode: EntryMode,
        entry_blockers: list[str],
        extension_penalty: float,
        overheat_penalty: float,
        fallback_origin: bool,
        thesis_semantics: dict[str, int | bool],
    ) -> tuple[str, list[str], bool]:
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        near_miss = any(blocker.startswith("near_miss_") for blocker in entry_blockers)

        for blocker in entry_blockers:
            if blocker in _HARD_ENTRY_BLOCKERS or blocker.startswith("regime_"):
                hard_reasons.append(blocker)
            elif blocker in _SOFT_ENTRY_BLOCKERS or blocker.startswith("near_miss_"):
                soft_reasons.append(blocker)

        if fallback_origin:
            hard_reasons.append("fallback_origin")
        if bool(thesis_semantics.get("strongly_bearish", False)) or int(thesis_semantics.get("no_entry_hits", 0)) > 0:
            hard_reasons.append("thesis_no_entry_or_bearish")
        if technical_memo.signal != "bullish":
            hard_reasons.append("technical_signal_not_bullish")
        if extension_penalty >= self.settings.data.risk_on_near_miss_max_extension_penalty:
            hard_reasons.append("extension_penalty_too_high")
        if overheat_penalty >= self.settings.data.risk_on_near_miss_max_overheat_penalty:
            hard_reasons.append("overheat_penalty_too_high")
        if candidate is None:
            hard_reasons.append("missing_candidate")
        elif not candidate.eligible or candidate.watchlist_only:
            hard_reasons.append("candidate_not_tradable")
        elif candidate.avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
            hard_reasons.append("liquidity_below_minimum")

        if hard_reasons:
            return "hard_reject", list(dict.fromkeys(hard_reasons)), False

        risk_on_contained = self._is_risk_on_contained(regime)
        confidence_floor = max(
            0.46,
            self.settings.data.entry_confidence_min - self.settings.data.risk_on_near_miss_confidence_relief,
        )
        relative_strength_ok = candidate is not None and candidate.relative_strength_20d >= -0.005
        momentum_ok = candidate is not None and candidate.return_20d >= (
            self.settings.data.risk_on_min_strong_trend_return_20d * 0.75
        )
        liquidity_ok = candidate is not None and candidate.avg_dollar_volume_20d >= (
            self.settings.data.min_avg_dollar_volume * 1.5
        )
        debate_not_hard_bear = not (debate.winning_side == "bear" and debate.confidence_balance >= 0.70)
        starter_eligible = (
            risk_on_contained
            and near_miss
            and entry_mode == EntryMode.NONE
            and technical_memo.confidence >= confidence_floor
            and decision.confidence >= confidence_floor
            and relative_strength_ok
            and momentum_ok
            and liquidity_ok
            and debate_not_hard_bear
        )
        if starter_eligible:
            return "starter_eligible_near_miss", list(dict.fromkeys(soft_reasons or ["near_miss"])), True
        if near_miss or soft_reasons:
            return "soft_reject", list(dict.fromkeys(soft_reasons or ["soft_reject"])), False
        return "none", [], False

    def _starter_position_fraction(
        self,
        *,
        decision: ResearchDecision,
        candidate: CandidateAssessment | None,
        extension_penalty: float,
        overheat_penalty: float,
        entry_blockers: list[str],
    ) -> float:
        base = self.settings.risk.risk_on_starter_position_fraction
        quality = 0.0
        if candidate is not None:
            quality += _clamp((candidate.avg_dollar_volume_20d / max(self.settings.data.min_avg_dollar_volume, 1.0) - 1.0) / 6.0, 0.0, 0.20)
            quality += _clamp((candidate.relative_strength_20d + 0.005) / 0.055, 0.0, 0.20)
            quality += _clamp((candidate.return_20d - 0.02) / 0.12, 0.0, 0.20)
        quality += _clamp((decision.confidence - 0.46) / 0.24, 0.0, 0.15)
        if "near_miss_breakout_confirmation" in entry_blockers and "near_miss_pullback_confirmation" in entry_blockers:
            quality -= 0.08
        quality -= (0.15 * extension_penalty) + (0.20 * overheat_penalty)
        sized = base * _clamp(0.55 + quality, 0.50, 1.10)
        return _clamp(
            sized,
            self.settings.risk.risk_on_starter_min_fraction,
            self.settings.risk.risk_on_starter_max_fraction,
        )

    @staticmethod
    def _thesis_semantics(thesis: str) -> dict[str, int | bool]:
        normalized = re.sub(r"\s+", " ", (thesis or "").lower()).strip()
        bullish_hits = sum(1 for token in _BULLISH_THESIS_HINTS if token in normalized)
        bearish_hits = sum(1 for token in _BEARISH_THESIS_HINTS if token in normalized)
        no_entry_hits = sum(1 for token in _NO_ENTRY_THESIS_HINTS if token in normalized)
        strongly_bearish = no_entry_hits > 0 or bearish_hits >= bullish_hits + 2
        bullish_supported = no_entry_hits == 0 and bullish_hits >= max(1, bearish_hits)
        return {
            "bullish_hits": bullish_hits,
            "bearish_hits": bearish_hits,
            "no_entry_hits": no_entry_hits,
            "strongly_bearish": strongly_bearish,
            "bullish_supported": bullish_supported,
        }

    @staticmethod
    def _is_fallback_origin(decision: ResearchDecision) -> tuple[bool, str | None]:
        parser_mode = decision.source_metadata.parser_mode.lower()
        extra = decision.source_metadata.extra
        fallback_mode = str(extra.get("upstream_fallback_mode", "")).lower()
        upstream_failure_type = str(extra.get("upstream_failure_type", "")).strip()
        if parser_mode == "upstream_error_no_entry":
            return True, "parser_mode_upstream_error_no_entry"
        if fallback_mode and fallback_mode != "none":
            return True, f"upstream_fallback_mode={fallback_mode}"
        if fallback_mode in {"research_error_no_entry", "upstream_error_no_entry"}:
            return True, f"upstream_fallback_mode={fallback_mode}"
        notes_joined = " ".join(decision.source_metadata.notes).lower()
        if "structured parser fallback applied" in notes_joined:
            return True, "source_notes_parser_fallback"
        if upstream_failure_type in {"ResourceExhausted", "InvalidArgument"} and parser_mode != "llm_json":
            return True, f"upstream_failure_type={upstream_failure_type}_without_llm_json"
        for flag in decision.risk_flags:
            lowered = flag.lower()
            if lowered.startswith("research_error:"):
                return True, f"risk_flag:{flag}"
            if lowered in {"upstream_graph_failure", "insufficient_research_confidence"}:
                return True, f"risk_flag:{flag}"
        thesis_lowered = decision.thesis.lower()
        if "insufficient research confidence" in thesis_lowered:
            return True, "thesis_insufficient_research_confidence"
        if thesis_lowered.strip().startswith("research adapter fallback"):
            return True, "thesis_research_adapter_fallback"
        return False, None

    @staticmethod
    def _find_buy_inconsistent_phrases(thesis: str) -> list[str]:
        normalized = re.sub(r"\s+", " ", (thesis or "").lower()).strip()
        return [phrase for phrase in _BUY_INCONSISTENT_PHRASES if phrase in normalized]

    def _is_buy_thesis_consistent(self, thesis: str) -> tuple[bool, list[str]]:
        inconsistent_phrases = self._find_buy_inconsistent_phrases(thesis)
        semantics = self._thesis_semantics(thesis)
        bullish_supported = bool(semantics.get("bullish_supported", False))
        if inconsistent_phrases or not bullish_supported:
            reasons = [f"inconsistent_phrase:{phrase}" for phrase in inconsistent_phrases]
            if not bullish_supported:
                reasons.append("bullish_support_insufficient")
            return False, reasons
        return True, []

    def _buy_promotion_gate(
        self,
        decision: ResearchDecision,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
        technical_state: dict[str, Any],
        debate: DebateSummary,
    ) -> tuple[bool, str]:
        is_fallback_origin, _ = self._is_fallback_origin(decision)
        if is_fallback_origin:
            return False, "fallback_origin_non_promotable"
        if any(flag.lower() == "insufficient_research_confidence" for flag in decision.risk_flags):
            return False, "insufficient_research_confidence"
        entry_confidence_min = self.settings.data.entry_confidence_min
        if self._is_risk_on_contained(regime):
            entry_confidence_min = max(0.50, entry_confidence_min - self.settings.data.risk_on_entry_confidence_relief)
        if decision.confidence < entry_confidence_min:
            return False, "decision_confidence_below_threshold"
        if debate.confidence_balance < 0.58:
            return False, "debate_balance_below_threshold"
        inconsistent_phrases = self._find_buy_inconsistent_phrases(decision.thesis)
        if inconsistent_phrases:
            return False, f"thesis_inconsistent:{inconsistent_phrases[0]}"
        if "avoid" in decision.thesis.lower() or "no entry" in decision.thesis.lower():
            return False, "thesis_contains_no_entry_language"
        entry_ok, entry_reason = self._entry_gate_satisfied(candidate, regime, technical_memo, technical_state)
        if not entry_ok:
            return False, entry_reason
        return True, entry_reason

    @staticmethod
    def _fallback_label(decision: ResearchDecision) -> str:
        extra = decision.source_metadata.extra
        failure_type = extra.get("upstream_failure_type")
        if isinstance(failure_type, str) and failure_type:
            return failure_type
        failure_counts = extra.get("upstream_failure_counts")
        if isinstance(failure_counts, dict) and failure_counts:
            failure_type = next(iter(failure_counts.keys()))
            if isinstance(failure_type, str):
                return failure_type
        return "upstream_failure"

    @staticmethod
    def _current_unrealized_return(current_position: PositionSnapshot | None) -> float:
        if current_position is None or current_position.quantity <= 0 or current_position.avg_cost <= 0:
            return 0.0
        return (current_position.market_price / current_position.avg_cost) - 1.0

    @staticmethod
    def _is_risk_on_contained(regime: RegimeSnapshot | None) -> bool:
        if regime is None or regime.label.value != "risk_on":
            return False
        return regime.volatility_regime in {"contained", "normal", "low"}

    def _inventory_lifecycle_overlay(
        self,
        *,
        current_position: PositionSnapshot | None,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_state: dict[str, Any],
        holding_days: int | None,
    ) -> dict[str, Any] | None:
        if current_position is None or current_position.quantity <= 0:
            return None
        close = float(technical_state.get("close", 0.0))
        sma20 = float(technical_state.get("sma20", 0.0))
        sma50 = float(technical_state.get("sma50", 0.0))
        extension = float(technical_state.get("extension_over_ma20", 0.0))
        rsi14 = float(technical_state.get("rsi14", 50.0))
        trend_alignment = bool(technical_state.get("trend_alignment", False))
        unrealized_return = self._current_unrealized_return(current_position)
        rel_strength = 0.0 if candidate is None else candidate.relative_strength_20d
        regime_label = None if regime is None else regime.label.value
        risk_on_contained = self._is_risk_on_contained(regime)
        trim_trigger = (
            self.settings.risk.risk_on_trim_profit_trigger_fraction
            if risk_on_contained
            else self.settings.risk.trim_profit_trigger_fraction
        )
        reduce_trigger = (
            self.settings.risk.risk_on_reduce_to_core_profit_trigger_fraction
            if risk_on_contained
            else self.settings.risk.reduce_to_core_profit_trigger_fraction
        )

        trend_failure = (
            close < (sma20 * (1.0 - self.settings.risk.trend_failure_ma_slack_fraction))
            and (sma20 < sma50 or rel_strength <= self.settings.risk.trend_failure_relative_strength_floor)
        )
        if trend_failure:
            if regime_label == "risk_on" and unrealized_return > self.settings.risk.trim_profit_trigger_fraction:
                return {
                    "action": TradeAction.SELL,
                    "lifecycle": OrderIntentType.TRIM_PARTIAL,
                    "reason": "trend_failure_trim_winner",
                    "target_fraction": None,
                    "exit_type": "trend_failure_exit",
                }
            return {
                "action": TradeAction.SELL,
                "lifecycle": OrderIntentType.EXIT,
                "reason": "trend_failure_exit",
                "target_fraction": 0.0,
                "exit_type": "trend_failure_exit",
            }

        if (
            holding_days is not None
            and holding_days >= self.settings.risk.time_stop_days
            and unrealized_return < self.settings.risk.time_stop_min_progress_fraction
        ):
            if risk_on_contained and trend_alignment and unrealized_return >= 0:
                return {
                    "action": TradeAction.SELL,
                    "lifecycle": OrderIntentType.TRIM_PARTIAL,
                    "reason": "time_stop_trim_stalled_position",
                    "target_fraction": None,
                    "exit_type": "time_stop_exit",
                }
            return {
                "action": TradeAction.SELL,
                "lifecycle": OrderIntentType.EXIT,
                "reason": "time_stop_exit",
                "target_fraction": 0.0,
                "exit_type": "time_stop_exit",
            }

        if regime_label in {"risk_off", "high_volatility"}:
            if unrealized_return >= self.settings.risk.trim_profit_trigger_fraction and trend_alignment:
                return {
                    "action": TradeAction.SELL,
                    "lifecycle": OrderIntentType.TRIM_PARTIAL,
                    "reason": "regime_de_risk_trim",
                    "target_fraction": None,
                    "exit_type": "regime_exit",
                }
            if close < sma20:
                return {
                    "action": TradeAction.SELL,
                    "lifecycle": OrderIntentType.EXIT,
                    "reason": "regime_de_risk_exit",
                    "target_fraction": 0.0,
                    "exit_type": "regime_exit",
                }

        if (
            unrealized_return >= reduce_trigger
            and (
                extension >= (self.settings.data.max_extension_over_ma20_fraction * (0.85 if risk_on_contained else 1.0))
                or rsi14 >= self.settings.data.overheat_rsi_threshold
            )
        ):
            return {
                "action": TradeAction.SELL,
                "lifecycle": OrderIntentType.REDUCE_TO_CORE,
                "reason": "take_profit_reduce_to_core",
                "target_fraction": self.settings.risk.reduce_to_core_target_fraction,
                "exit_type": "take_profit_reduce_to_core",
            }
        if unrealized_return >= trim_trigger and (
            extension >= (self.settings.data.max_extension_over_ma20_fraction * (0.65 if risk_on_contained else 1.0))
            or rsi14 >= (self.settings.data.overheat_rsi_threshold - (4.0 if risk_on_contained else 0.0))
        ):
            return {
                "action": TradeAction.SELL,
                "lifecycle": OrderIntentType.TRIM_PARTIAL,
                "reason": "take_profit_trim_partial",
                "target_fraction": None,
                "exit_type": "take_profit_trim_partial",
            }
        return None

    def _synthesize_final_thesis(
        self,
        *,
        final_action: TradeAction,
        decision: ResearchDecision,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
        has_inventory: bool,
        fallback_origin: bool,
    ) -> str:
        if final_action == TradeAction.BUY:
            entry_mode = decision.entry_mode.value if hasattr(decision, "entry_mode") else "none"
            entry_reason = decision.entry_trigger_reason or "entry_confirmation_passed"
            extension = decision.extension_metrics.get("extension_over_ma20", 0.0)
            rsi = decision.extension_metrics.get("rsi14", 50.0)
            lifecycle_state = decision.position_lifecycle_state
            starter_entry = lifecycle_state in {OrderIntentType.STARTER_ENTRY, OrderIntentType.STARTER_ADD}
            regime_text = "Regime context is favorable for selective long entries."
            if regime is not None:
                regime_text = (
                    f"Regime is {regime.label.value} with risk budget multiplier "
                    f"{regime.risk_budget_multiplier:.2f}, supporting measured long exposure."
                )
            candidate_text = "Liquidity and ranking pre-screens passed."
            if candidate is not None:
                candidate_text = (
                    f"{candidate.symbol} remains eligible with ADV20 ${candidate.avg_dollar_volume_20d:,.0f}, "
                    f"20d return {candidate.return_20d:.2%}, and relative strength {candidate.relative_strength_20d:.2%}."
                )
            intro = (
                "Starter entry rationale: initiate a small long position because this is a validated "
                "risk-on near-miss setup, not a full-size entry. "
                if starter_entry
                else "Entry rationale: initiate a long position because cross-checks are supportive now. "
            )
            return (
                intro +
                f"{regime_text} {candidate_text} "
                f"Entry mode={entry_mode} ({entry_reason}), extension_over_ma20={extension:.2%}, RSI14={rsi:.1f}. "
                f"Technical evidence: {technical_memo.summary} "
                f"Decision confidence is {decision.confidence:.2f}; position sizing remains risk-constrained."
            )
        if final_action == TradeAction.AVOID:
            if fallback_origin:
                return (
                    "No-entry rationale: upstream research execution failed, so this signal is marked "
                    "insufficient research confidence and cannot be traded."
                    f" Failure type: {self._fallback_label(decision)}."
                )
            if has_inventory:
                return (
                    "No-entry rationale: evidence does not justify adding to the existing long, "
                    "so the position is held unchanged instead of increasing exposure."
                )
            return (
                "No-entry rationale: current evidence does not justify initiating a new long position "
                "under long-only constraints."
            )
        if final_action == TradeAction.SELL:
            lifecycle_state = decision.position_lifecycle_state
            lifecycle_label = "exit" if lifecycle_state is None else lifecycle_state.value
            exit_type = str(decision.source_metadata.extra.get("exit_type", "risk_reduction"))
            if lifecycle_state in {OrderIntentType.TRIM_PARTIAL, OrderIntentType.TRIM, OrderIntentType.SCALE_OUT}:
                return (
                    "Partial trim rationale: existing long inventory is profitable or heated enough to reduce risk, "
                    "but the favorable regime does not justify abandoning the position. "
                    f"Lifecycle={lifecycle_label}; exit_type={exit_type}; remaining exposure preserves upside participation."
                )
            if lifecycle_state == OrderIntentType.REDUCE_TO_CORE:
                return (
                    "Reduce-to-core rationale: position risk/reward is compressed after gains or extension, "
                    "so exposure is cut to a core allocation rather than fully exited. "
                    f"Lifecycle={lifecycle_label}; exit_type={exit_type}."
                )
            return (
                "Full exit rationale: existing long inventory is being closed because measurable exit conditions "
                "or portfolio constraints now outweigh continued participation. "
                f"Lifecycle={lifecycle_label}; exit_type={exit_type}."
            )
        if has_inventory:
            return "Hold rationale: maintain the existing long position while waiting for clearer directional evidence."
        return "Hold rationale: remain flat because the setup is not sufficiently compelling for a new long entry."

    def _preferred_action_from_debate(self, winning_side: str, has_inventory: bool) -> TradeAction:
        if winning_side == "bull":
            return TradeAction.BUY
        if winning_side == "bear":
            return TradeAction.SELL if has_inventory else TradeAction.AVOID
        return TradeAction.HOLD if has_inventory else TradeAction.AVOID

    def _adjudicate_long_only_action(
        self,
        decision: ResearchDecision,
        debate: DebateSummary,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
        current_position: PositionSnapshot | None,
        technical_state: dict[str, Any],
        position_holding_days: int | None = None,
    ) -> tuple[ResearchDecision, DebateSummary]:
        has_inventory = self._has_inventory(current_position)
        original_action = decision.action
        final_action = decision.action
        override_reason: str | None = None
        override_reasons: list[str] = []
        final_action_changed = False
        promoted_buy = False
        promoted_buy_from_debate = False
        buy_blocked_due_to_fallback = False
        buy_blocked_due_to_thesis_inconsistency = False
        action_thesis_mismatch_detected = False
        consistency_enforcement_changed_action = False
        buy_rewrite_attempted = False
        buy_rewrite_success = False
        buy_rewrite_failure = False
        final_action_downgraded = False
        inconsistent_buy_prevented = False
        buy_blocked_due_to_extension = False
        buy_blocked_due_to_overheat = False
        buy_blocked_due_to_missing_pullback_confirmation = False
        buy_blocked_due_to_missing_breakout_confirmation = False
        buy_near_miss_due_to_breakout_confirmation = False
        buy_near_miss_due_to_pullback_confirmation = False
        risk_on_participation_bias_applied = False
        starter_entry = False
        starter_entry_due_to_risk_on_bias = False
        starter_entry_rejected = False
        near_miss_promoted = False
        near_miss_not_promoted = False
        hard_reject = False
        soft_reject = False
        entry_reject_class = "none"
        entry_reject_reasons: list[str] = []
        hold_existing = False
        full_exit_due_to_risk_reduction = False
        full_exit_rejected_in_favor_of_trim = False
        full_exit_rejected_in_favor_of_reduce_to_core = False
        starter_position_kept_due_to_regime = False
        lifecycle_overlay_applied = False
        lifecycle_state: OrderIntentType | None = decision.position_lifecycle_state
        lifecycle_target_fraction: float | None = None
        starter_position_fraction: float | None = None

        fallback_origin, fallback_reason = self._is_fallback_origin(decision)
        entry_mode, entry_reason, entry_blockers, extension_penalty, overheat_penalty = self._entry_mode_and_gates(
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            technical_state=technical_state,
        )
        if "extension_overheat_block" in entry_blockers:
            buy_blocked_due_to_extension = True
        if "rsi_overheat_block" in entry_blockers:
            buy_blocked_due_to_overheat = True
        if "missing_pullback_confirmation" in entry_blockers:
            buy_blocked_due_to_missing_pullback_confirmation = True
        if "missing_breakout_confirmation" in entry_blockers:
            buy_blocked_due_to_missing_breakout_confirmation = True
        if "near_miss_breakout_confirmation" in entry_blockers:
            buy_near_miss_due_to_breakout_confirmation = True
        if "near_miss_pullback_confirmation" in entry_blockers:
            buy_near_miss_due_to_pullback_confirmation = True
        buy_near_miss = buy_near_miss_due_to_breakout_confirmation or buy_near_miss_due_to_pullback_confirmation
        thesis_semantics = self._thesis_semantics(decision.thesis)
        entry_reject_class, entry_reject_reasons, starter_eligible_near_miss = self._classify_entry_reject(
            decision=decision,
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            debate=debate,
            entry_mode=entry_mode,
            entry_blockers=entry_blockers,
            extension_penalty=extension_penalty,
            overheat_penalty=overheat_penalty,
            fallback_origin=fallback_origin,
            thesis_semantics=thesis_semantics,
        )
        hard_reject = entry_reject_class == "hard_reject"
        soft_reject = entry_reject_class in {"soft_reject", "starter_eligible_near_miss"}

        inventory_overlay = self._inventory_lifecycle_overlay(
            current_position=current_position,
            candidate=candidate,
            regime=regime,
            technical_state=technical_state,
            holding_days=position_holding_days,
        )
        if inventory_overlay is not None:
            lifecycle_overlay_applied = True
            lifecycle_state = inventory_overlay["lifecycle"]
            lifecycle_target_fraction = inventory_overlay["target_fraction"]
            if inventory_overlay["action"] == TradeAction.SELL and final_action != TradeAction.SELL:
                final_action = TradeAction.SELL
                final_action_changed = True
                override_reasons.append(f"inventory_lifecycle_overlay:{inventory_overlay['reason']}")
            if original_action == TradeAction.SELL and self._is_risk_on_contained(regime):
                if lifecycle_state in {OrderIntentType.TRIM_PARTIAL, OrderIntentType.TRIM, OrderIntentType.SCALE_OUT}:
                    full_exit_rejected_in_favor_of_trim = True
                    override_reasons.append("risk_on_full_exit_trimmed_instead")
                if lifecycle_state == OrderIntentType.REDUCE_TO_CORE:
                    full_exit_rejected_in_favor_of_reduce_to_core = True
                    override_reasons.append("risk_on_full_exit_reduced_to_core")

        if final_action == TradeAction.SELL and not has_inventory:
            final_action = TradeAction.AVOID
            final_action_changed = True
            override_reasons.append("sell_recast_to_avoid_no_inventory")

        if (
            debate.winning_side == "bull"
            and final_action in {TradeAction.SELL, TradeAction.HOLD, TradeAction.AVOID}
            and not has_inventory
        ):
            entry_ok, entry_reason = self._buy_promotion_gate(
                decision=decision,
                candidate=candidate,
                regime=regime,
                technical_memo=technical_memo,
                technical_state=technical_state,
                debate=debate,
            )
            if entry_ok:
                final_action = TradeAction.BUY
                final_action_changed = final_action != original_action or final_action_changed
                promoted_buy = True
                promoted_buy_from_debate = True
                override_reasons.append(f"debate_bull_entry_gate_passed:{entry_reason}")
            else:
                override_reasons.append(f"debate_bull_override_rejected:{entry_reason}")
                if entry_reason == "fallback_origin_non_promotable":
                    buy_blocked_due_to_fallback = True
                    inconsistent_buy_prevented = True
                if entry_reason.startswith("thesis_inconsistent:"):
                    buy_blocked_due_to_thesis_inconsistency = True
                    inconsistent_buy_prevented = True
                if entry_reason == "insufficient_research_confidence":
                    buy_blocked_due_to_fallback = True
                    inconsistent_buy_prevented = True
                if entry_reason in {"extension_overheat_block", "max_extension_exceeded"}:
                    buy_blocked_due_to_extension = True
                if entry_reason == "rsi_overheat_block":
                    buy_blocked_due_to_overheat = True
                if entry_reason == "missing_pullback_confirmation":
                    buy_blocked_due_to_missing_pullback_confirmation = True
                if entry_reason == "missing_breakout_confirmation":
                    buy_blocked_due_to_missing_breakout_confirmation = True

        if final_action in {TradeAction.HOLD, TradeAction.AVOID} and not has_inventory:
            participation_semantics = self._thesis_semantics(decision.thesis)
            confidence_floor = max(
                0.46,
                self.settings.data.entry_confidence_min - self.settings.data.risk_on_near_miss_confidence_relief,
            )
            participation_bias_allowed = (
                self._is_risk_on_contained(regime)
                and (entry_mode != EntryMode.NONE or starter_eligible_near_miss)
                and not fallback_origin
                and technical_memo.signal == "bullish"
                and decision.confidence >= (confidence_floor if starter_eligible_near_miss else max(0.50, self.settings.data.entry_confidence_min - 0.08))
                and technical_memo.confidence >= confidence_floor
                and not bool(participation_semantics.get("strongly_bearish", False))
                and not hard_reject
                and overheat_penalty < self.settings.data.risk_on_near_miss_max_overheat_penalty
                and extension_penalty < self.settings.data.risk_on_near_miss_max_extension_penalty
                and (debate.winning_side != "bear" or debate.confidence_balance < (0.70 if starter_eligible_near_miss else 0.64))
                and candidate is not None
                and candidate.eligible
                and not candidate.watchlist_only
                and candidate.relative_strength_20d >= (
                    -0.005 if starter_eligible_near_miss else self.settings.data.risk_on_min_relative_strength_20d
                )
                and candidate.return_20d >= (self.settings.data.risk_on_min_strong_trend_return_20d * 0.75)
                and candidate.avg_dollar_volume_20d >= self.settings.data.min_avg_dollar_volume
            )
            if participation_bias_allowed:
                final_action = TradeAction.BUY
                final_action_changed = True
                promoted_buy = True
                risk_on_participation_bias_applied = True
                if starter_eligible_near_miss:
                    starter_entry = True
                    starter_entry_due_to_risk_on_bias = True
                    near_miss_promoted = True
                    lifecycle_state = OrderIntentType.STARTER_ENTRY
                    starter_position_fraction = self._starter_position_fraction(
                        decision=decision,
                        candidate=candidate,
                        extension_penalty=extension_penalty,
                        overheat_penalty=overheat_penalty,
                        entry_blockers=entry_blockers,
                    )
                    entry_reason = (
                        "risk_on_starter_near_miss_breakout"
                        if buy_near_miss_due_to_breakout_confirmation
                        else "risk_on_starter_near_miss_pullback"
                    )
                override_reasons.append(f"risk_on_participation_bias:{entry_reason}")

        if (
            final_action == TradeAction.BUY
            and entry_mode == EntryMode.NONE
            and starter_eligible_near_miss
            and not starter_entry_due_to_risk_on_bias
        ):
            starter_entry = True
            starter_entry_due_to_risk_on_bias = True
            near_miss_promoted = True
            risk_on_participation_bias_applied = True
            lifecycle_state = OrderIntentType.STARTER_ENTRY
            starter_position_fraction = self._starter_position_fraction(
                decision=decision,
                candidate=candidate,
                extension_penalty=extension_penalty,
                overheat_penalty=overheat_penalty,
                entry_blockers=entry_blockers,
            )
            entry_reason = (
                "risk_on_starter_near_miss_breakout"
                if buy_near_miss_due_to_breakout_confirmation
                else "risk_on_starter_near_miss_pullback"
            )
            override_reasons.append(f"risk_on_starter_entry_sized:{entry_reason}")

        if final_action == TradeAction.BUY and fallback_origin:
            final_action = TradeAction.HOLD if has_inventory else TradeAction.AVOID
            final_action_changed = True
            consistency_enforcement_changed_action = True
            final_action_downgraded = True
            buy_blocked_due_to_fallback = True
            action_thesis_mismatch_detected = True
            inconsistent_buy_prevented = True
            override_reasons.append(f"buy_blocked_fallback_origin:{fallback_reason or 'unknown'}")

        if final_action == TradeAction.BUY and entry_mode == EntryMode.NONE and not starter_entry_due_to_risk_on_bias:
            final_action = TradeAction.HOLD if has_inventory else TradeAction.AVOID
            final_action_changed = True
            consistency_enforcement_changed_action = True
            final_action_downgraded = True
            inconsistent_buy_prevented = True
            if buy_blocked_due_to_extension:
                override_reasons.append("buy_blocked_extension")
            if buy_blocked_due_to_overheat:
                override_reasons.append("buy_blocked_overheat")
            if buy_blocked_due_to_missing_pullback_confirmation:
                override_reasons.append("buy_blocked_missing_pullback_confirmation")
            if buy_blocked_due_to_missing_breakout_confirmation:
                override_reasons.append("buy_blocked_missing_breakout_confirmation")
            if not (
                buy_blocked_due_to_extension
                or buy_blocked_due_to_overheat
                or buy_blocked_due_to_missing_pullback_confirmation
                or buy_blocked_due_to_missing_breakout_confirmation
            ):
                override_reasons.append(f"buy_blocked_entry_gate:{entry_reason}")

        if final_action == TradeAction.BUY:
            buy_rewrite_attempted = True
            draft_buy_decision = decision.model_copy(
                update={
                    "entry_mode": entry_mode,
                    "entry_trigger_reason": entry_reason,
                    "extension_penalty": extension_penalty,
                    "overheat_penalty": overheat_penalty,
                    "extension_metrics": {
                        "extension_over_ma20": float(technical_state.get("extension_over_ma20", 0.0)),
                        "extension_over_ma50": float(technical_state.get("extension_over_ma50", 0.0)),
                        "rsi14": float(technical_state.get("rsi14", 50.0)),
                        "breakout_distance": float(technical_state.get("breakout_distance", 0.0)),
                        "pullback_distance_ma20": float(technical_state.get("pullback_distance_ma20", 1.0)),
                    },
                }
            )
            buy_thesis_candidate = self._synthesize_final_thesis(
                final_action=TradeAction.BUY,
                decision=draft_buy_decision,
                candidate=candidate,
                regime=regime,
                technical_memo=technical_memo,
                has_inventory=has_inventory,
                fallback_origin=fallback_origin,
            )
            buy_consistent, buy_inconsistency_reasons = self._is_buy_thesis_consistent(buy_thesis_candidate)
            if buy_consistent:
                buy_rewrite_success = True
            else:
                buy_rewrite_failure = True
                final_action = TradeAction.HOLD if has_inventory else TradeAction.AVOID
                final_action_changed = True
                consistency_enforcement_changed_action = True
                final_action_downgraded = True
                buy_blocked_due_to_thesis_inconsistency = True
                if starter_entry_due_to_risk_on_bias:
                    starter_entry_rejected = True
                    starter_entry = False
                action_thesis_mismatch_detected = True
                inconsistent_buy_prevented = True
                override_reasons.append(
                    "buy_rewrite_failed:"
                    + ",".join(dict.fromkeys(reason.replace("inconsistent_phrase:", "") for reason in buy_inconsistency_reasons))
                )

        if final_action == TradeAction.SELL and not has_inventory:
            final_action = TradeAction.AVOID
            final_action_changed = True
            override_reasons.append("sell_recast_to_avoid_no_inventory")

        if (
            final_action == TradeAction.SELL
            and has_inventory
            and self._is_risk_on_contained(regime)
            and lifecycle_state in {None, OrderIntentType.EXIT}
        ):
            close = float(technical_state.get("close", 0.0))
            sma20 = float(technical_state.get("sma20", 0.0))
            sma50 = float(technical_state.get("sma50", 0.0))
            extension = float(technical_state.get("extension_over_ma20", 0.0))
            rsi14 = float(technical_state.get("rsi14", 50.0))
            rel_strength = 0.0 if candidate is None else candidate.relative_strength_20d
            unrealized_return = self._current_unrealized_return(current_position)
            explicit_exit_type = None if inventory_overlay is None else inventory_overlay.get("exit_type")
            severe_breakdown = (
                close < sma20
                and (sma20 < sma50 or rel_strength <= self.settings.risk.trend_failure_relative_strength_floor)
            )
            severe_loss_or_heat = unrealized_return <= -0.04 or (
                extension >= self.settings.data.overheat_extension_fraction
                and rsi14 >= (self.settings.data.overheat_rsi_threshold + 3.0)
            )
            explicit_hard_exit = explicit_exit_type in {"trend_failure_exit", "regime_exit"}
            if not (severe_breakdown or severe_loss_or_heat or explicit_hard_exit):
                if (
                    unrealized_return >= self.settings.risk.risk_on_reduce_to_core_profit_trigger_fraction
                    or extension >= (self.settings.data.max_extension_over_ma20_fraction * 0.95)
                ):
                    lifecycle_state = OrderIntentType.REDUCE_TO_CORE
                    lifecycle_target_fraction = self.settings.risk.reduce_to_core_target_fraction
                    full_exit_rejected_in_favor_of_reduce_to_core = True
                    override_reasons.append("risk_on_full_exit_reduced_to_core")
                elif (
                    unrealized_return >= self.settings.risk.risk_on_trim_profit_trigger_fraction
                    or extension >= (self.settings.data.max_extension_over_ma20_fraction * 0.65)
                    or rsi14 >= (self.settings.data.overheat_rsi_threshold - 4.0)
                ):
                    lifecycle_state = OrderIntentType.TRIM_PARTIAL
                    lifecycle_target_fraction = None
                    full_exit_rejected_in_favor_of_trim = True
                    override_reasons.append("risk_on_full_exit_trimmed_instead")
                else:
                    final_action = TradeAction.HOLD
                    lifecycle_state = None
                    lifecycle_target_fraction = None
                    final_action_changed = True
                    starter_position_kept_due_to_regime = True
                    lifecycle_state = OrderIntentType.STARTER_KEEP
                    override_reasons.append("risk_on_full_exit_rejected_weak_justification")

        if final_action == TradeAction.HOLD and has_inventory:
            hold_existing = True
            if starter_position_kept_due_to_regime and lifecycle_state is None:
                lifecycle_state = OrderIntentType.STARTER_KEEP

        preferred_action = self._preferred_action_from_debate(debate.winning_side, has_inventory)
        aligned = final_action == preferred_action
        if not aligned:
            override_reasons.append(
                f"debate_{debate.winning_side}_preferred_{preferred_action.value}_but_final_{final_action.value}"
            )

        if override_reasons:
            override_reason = ";".join(dict.fromkeys(override_reasons))

        desired_position_fraction = decision.desired_position_fraction
        if final_action != TradeAction.BUY:
            if final_action == TradeAction.SELL and lifecycle_state == OrderIntentType.REDUCE_TO_CORE:
                desired_position_fraction = self.settings.risk.reduce_to_core_target_fraction
            elif lifecycle_target_fraction is not None:
                desired_position_fraction = lifecycle_target_fraction
            else:
                desired_position_fraction = 0.0
        elif desired_position_fraction is None or desired_position_fraction <= 0:
            if starter_position_fraction is not None:
                desired_position_fraction = min(starter_position_fraction, self.settings.risk.max_position_size_fraction)
            elif risk_on_participation_bias_applied:
                desired_position_fraction = min(
                    self.settings.risk.risk_on_starter_position_fraction,
                    self.settings.risk.max_position_size_fraction,
                )
            else:
                desired_position_fraction = min(0.04, self.settings.risk.max_position_size_fraction)
        if final_action == TradeAction.BUY:
            if entry_mode == EntryMode.BREAKOUT and regime is not None and regime.label.value == "balanced":
                desired_position_fraction *= 0.75
            if entry_mode == EntryMode.PULLBACK and regime is not None and regime.label.value == "risk_on":
                desired_position_fraction *= 1.05
            penalty_scale = _clamp(1.0 - (0.35 * extension_penalty) - (0.45 * overheat_penalty), 0.35, 1.0)
            desired_position_fraction *= penalty_scale
            desired_position_fraction = _clamp(desired_position_fraction, 0.0, self.settings.risk.max_position_size_fraction)

        pre_synthesis_exit_type: str | None = None
        if final_action == TradeAction.SELL:
            if full_exit_rejected_in_favor_of_reduce_to_core:
                pre_synthesis_exit_type = "risk_compression_reduce_to_core"
            elif full_exit_rejected_in_favor_of_trim:
                pre_synthesis_exit_type = "risk_compression_trim_partial"
            elif inventory_overlay is not None:
                pre_synthesis_exit_type = str(inventory_overlay.get("exit_type") or "inventory_reduction")
            else:
                pre_synthesis_exit_type = "risk_reduction"
        synthesis_source_extra = dict(decision.source_metadata.extra)
        synthesis_source_extra["exit_type"] = pre_synthesis_exit_type
        synthesis_decision = decision.model_copy(
            update={
                "entry_mode": entry_mode,
                "entry_trigger_reason": entry_reason,
                "extension_penalty": extension_penalty,
                "overheat_penalty": overheat_penalty,
                "extension_metrics": {
                    "extension_over_ma20": float(technical_state.get("extension_over_ma20", 0.0)),
                    "extension_over_ma50": float(technical_state.get("extension_over_ma50", 0.0)),
                    "rsi14": float(technical_state.get("rsi14", 50.0)),
                    "breakout_distance": float(technical_state.get("breakout_distance", 0.0)),
                    "pullback_distance_ma20": float(technical_state.get("pullback_distance_ma20", 1.0)),
                },
                "position_lifecycle_state": lifecycle_state,
                "source_metadata": decision.source_metadata.model_copy(update={"extra": synthesis_source_extra}),
            }
        )
        synthesized_thesis = self._synthesize_final_thesis(
            final_action=final_action,
            decision=synthesis_decision,
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            has_inventory=has_inventory,
            fallback_origin=fallback_origin,
        )
        thesis_rewritten = synthesized_thesis.strip() != decision.thesis.strip()

        resolved_exit_type: str | None = None
        if final_action == TradeAction.SELL:
            resolved_exit_type = pre_synthesis_exit_type
            full_exit_due_to_risk_reduction = (
                resolved_exit_type == "risk_reduction" and lifecycle_state in {None, OrderIntentType.EXIT}
            )
        if buy_near_miss and not near_miss_promoted:
            near_miss_not_promoted = True
        if starter_eligible_near_miss and not starter_entry:
            starter_entry_rejected = True

        updated_source_extra = dict(decision.source_metadata.extra)
        updated_source_extra.update(
            {
                "fallback_origin": fallback_origin,
                "fallback_origin_reason": fallback_reason,
                "buy_promotion_applied": promoted_buy,
                "buy_promotion_source": (
                    "debate_bull"
                    if promoted_buy_from_debate
                    else ("risk_on_participation_bias" if risk_on_participation_bias_applied else None)
                ),
                "buy_blocked_due_to_fallback": buy_blocked_due_to_fallback,
                "buy_blocked_due_to_thesis_inconsistency": buy_blocked_due_to_thesis_inconsistency,
                "action_thesis_mismatch_detected": action_thesis_mismatch_detected,
                "final_action_changed": final_action_changed or (original_action != final_action),
                "final_action_changed_after_consistency_enforcement": consistency_enforcement_changed_action,
                "fallback_buy_blocked": buy_blocked_due_to_fallback,
                "thesis_inconsistency_blocked": buy_blocked_due_to_thesis_inconsistency,
                "buy_rewrite_attempted": buy_rewrite_attempted,
                "buy_rewrite_success": buy_rewrite_success,
                "buy_rewrite_failure": buy_rewrite_failure,
                "final_action_downgraded": final_action_downgraded,
                "inconsistent_buy_prevented": inconsistent_buy_prevented,
                "buy_blocked_due_to_extension": buy_blocked_due_to_extension,
                "buy_blocked_due_to_overheat": buy_blocked_due_to_overheat,
                "buy_blocked_due_to_missing_pullback_confirmation": buy_blocked_due_to_missing_pullback_confirmation,
                "buy_blocked_due_to_missing_breakout_confirmation": buy_blocked_due_to_missing_breakout_confirmation,
                "buy_near_miss_due_to_breakout_confirmation": buy_near_miss_due_to_breakout_confirmation,
                "buy_near_miss_due_to_pullback_confirmation": buy_near_miss_due_to_pullback_confirmation,
                "buy_near_miss": buy_near_miss,
                "risk_on_participation_bias_applied": risk_on_participation_bias_applied,
                "starter_entry": starter_entry,
                "starter_entry_due_to_risk_on_bias": starter_entry_due_to_risk_on_bias,
                "starter_entry_rejected": starter_entry_rejected,
                "near_miss_promoted": near_miss_promoted,
                "near_miss_not_promoted": near_miss_not_promoted,
                "entry_reject_class": entry_reject_class,
                "entry_reject_reasons": entry_reject_reasons,
                "hard_reject": hard_reject,
                "soft_reject": soft_reject,
                "hold_existing": hold_existing,
                "full_exit_due_to_risk_reduction": full_exit_due_to_risk_reduction,
                "full_exit_rejected_in_favor_of_trim": full_exit_rejected_in_favor_of_trim,
                "full_exit_rejected_in_favor_of_reduce_to_core": full_exit_rejected_in_favor_of_reduce_to_core,
                "starter_position_kept_due_to_regime": starter_position_kept_due_to_regime,
                "entry_mode": entry_mode.value,
                "entry_trigger_reason": entry_reason,
                "entry_blockers": entry_blockers,
                "entry_mode_confirmed": entry_mode != EntryMode.NONE,
                "extension_penalty": extension_penalty,
                "overheat_penalty": overheat_penalty,
                "extension_metrics": synthesis_decision.extension_metrics,
                "lifecycle_overlay_applied": lifecycle_overlay_applied,
                "position_lifecycle_state": None if lifecycle_state is None else lifecycle_state.value,
                "position_holding_days": position_holding_days,
                "exit_type": resolved_exit_type,
                "final_thesis_rewritten": thesis_rewritten,
                "thesis_semantics": thesis_semantics,
                "final_action": final_action.value,
            }
        )

        updated_risk_flags = list(
            dict.fromkeys(
                decision.risk_flags
                + ([f"action_override:{override_reason}"] if override_reason else [])
                + (["buy_blocked_due_to_fallback"] if buy_blocked_due_to_fallback else [])
                + (["buy_blocked_due_to_thesis_inconsistency"] if buy_blocked_due_to_thesis_inconsistency else [])
                + (["buy_blocked_due_to_extension"] if buy_blocked_due_to_extension else [])
                + (["buy_blocked_due_to_overheat"] if buy_blocked_due_to_overheat else [])
                + (
                    ["buy_blocked_due_to_missing_pullback_confirmation"]
                    if buy_blocked_due_to_missing_pullback_confirmation
                    else []
                )
                + (
                    ["buy_blocked_due_to_missing_breakout_confirmation"]
                    if buy_blocked_due_to_missing_breakout_confirmation
                    else []
                )
                + (["action_thesis_mismatch_detected"] if action_thesis_mismatch_detected else [])
            )
        )
        updated_decision = decision.model_copy(
            update={
                "action": final_action,
                "desired_position_fraction": desired_position_fraction,
                "entry_mode": entry_mode,
                "entry_trigger_reason": entry_reason,
                "extension_penalty": extension_penalty,
                "overheat_penalty": overheat_penalty,
                "extension_metrics": synthesis_decision.extension_metrics,
                "position_lifecycle_state": lifecycle_state,
                "risk_flags": updated_risk_flags,
                "thesis": synthesized_thesis,
                "source_metadata": decision.source_metadata.model_copy(update={"extra": updated_source_extra}),
            }
        )
        updated_key_points = [point for point in debate.key_points if not point.startswith("final_action=")]
        updated_key_points.append(f"final_action={final_action.value}")
        if override_reason:
            updated_key_points.append(f"override_reason={override_reason}")
        updated_debate = debate.model_copy(
            update={
                "final_action": final_action,
                "aligned_with_final_action": aligned,
                "override_reason": override_reason,
                "key_points": updated_key_points,
            }
        )
        return updated_decision, updated_debate

    def _bull_bear_and_debate(
        self,
        symbol: str,
        as_of_date: date,
        decision: ResearchDecision,
        memos: list[AnalystMemo],
    ) -> tuple[BullCaseMemo, BearCaseMemo, DebateSummary]:
        bullish = [memo for memo in memos if memo.signal == "bullish"]
        bearish = [memo for memo in memos if memo.signal == "bearish"]
        neutral = [memo for memo in memos if memo.signal in {"neutral", "mixed"}]
        bull_conviction = _clamp((sum(m.confidence for m in bullish) + (0.2 if decision.action == TradeAction.BUY else 0.0)) / max(1, len(memos)))
        bear_conviction = _clamp((sum(m.confidence for m in bearish) + (0.2 if decision.action == TradeAction.SELL else 0.0)) / max(1, len(memos)))
        bull_summary = (
            "Strongest long thesis emphasizes positive momentum, adequate liquidity, and supportive context signals."
            if bullish
            else "Long thesis is weak because supportive evidence is sparse."
        )
        bear_summary = (
            "Counter-thesis emphasizes downside catalysts, adverse event/macro risks, and valuation fragility."
            if bearish
            else "Counter-thesis is modest due to limited explicit downside evidence."
        )
        bull_case = BullCaseMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            summary=bull_summary,
            catalysts=[memo.role for memo in bullish[:4]],
            invalidation_conditions=[
                "20-day trend flips negative with weak breadth confirmation.",
                "Material negative event/earnings surprise emerges.",
            ],
            conviction=bull_conviction,
        )
        bear_case = BearCaseMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            summary=bear_summary,
            risks=[memo.role for memo in bearish[:4]],
            invalidation_conditions=[
                "Price sustains above medium-term trend with improving breadth.",
                "Macro risk proxies stabilize and volatility compresses.",
            ],
            conviction=bear_conviction,
        )
        if abs(bull_conviction - bear_conviction) < 0.08:
            winning = "draw"
            adjudication = "Bull and bear evidence is balanced; default to conservative sizing."
        elif bull_conviction > bear_conviction:
            winning = "bull"
            adjudication = "Bull case has broader cross-analyst support than the bear case."
        else:
            winning = "bear"
            adjudication = "Bear case risk signals dominate current evidence."
        debate = DebateSummary(
            symbol=symbol,
            as_of_date=as_of_date,
            adjudication=adjudication,
            winning_side=winning,
            confidence_balance=_clamp(abs(bull_conviction - bear_conviction) + 0.35, 0.35, 0.95),
            final_action=decision.action,
            aligned_with_final_action=True,
            override_reason=None,
            falsifiers=[
                "If trend and breadth diverge from the selected stance for 2 consecutive sessions.",
                "If event/news flow contradicts the current dominant thesis.",
            ],
            key_points=[
                f"bull_support_count={len(bullish)}",
                f"bear_support_count={len(bearish)}",
                f"neutral_support_count={len(neutral)}",
                f"final_action={decision.action.value}",
            ],
        )
        return bull_case, bear_case, debate

    def run(
        self,
        symbol: str,
        as_of_date: date,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        current_position: PositionSnapshot | None = None,
        position_holding_days: int | None = None,
    ) -> tuple[ResearchDecision, ResearchBundle]:
        decision = self.adapter.research(symbol, as_of_date)
        history = self.provider.get_history(symbol, as_of_date, self.settings.data.history_lookback_days)
        technical_state = self._technical_state(history)
        fundamentals = self.provider.get_fundamentals(symbol)
        news_items = self.provider.get_news(symbol, as_of_date, self.settings.data.max_news_items)

        universe_scout = AnalystMemo(
            symbol=symbol,
            as_of_date=as_of_date,
            role="Universe Scout",
            signal="neutral",
            confidence=0.6 if candidate and candidate.eligible else 0.3,
            summary=(
                "Candidate selected from ranked tradable universe."
                if candidate and candidate.eligible
                else "Candidate surfaced manually or with limited pre-screen evidence."
            ),
            evidence=[] if not candidate else [f"ranking_score={candidate.ranking_score:.3f}"],
            warnings=[] if candidate and candidate.eligible else ["screening_weak_evidence"],
        )
        regime_memo, macro_memo = self._regime_memos(symbol, as_of_date, regime)
        technical_memo = self._technical_memo(symbol, as_of_date, history)
        fundamental_memo = self._fundamental_memo(symbol, as_of_date, fundamentals)
        news_memo, sentiment_memo = self._news_memos(symbol, as_of_date, news_items)
        memos = [universe_scout, regime_memo, macro_memo, technical_memo, fundamental_memo, news_memo, sentiment_memo]
        bull_case, bear_case, debate = self._bull_bear_and_debate(symbol, as_of_date, decision, memos)
        decision, debate = self._adjudicate_long_only_action(
            decision=decision,
            debate=debate,
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            current_position=current_position,
            technical_state=technical_state,
            position_holding_days=position_holding_days,
        )
        memos.append(
            AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Bull Researcher",
                signal="bullish",
                confidence=bull_case.conviction,
                summary=bull_case.summary,
                evidence=bull_case.catalysts,
                warnings=[],
            )
        )
        memos.append(
            AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Bear Researcher",
                signal="bearish",
                confidence=bear_case.conviction,
                summary=bear_case.summary,
                evidence=bear_case.risks,
                warnings=[],
            )
        )
        memos.append(
            AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Debate / Adjudication Layer",
                signal="neutral" if debate.winning_side == "draw" else ("bullish" if debate.winning_side == "bull" else "bearish"),
                confidence=debate.confidence_balance,
                summary=debate.adjudication,
                evidence=debate.key_points,
                warnings=[],
            )
        )
        memos.append(
            AnalystMemo(
                symbol=symbol,
                as_of_date=as_of_date,
                role="Trader",
                signal=(
                    "bullish"
                    if decision.action == TradeAction.BUY
                    else ("bearish" if decision.action == TradeAction.SELL else "neutral")
                ),
                confidence=decision.confidence,
                summary=decision.thesis[:600],
                evidence=[
                    f"resolved_action={decision.action.value}",
                    f"time_horizon={decision.time_horizon}",
                    f"entry_mode={decision.entry_mode.value}",
                    (
                        "lifecycle_state=none"
                        if decision.position_lifecycle_state is None
                        else f"lifecycle_state={decision.position_lifecycle_state.value}"
                    ),
                ],
                warnings=[],
            )
        )

        bundle = ResearchBundle(
            symbol=symbol,
            as_of_date=as_of_date,
            candidate_id=None if candidate is None else candidate.candidate_id,
            regime_snapshot_id=None if regime is None else regime.regime_snapshot_id,
            analyst_memos=memos,
            bull_case=bull_case,
            bear_case=bear_case,
            debate_summary=debate,
            trader_note=(
                f"Trader recommendation: {decision.action.value.upper()} with confidence {decision.confidence:.2f}."
                if debate.override_reason is None
                else (
                    f"Trader recommendation: {decision.action.value.upper()} with confidence {decision.confidence:.2f}; "
                    f"override_reason={debate.override_reason}."
                )
            ),
            final_decision_id=decision.decision_id,
            warnings=[warning for memo in memos for warning in memo.warnings][:12],
        )
        return decision, bundle
