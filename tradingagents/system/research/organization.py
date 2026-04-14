from __future__ import annotations

from datetime import date

import pandas as pd

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import FundamentalSnapshot, MarketDataProvider, NewsItem
from tradingagents.system.schemas import (
    AnalystMemo,
    BearCaseMemo,
    BullCaseMemo,
    CandidateAssessment,
    DebateSummary,
    RegimeSnapshot,
    ResearchBundle,
    ResearchDecision,
    TradeAction,
)

from .adapter import ResearchAdapter


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


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
        if history.empty or len(history) < 30:
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

        close = float(history["Close"].iloc[-1])
        sma_20 = float(history["Close"].tail(20).mean())
        sma_50 = float(history["Close"].tail(50).mean()) if len(history) >= 50 else sma_20
        ret_20 = float(history["Close"].iloc[-1] / history["Close"].iloc[-21] - 1) if len(history) > 21 else 0.0
        ret_60 = float(history["Close"].iloc[-1] / history["Close"].iloc[-61] - 1) if len(history) > 61 else ret_20
        vol_20 = float(history["Close"].pct_change().tail(20).std() * (252**0.5))
        rolling_peak = float(history["Close"].tail(120).max())
        drawdown = 0.0 if rolling_peak <= 0 else (close / rolling_peak) - 1.0
        rsi = self._compute_rsi(history)
        if rsi is None:
            warnings.append("rsi_unavailable")
            rsi = 50.0

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
    ) -> tuple[ResearchDecision, ResearchBundle]:
        decision = self.adapter.research(symbol, as_of_date)
        history = self.provider.get_history(symbol, as_of_date, self.settings.data.history_lookback_days)
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
                signal="bullish" if decision.action == TradeAction.BUY else ("bearish" if decision.action == TradeAction.SELL else "neutral"),
                confidence=decision.confidence,
                summary=decision.thesis[:600],
                evidence=[f"upstream_action={decision.action.value}", f"time_horizon={decision.time_horizon}"],
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
            trader_note=f"Trader recommendation: {decision.action.value.upper()} with confidence {decision.confidence:.2f}.",
            final_decision_id=decision.decision_id,
            warnings=[warning for memo in memos for warning in memo.warnings][:12],
        )
        return decision, bundle
