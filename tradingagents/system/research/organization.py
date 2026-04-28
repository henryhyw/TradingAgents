from __future__ import annotations

from datetime import date
import re

import pandas as pd

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import FundamentalSnapshot, MarketDataProvider, NewsItem
from tradingagents.system.schemas import (
    AnalystMemo,
    BearCaseMemo,
    BullCaseMemo,
    CandidateAssessment,
    DebateSummary,
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

    @staticmethod
    def _has_inventory(current_position: PositionSnapshot | None) -> bool:
        return current_position is not None and current_position.quantity > 0

    def _entry_gate_satisfied(
        self,
        candidate: CandidateAssessment | None,
        regime: RegimeSnapshot | None,
        technical_memo: AnalystMemo,
    ) -> tuple[bool, str]:
        if candidate is not None and candidate.watchlist_only:
            return False, "watchlist_only_candidate"
        if candidate is not None and not candidate.eligible:
            return False, "candidate_not_eligible"
        if candidate is not None and candidate.avg_dollar_volume_20d < self.settings.data.min_avg_dollar_volume:
            return False, "liquidity_below_minimum"
        if regime is not None and regime.label.value not in {"risk_on", "balanced"}:
            return False, f"regime_{regime.label.value}_not_entry_favorable"
        if technical_memo.signal != "bullish":
            return False, "technical_signal_not_bullish"
        if technical_memo.confidence < 0.55:
            return False, "technical_confidence_too_low"
        if candidate is not None and candidate.relative_strength_20d <= 0:
            return False, "relative_strength_not_positive"
        if candidate is not None and candidate.return_20d <= 0:
            return False, "short_term_return_not_positive"
        return True, "entry_gate_passed"

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
        debate: DebateSummary,
    ) -> tuple[bool, str]:
        is_fallback_origin, _ = self._is_fallback_origin(decision)
        if is_fallback_origin:
            return False, "fallback_origin_non_promotable"
        if any(flag.lower() == "insufficient_research_confidence" for flag in decision.risk_flags):
            return False, "insufficient_research_confidence"
        if decision.confidence < 0.55:
            return False, "decision_confidence_below_threshold"
        if debate.confidence_balance < 0.58:
            return False, "debate_balance_below_threshold"
        inconsistent_phrases = self._find_buy_inconsistent_phrases(decision.thesis)
        if inconsistent_phrases:
            return False, f"thesis_inconsistent:{inconsistent_phrases[0]}"
        entry_ok, entry_reason = self._entry_gate_satisfied(candidate, regime, technical_memo)
        if not entry_ok:
            return False, entry_reason
        return True, "entry_gate_passed"

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
            return (
                "Entry rationale: initiate a long position because cross-checks are supportive now. "
                f"{regime_text} {candidate_text} "
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
            return (
                "Exit rationale: existing long inventory should be reduced or closed due to deteriorating "
                "risk/reward versus current portfolio constraints."
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

        fallback_origin, fallback_reason = self._is_fallback_origin(decision)

        if final_action == TradeAction.SELL and not has_inventory:
            final_action = TradeAction.AVOID
            final_action_changed = True
            override_reasons.append("sell_recast_to_avoid_no_inventory")

        if debate.winning_side == "bull" and final_action in {TradeAction.SELL, TradeAction.HOLD, TradeAction.AVOID}:
            entry_ok, entry_reason = self._buy_promotion_gate(
                decision=decision,
                candidate=candidate,
                regime=regime,
                technical_memo=technical_memo,
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

        if final_action == TradeAction.BUY and fallback_origin:
            final_action = TradeAction.HOLD if has_inventory else TradeAction.AVOID
            final_action_changed = True
            consistency_enforcement_changed_action = True
            final_action_downgraded = True
            buy_blocked_due_to_fallback = True
            action_thesis_mismatch_detected = True
            inconsistent_buy_prevented = True
            override_reasons.append(f"buy_blocked_fallback_origin:{fallback_reason or 'unknown'}")

        thesis_semantics = self._thesis_semantics(decision.thesis)
        if final_action == TradeAction.BUY:
            buy_rewrite_attempted = True
            buy_thesis_candidate = self._synthesize_final_thesis(
                final_action=TradeAction.BUY,
                decision=decision,
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
            desired_position_fraction = 0.0
        elif desired_position_fraction is None or desired_position_fraction <= 0:
            desired_position_fraction = min(0.04, self.settings.risk.max_position_size_fraction)

        synthesized_thesis = self._synthesize_final_thesis(
            final_action=final_action,
            decision=decision,
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            has_inventory=has_inventory,
            fallback_origin=fallback_origin,
        )
        thesis_rewritten = synthesized_thesis.strip() != decision.thesis.strip()

        updated_source_extra = dict(decision.source_metadata.extra)
        updated_source_extra.update(
            {
                "fallback_origin": fallback_origin,
                "fallback_origin_reason": fallback_reason,
                "buy_promotion_applied": promoted_buy,
                "buy_promotion_source": "debate_bull" if promoted_buy_from_debate else None,
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
                + (["action_thesis_mismatch_detected"] if action_thesis_mismatch_detected else [])
            )
        )
        updated_decision = decision.model_copy(
            update={
                "action": final_action,
                "desired_position_fraction": desired_position_fraction,
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
        decision, debate = self._adjudicate_long_only_action(
            decision=decision,
            debate=debate,
            candidate=candidate,
            regime=regime,
            technical_memo=technical_memo,
            current_position=current_position,
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
