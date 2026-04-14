from __future__ import annotations

from datetime import date

import pandas as pd

from tradingagents.system.config import SystemSettings
from tradingagents.system.data import MarketDataProvider
from tradingagents.system.schemas import RegimeLabel, RegimeSnapshot


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _safe_return(history: pd.DataFrame, window: int) -> float:
    if history.empty or len(history) <= window:
        return 0.0
    previous = float(history["Close"].iloc[-window - 1])
    if previous <= 0:
        return 0.0
    current = float(history["Close"].iloc[-1])
    return (current / previous) - 1.0


def _trend_score(history: pd.DataFrame) -> float:
    if history.empty or len(history) < 200:
        return 0.0
    close = float(history["Close"].iloc[-1])
    sma_50 = float(history["Close"].tail(50).mean())
    sma_200 = float(history["Close"].tail(200).mean())
    score = 0.0
    score += 0.5 if close > sma_200 else -0.5
    score += 0.3 if close > sma_50 else -0.3
    score += 0.2 if sma_50 > sma_200 else -0.2
    return _clamp(score, -1.0, 1.0)


class RegimeAnalyzer:
    def __init__(self, settings: SystemSettings, provider: MarketDataProvider):
        self.settings = settings
        self.provider = provider

    def analyze(self, as_of_date: date) -> RegimeSnapshot:
        lookback = max(self.settings.data.history_lookback_days, 260)
        proxies = self.settings.data.regime_proxies
        histories = self.provider.batch_get_history(proxies, as_of_date, lookback)
        warnings: list[str] = []
        notes: list[str] = []

        def history(symbol: str) -> pd.DataFrame:
            frame = histories.get(symbol)
            if frame is None or frame.empty:
                warnings.append(f"missing_proxy:{symbol}")
                return pd.DataFrame(columns=["Date", "Close"])
            return frame

        spy = history("SPY")
        qqq = history("QQQ")
        iwm = history("IWM")
        xlk = history("XLK")
        xlu = history("XLU")
        xlp = history("XLP")
        tlt = history("TLT")
        uup = history("UUP")
        vix = history("^VIX")

        spy_trend = _trend_score(spy)
        qqq_rel = _safe_return(qqq, 20) - _safe_return(spy, 20)
        iwm_rel = _safe_return(iwm, 20) - _safe_return(spy, 20)
        cyclical_vs_defensive = _safe_return(xlk, 20) - _safe_return(xlu, 20)
        staples_vs_market = _safe_return(xlp, 20) - _safe_return(spy, 20)
        duration_signal = _safe_return(tlt, 20)
        dollar_signal = _safe_return(uup, 20)
        vix_level = float(vix["Close"].iloc[-1]) if not vix.empty else 20.0
        vix_return_20d = _safe_return(vix, 20)

        # Composite risk-on score in [-1, 1].
        risk_on_score = (
            0.35 * spy_trend
            + 0.20 * _clamp(qqq_rel * 8.0, -1.0, 1.0)
            + 0.15 * _clamp(iwm_rel * 8.0, -1.0, 1.0)
            + 0.15 * _clamp(cyclical_vs_defensive * 8.0, -1.0, 1.0)
            + 0.05 * _clamp(-staples_vs_market * 8.0, -1.0, 1.0)
            + 0.05 * _clamp(duration_signal * 8.0, -1.0, 1.0)
            + 0.10 * _clamp(-vix_return_20d * 3.0, -1.0, 1.0)
            + 0.05 * _clamp(-dollar_signal * 6.0, -1.0, 1.0)
        )
        risk_on_score = _clamp(risk_on_score, -1.0, 1.0)

        if vix_level >= 30:
            label = RegimeLabel.HIGH_VOLATILITY
            multiplier = self.settings.risk.regime_high_vol_multiplier
            volatility_regime = "stressed"
            trend_regime = "unstable"
        elif risk_on_score <= -0.25:
            label = RegimeLabel.RISK_OFF
            multiplier = self.settings.risk.regime_risk_off_multiplier
            volatility_regime = "elevated" if vix_level >= 22 else "normal"
            trend_regime = "defensive"
        elif risk_on_score >= 0.25 and spy_trend > 0:
            label = RegimeLabel.RISK_ON
            multiplier = self.settings.risk.regime_risk_on_multiplier
            volatility_regime = "contained" if vix_level < 20 else "normal"
            trend_regime = "pro-cyclical"
        else:
            label = RegimeLabel.BALANCED
            multiplier = self.settings.risk.regime_balanced_multiplier
            volatility_regime = "normal" if vix_level < 24 else "elevated"
            trend_regime = "mixed"

        max_gross = _clamp(
            self.settings.risk.max_gross_exposure_fraction * multiplier,
            0.05,
            0.80,
        )
        notes.append(f"SPY trend score {spy_trend:+.2f}")
        notes.append(f"VIX level {vix_level:.2f}")
        notes.append(f"Risk-on composite {risk_on_score:+.2f}")

        return RegimeSnapshot(
            as_of_date=as_of_date,
            label=label,
            volatility_regime=volatility_regime,
            trend_regime=trend_regime,
            risk_on_score=risk_on_score,
            risk_budget_multiplier=multiplier,
            max_gross_exposure_fraction=max_gross,
            signals={
                "spy_trend": spy_trend,
                "qqq_relative_strength_20d": qqq_rel,
                "iwm_relative_strength_20d": iwm_rel,
                "cyclical_vs_defensive_20d": cyclical_vs_defensive,
                "staples_vs_market_20d": staples_vs_market,
                "duration_return_20d": duration_signal,
                "dollar_return_20d": dollar_signal,
                "vix_level": vix_level,
                "vix_return_20d": vix_return_20d,
            },
            notes=notes,
            warnings=sorted(set(warnings)),
            data_quality="degraded" if warnings else "ok",
        )
