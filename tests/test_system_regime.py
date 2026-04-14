from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.context import RegimeAnalyzer
from tradingagents.system.schemas import RegimeLabel

from .system_helpers import FakeMarketDataProvider, make_price_history


def _build_histories(as_of: date, vix_start: float, vix_step: float):
    symbols = ["SPY", "QQQ", "IWM", "XLK", "XLU", "XLP", "TLT", "UUP", "^VIX"]
    histories = {}
    for symbol in symbols:
        step = 0.45
        start = 100.0
        if symbol in {"QQQ", "XLK"}:
            step = 0.8
        if symbol in {"XLU", "XLP"}:
            step = 0.2
        if symbol == "^VIX":
            step = vix_step
            start = vix_start
        histories[symbol] = make_price_history(as_of, periods=220, start_price=start, step=step, volume=5_000_000)
    return histories


def test_regime_analyzer_classifies_risk_on(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 13)
    provider = FakeMarketDataProvider(_build_histories(as_of, vix_start=16.0, vix_step=-0.02))
    regime = RegimeAnalyzer(settings, provider).analyze(as_of)
    assert regime.label in {RegimeLabel.RISK_ON, RegimeLabel.BALANCED}
    assert 0.0 < regime.max_gross_exposure_fraction <= 1.0


def test_regime_analyzer_flags_high_vol_regime(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    as_of = date(2026, 4, 13)
    provider = FakeMarketDataProvider(_build_histories(as_of, vix_start=35.0, vix_step=0.0))
    regime = RegimeAnalyzer(settings, provider).analyze(as_of)
    assert regime.label == RegimeLabel.HIGH_VOLATILITY
    assert regime.risk_budget_multiplier == settings.risk.regime_high_vol_multiplier
