from __future__ import annotations

from datetime import date

import pandas as pd

from tradingagents.system.config import load_settings
from tradingagents.system.data.yfinance_provider import YFinanceMarketDataProvider


def _raw_history(start_price: float = 100.0, step: float = 0.5, periods: int = 120) -> pd.DataFrame:
    dates = pd.bdate_range(end=date(2026, 4, 13), periods=periods)
    rows = []
    close = start_price
    for ts in dates:
        rows.append(
            {
                "Open": close - 0.2,
                "High": close + 0.4,
                "Low": close - 0.6,
                "Close": close,
                "Volume": 2_000_000.0,
            }
        )
        close += step
    return pd.DataFrame(rows, index=dates)


def _batch_multi(symbols: list[str]) -> pd.DataFrame:
    base = _raw_history()
    data = {}
    for symbol in symbols:
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            data[(symbol, column)] = base[column].values
    return pd.DataFrame(data, index=base.index, columns=pd.MultiIndex.from_tuples(data.keys()))


def test_batch_history_success_path(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    provider = YFinanceMarketDataProvider(settings)
    calls: list[str] = []

    def fake_download(*args, **kwargs):  # noqa: ANN002, ANN003
        return _batch_multi(["SPY", "NVDA"])

    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, *args, **kwargs):  # noqa: ANN002, ANN003
            calls.append(self.symbol)
            return pd.DataFrame()

    monkeypatch.setattr("tradingagents.system.data.yfinance_provider.yf.download", fake_download)
    monkeypatch.setattr("tradingagents.system.data.yfinance_provider.yf.Ticker", FakeTicker)
    histories = provider.batch_get_history(["SPY", "NVDA"], date(2026, 4, 13), 90)
    assert set(histories.keys()) == {"SPY", "NVDA"}
    assert calls == []


def test_single_symbol_history_prefers_ticker_history(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    provider = YFinanceMarketDataProvider(settings)

    def fail_download(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("yf.download should not be used for single-symbol get_history")

    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, *args, **kwargs):  # noqa: ANN002, ANN003
            return _raw_history(start_price=110.0, step=0.7, periods=160)

    monkeypatch.setattr("tradingagents.system.data.yfinance_provider.yf.download", fail_download)
    monkeypatch.setattr("tradingagents.system.data.yfinance_provider.yf.Ticker", FakeTicker)
    frame = provider.get_history("SPY", date(2026, 4, 13), 90)
    assert not frame.empty
    assert len(frame) >= 60


def test_batch_history_partial_failure_falls_back_to_ticker(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    settings = load_settings()
    provider = YFinanceMarketDataProvider(settings)
    calls: list[str] = []

    def fake_download(*args, **kwargs):  # noqa: ANN002, ANN003
        return _batch_multi(["SPY"])

    class FakeTicker:
        def __init__(self, symbol: str):
            self.symbol = symbol

        def history(self, *args, **kwargs):  # noqa: ANN002, ANN003
            calls.append(self.symbol)
            if self.symbol == "NVDA":
                return _raw_history(start_price=200.0, step=1.0, periods=140)
            return pd.DataFrame()

    monkeypatch.setattr("tradingagents.system.data.yfinance_provider.yf.download", fake_download)
    monkeypatch.setattr("tradingagents.system.data.yfinance_provider.yf.Ticker", FakeTicker)
    histories = provider.batch_get_history(["SPY", "NVDA"], date(2026, 4, 13), 90)
    assert set(histories.keys()) == {"SPY", "NVDA"}
    assert "NVDA" in calls
