from __future__ import annotations

from datetime import date

from tradingagents.system.config import load_settings
from tradingagents.system.schemas import RegimeLabel, RegimeSnapshot
from tradingagents.system.universe import UniverseSelector

from .system_helpers import FakeMarketDataProvider, make_price_history, symbols_with_same_history


def test_universe_selector_builds_ranked_shortlist_with_reasons(monkeypatch, tmp_path):
    monkeypatch.setenv("TRADINGAGENTS_HOME", str(tmp_path / ".tradingagents"))
    monkeypatch.setenv("TRADINGAGENTS_MAX_SHORTLIST_PER_SECTOR", "1")
    settings = load_settings()
    as_of = date(2026, 4, 13)

    universe_path = tmp_path / "universe.csv"
    universe_path.write_text(
        "\n".join(
            [
                "symbol,name,asset_type,sector,style_tags,benchmark_symbol,peer_group",
                "AAA,Alpha A,Equity,Technology,growth|quality,XLK,Tech",
                "AAB,Alpha B,Equity,Technology,growth,XLK,Tech",
                "BBB,Beta B,Equity,Financials,value|quality,XLF,Financials",
                "SPY,S&P 500 ETF,ETF,Broad Market,core,SPY,Broad",
            ]
        ),
        encoding="utf-8",
    )

    histories = {
        **symbols_with_same_history(["AAA"], make_price_history(as_of, periods=180, start_price=80, step=1.0, volume=6_000_000)),
        **symbols_with_same_history(["AAB"], make_price_history(as_of, periods=180, start_price=120, step=0.3, volume=6_000_000)),
        **symbols_with_same_history(["BBB"], make_price_history(as_of, periods=180, start_price=60, step=0.7, volume=6_000_000)),
        **symbols_with_same_history(["SPY"], make_price_history(as_of, periods=180, start_price=100, step=0.4, volume=8_000_000)),
    }
    selector = UniverseSelector(settings, FakeMarketDataProvider(histories), universe_path=universe_path)
    regime = RegimeSnapshot(
        as_of_date=as_of,
        label=RegimeLabel.RISK_ON,
        volatility_regime="contained",
        trend_regime="pro-cyclical",
        risk_on_score=0.55,
        risk_budget_multiplier=1.1,
        max_gross_exposure_fraction=0.33,
    )
    screened = selector.screen_universe(as_of, regime=regime)
    shortlist = selector.build_shortlist_from_screened(screened, shortlist_size=2)
    assert len(shortlist) == 2
    assert all(asset.shortlist_reason for asset in shortlist)
    assert all("momentum" in asset.ranking_breakdown for asset in shortlist if not asset.rejection_reasons)
