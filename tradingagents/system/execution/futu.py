from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict

from tradingagents.system.schemas import FillRecord, OrderIntent, OrderRecord, PortfolioSnapshot

from .base import BrokerAdapter


class FutuBrokerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str | None = None
    port: int | None = None
    unlock_password: str | None = None
    trading_enabled: bool = False


class FutuBroker(BrokerAdapter):
    def __init__(self, config: FutuBrokerConfig):
        self.config = config

    def _raise_unavailable(self) -> None:
        raise RuntimeError(
            "Futu live trading is intentionally disabled in phase 1. "
            "Provide a valid OpenD endpoint and explicit live-trading enablement in phase 2."
        )

    def bootstrap(self, as_of_date: date) -> PortfolioSnapshot:
        self._raise_unavailable()

    def get_portfolio_snapshot(self, as_of_date: date) -> PortfolioSnapshot:
        self._raise_unavailable()

    def submit_order(self, intent: OrderIntent, as_of_date: date) -> tuple[OrderRecord, FillRecord | None]:
        self._raise_unavailable()
