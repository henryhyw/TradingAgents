from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from tradingagents.system.schemas import FillRecord, OrderIntent, OrderRecord, PortfolioSnapshot


class BrokerAdapter(ABC):
    @abstractmethod
    def bootstrap(self, as_of_date: date) -> PortfolioSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_portfolio_snapshot(self, as_of_date: date) -> PortfolioSnapshot:
        raise NotImplementedError

    @abstractmethod
    def submit_order(self, intent: OrderIntent, as_of_date: date) -> tuple[OrderRecord, FillRecord | None]:
        raise NotImplementedError
