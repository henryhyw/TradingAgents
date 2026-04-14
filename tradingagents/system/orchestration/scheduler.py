from __future__ import annotations

import logging
import time
from datetime import datetime
from zoneinfo import ZoneInfo

from tradingagents.system.orchestration.calendar_utils import is_market_day
from tradingagents.system.schemas import RunMode

from .runner import TradingSystemRunner


logger = logging.getLogger(__name__)


class DailyScheduler:
    def __init__(self, runner: TradingSystemRunner):
        self.runner = runner
        self._last_run_date = None

    def run_forever(self, run_at: str = "15:45", shortlist_size: int | None = None, execute: bool = True) -> None:
        hour, minute = [int(part) for part in run_at.split(":", maxsplit=1)]
        tz = ZoneInfo(self.runner.settings.run.market_timezone)
        while True:
            now = datetime.now(tz)
            today = now.date()
            should_run = (
                is_market_day(today)
                and (now.hour, now.minute) >= (hour, minute)
                and self._last_run_date != today
            )
            if should_run:
                try:
                    self.runner.run_once(
                        as_of_date=today,
                        mode=RunMode.DAILY,
                        shortlist_size=shortlist_size,
                        execute=execute,
                    )
                except RuntimeError as exc:
                    # Scheduler should stay alive even if one run fails.
                    logger.error("Scheduled run failed for %s: %s", today, exc)
                self._last_run_date = today
            time.sleep(self.runner.settings.run.loop_sleep_seconds)
