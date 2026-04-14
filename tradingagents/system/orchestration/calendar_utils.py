from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas_market_calendars as mcal


NYSE = mcal.get_calendar("NYSE")


def is_market_day(day: date) -> bool:
    schedule = NYSE.schedule(start_date=day.isoformat(), end_date=day.isoformat())
    return not schedule.empty


def previous_market_day(day: date) -> date:
    cursor = day - timedelta(days=1)
    while not is_market_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def next_market_days(start: date, end: date) -> list[date]:
    valid = NYSE.valid_days(start_date=start.isoformat(), end_date=end.isoformat())
    return [ts.tz_localize(None).date() for ts in valid]


def default_as_of_date(market_timezone: str) -> date:
    now = datetime.now(ZoneInfo(market_timezone))
    today = now.date()
    if is_market_day(today) and now.hour >= 16:
        return today
    return previous_market_day(today)
