from __future__ import annotations

import json
import re
from typing import Any

from tradingagents.system.schemas import TradeAction


RATING_PATTERN = re.compile(r"\b(BUY|OVERWEIGHT|HOLD|UNDERWEIGHT|SELL)\b", re.IGNORECASE)


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found")
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])
    raise ValueError("Unterminated JSON object")


def normalize_rating(text: str) -> str:
    match = RATING_PATTERN.search(text or "")
    if not match:
        return "HOLD"
    return match.group(1).upper()


def rating_to_action(rating: str) -> TradeAction:
    normalized = rating.upper()
    if normalized in {"BUY", "OVERWEIGHT"}:
        return TradeAction.BUY
    if normalized in {"SELL", "UNDERWEIGHT"}:
        return TradeAction.SELL
    return TradeAction.HOLD


def rating_to_confidence(rating: str) -> float:
    return {
        "BUY": 0.72,
        "SELL": 0.72,
        "OVERWEIGHT": 0.58,
        "UNDERWEIGHT": 0.58,
        "HOLD": 0.42,
    }.get(rating.upper(), 0.42)
