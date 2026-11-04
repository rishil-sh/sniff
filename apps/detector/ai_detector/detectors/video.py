"""Frame sampling + aggregation. Implemented on Day 5."""

from __future__ import annotations

from ai_detector.router import RoutedInput
from ai_detector.schemas import Evidence


async def collect(routed: RoutedInput) -> list[Evidence]:
    del routed
    return []
