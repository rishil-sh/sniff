from __future__ import annotations

from ai_detector.detectors import audio, image, provenance, text, video, watermark
from ai_detector.router import RoutedInput
from ai_detector.schemas import Evidence

_COLLECTORS = (
    provenance.collect,
    watermark.collect,
    image.collect,
    video.collect,
    audio.collect,
    text.collect,
)


async def gather_evidence(routed: RoutedInput) -> list[Evidence]:
    items: list[Evidence] = []
    for collect in _COLLECTORS:
        items.extend(await collect(routed))
    if routed.out_of_domain:
        items.append(
            Evidence(
                id="sys.router",
                detector="modality-router",
                version="0.1.0",
                modality=routed.modality,
                signal="learned",
                status="out_of_domain",
                details={"reason": routed.reason},
            )
        )
    return items
