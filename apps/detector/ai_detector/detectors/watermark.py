"""Stable Diffusion invisible watermark (DWT-DCT). Video Seal / PixelSeal / AudioSeal on Day 2."""

from __future__ import annotations

import io
import time

from ai_detector.router import RoutedInput
from ai_detector.schemas import Evidence

VERSION = "invisible-watermark@0.2.0"

# Messages embedded by Stability's reference pipelines and diffusers.
SCHEMES = {
    "SDXL": [int(b) for b in bin(0b101100111110110010010000011110111011000110011110)[2:]],
    "StableDiffusionV1": [int(b) for byte in b"StableDiffusionV1" for b in f"{byte:08b}"],
}
# Chance of a random image landing within this many bit errors is < 1e-7 (6 of 48 SDXL bits);
# 12.5% keeps the mark readable through JPEG q90, which leaves ~5 bit errors.
MAX_BIT_ERROR_RATE = 0.125


def match_scheme(decoded: dict[str, list[int]]) -> tuple[str | None, dict[str, int]]:
    """Compare decoded bits against each known message. Pure; unit-tested."""
    errors = {
        name: sum(a != b for a, b in zip(decoded[name], bits, strict=True))
        for name, bits in SCHEMES.items()
    }
    hit = next(
        (name for name, err in errors.items() if err <= MAX_BIT_ERROR_RATE * len(SCHEMES[name])),
        None,
    )
    return hit, errors


def decode(data: bytes) -> dict[str, list[int]]:
    import cv2
    import numpy as np
    from imwatermark import WatermarkDecoder
    from PIL import Image

    rgb = Image.open(io.BytesIO(data)).convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    out = {}
    for name, bits in SCHEMES.items():
        raw = WatermarkDecoder("bits", len(bits)).decode(bgr, "dwtDct")
        out[name] = [int(b) for b in raw]
    return out


async def collect(routed: RoutedInput) -> list[Evidence]:
    if routed.modality != "image" or routed.data is None or routed.out_of_domain:
        return []
    started = time.perf_counter()
    try:
        scheme, errors = match_scheme(decode(routed.data))
        status, details = "ok", {"scheme": scheme, "bit_errors": errors}
    except Exception as exc:  # noqa: BLE001
        scheme, status, details = None, "error", {"error": str(exc)}
    return [
        Evidence(
            id="img.sdwm",
            detector="invisible-watermark-dwtDct",
            version=VERSION,
            modality="image",
            signal="watermark",
            status=status,  # type: ignore[arg-type]
            points_to="ai" if scheme else "none",
            details=details,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    ]
