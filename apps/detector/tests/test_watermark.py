import io

import cv2
import numpy as np
from imwatermark import WatermarkEncoder
from PIL import Image

from ai_detector.detectors.watermark import SCHEMES, decode, match_scheme


def _png(bgr) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).save(buf, format="PNG")
    return buf.getvalue()


def _photo_like(seed: int):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    return cv2.resize(small, (512, 512), interpolation=cv2.INTER_CUBIC)


def test_sdxl_watermark_round_trip():
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", SCHEMES["SDXL"])
    marked = encoder.encode(_photo_like(0), "dwtDct")
    scheme, errors = match_scheme(decode(_png(marked)))
    assert scheme == "SDXL", errors


def test_unmarked_image_has_no_scheme():
    scheme, _ = match_scheme(decode(_png(_photo_like(1))))
    assert scheme is None
