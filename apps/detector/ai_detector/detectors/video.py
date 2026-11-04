"""Frame sampling + aggregation. Soundtrack goes through the audio path."""

from __future__ import annotations

import io
import math
import os
import re
import subprocess
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from ai_detector.detectors import audio as audio_mod
from ai_detector.detectors.image import HEAD_VERSION, score_head
from ai_detector.router import RoutedInput, ffmpeg_exe, probe_duration_s
from ai_detector.schemas import Calibration, Evidence

N_FRAMES = 16
EDGE_SKIP_S = 0.5
FFMPEG = ffmpeg_exe()
# Every sampled frame is read (~2.5 s for 16 at 848p, ~6 s at 4K, CPU): on 848p re-encodes the Kling mark reads
# in only 7-10 of 16 frames.
# Visible marks generators burn into output: name -> (pattern on one lowercased OCR box, frames it must be read in).
# Distinctive brand names need 2 frames, so one frame of scene text is not enough. Names that are also ordinary
# words ("veo" is Spanish, "sora" Japanese) must be the whole box and read in 3 frames, so a subtitle can't trip them.
# Runway, Pika and Luma are left out: ordinary words, and no confirmed mark text to anchor on.
VISIBLE_MARKS: dict[str, tuple[re.Pattern[str], int]] = {
    # checked on 6 Kling 3.0/4.0 clips: "KlingAI 3.0", "KingAl3.0", "KlngAI40", "KIingAI", "OKlingAI3.04K" (logo as O)
    "Kling AI": (re.compile(r"(?<![a-z])o?k[l1i]{0,2}nga[il1](?![a-z])|可灵"), 2),
    # Google: a visible "Veo" in the bottom-right of Veo 3 videos (all but Flow on AI Ultra)
    "Google Veo": (re.compile(r"\W*ve[o0]\W*"), 3),
    # OpenAI: Sora 2 downloads carry a visible Sora mark, with the creator's @handle beside it
    "OpenAI Sora": (re.compile(r"\W*s[o0]ra(\s*@\S+)?\W*"), 3),
    # shortcut: names from vendor docs and reviews, no sample checked yet; confirm each when a clip turns up
    "Hailuo AI": (re.compile(r"(?<![a-z])hailuo|海螺\s*ai"), 2),
    "Jimeng / Dreamina": (re.compile(r"即梦|(?<![a-z])dreamina(?![a-z])"), 2),
    "PixVerse": (re.compile(r"(?<![a-z])pixverse(?![a-z])"), 2),
    "Vidu": (re.compile(r"(?<![a-z])vidu\s*(ai)?(?![a-z])"), 2),
    # China's explicit label (in force 2025-09-01), shown by any compliant generator or platform
    "AI label (AI生成)": (re.compile(r"ai\s*生成|人工智能生成"), 3),
}
FULL_BOX_MARKS = {"Google Veo", "OpenAI Sora"}  # pattern must match the whole OCR box, not a word inside it


def frame_times(duration_s: float, n: int = N_FRAMES, edge: float = EDGE_SKIP_S) -> list[float]:
    """Evenly spaced timestamps, skipping first/last `edge` seconds. Pure; unit-tested."""
    if duration_s <= 0 or n <= 0:
        return [0.0]
    start = min(edge, duration_s / 2)
    end = max(duration_s - edge, start)
    if n == 1 or end <= start:
        return [round((start + end) / 2, 4)]
    return [round(start + (end - start) * i / (n - 1), 4) for i in range(n)]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def find_visible_mark(texts_per_frame: list[list[str]]) -> tuple[str | None, int]:
    """(generator, frames) for the mark read in the most frames, if enough to count; else (None, best). Pure."""
    found: tuple[str | None, int] = (None, 0)
    best = 0
    for name, (pattern, min_frames) in VISIBLE_MARKS.items():
        match = pattern.fullmatch if name in FULL_BOX_MARKS else pattern.search
        hits = sum(any(match(text.strip().lower()) for text in texts) for texts in texts_per_frame)
        best = max(best, hits)
        if hits >= min_frames and hits > found[1]:
            found = (name, hits)
    return found if found[0] else (None, best)


@lru_cache(maxsize=1)
def _ocr():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()


def read_text(image) -> list[str]:
    result, _ = _ocr()(np.asarray(image)[:, :, ::-1])  # RapidOCR expects BGR
    return [row[1] for row in result or []]


def _mark_evidence(frames: list[tuple[float, object]]) -> Evidence:
    started = time.perf_counter()
    base = {"id": "vid.watermark", "detector": "rapidocr-visible-mark", "version": "rapidocr-onnxruntime@1.4.4",
            "modality": "video", "signal": "watermark"}
    try:
        texts = [read_text(image) for _, image in frames]
        generator, hits = find_visible_mark(texts)
        details = {
            "scheme": "visible mark (OCR)",
            "detected": generator is not None,
            "generator": generator,
            "frames_read": len(frames),
            "frames_with_mark": hits,
            "text": [t for frame in texts for t in frame][:12],
            "note": "Reads only generator names burned into the picture. Cropped or paid-tier videos have none.",
        }
        return Evidence(**base, status="ok", points_to="ai" if generator else "none", details=details,
                        latency_ms=int((time.perf_counter() - started) * 1000))
    except Exception as exc:  # noqa: BLE001
        return Evidence(**base, status="error", details={"error": str(exc)},
                        latency_ms=int((time.perf_counter() - started) * 1000))


def extract_frames(data: bytes, times: list[float]) -> list[tuple[float, object]]:
    """(t, RGB PIL Image) for each timestamp. One ffmpeg seek per frame."""
    from PIL import Image

    frames: list[tuple[float, object]] = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        for t in times:
            out = subprocess.run(
                [
                    FFMPEG,
                    "-v",
                    "error",
                    "-ss",
                    f"{t:.3f}",
                    "-i",
                    tmp.name,
                    "-frames:v",
                    "1",
                    "-f",
                    "image2pipe",
                    "-vcodec",
                    "mjpeg",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=60,
                check=True,
            )
            image = Image.open(io.BytesIO(out.stdout))
            image.load()
            frames.append((t, image.convert("RGB")))
    return frames


def extract_soundtrack(data: bytes) -> bytes | None:
    """16-bit PCM mono 16 kHz WAV from the video's audio track, or None if silent/missing."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        try:
            out = subprocess.run(
                [
                    FFMPEG,
                    "-v",
                    "error",
                    "-i",
                    tmp.name,
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    "-map_metadata",
                    "-1",
                    "-f",
                    "wav",
                    "pipe:1",
                ],
                capture_output=True,
                timeout=120,
                check=True,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
    return out.stdout or None


def _frames_evidence(data: bytes, duration_s: float | None) -> tuple[list[tuple[float, object]], Evidence]:
    """(frames, vid.head evidence); frames are reused for the visible-mark OCR, empty if extraction failed."""
    started = time.perf_counter()
    frames: list[tuple[float, object]] = []
    base = {
        "id": "vid.head",
        "detector": "commfor-screenshot-head",
        "version": HEAD_VERSION,
        "modality": "video",
        "signal": "learned",
        "weight": 1.0,
    }
    try:
        dur = duration_s if duration_s is not None else probe_duration_s(data, ".mp4")
        if dur is None:
            raise RuntimeError("could not read video duration")
        times = frame_times(dur)
        frames = extract_frames(data, times)
        logits: list[float] = []
        timeline: list[dict] = []
        for t, image in frames:
            raw, p_ai = score_head(image)
            logits.append(raw)
            timeline.append({"t": t, "p_ai": round(p_ai, 4)})
        agg = audio_mod.aggregate(logits)
        p_ai = _sigmoid(agg["median"])
        return frames, Evidence(
            **base,
            status="ok",
            raw_score=agg["median"],
            p_ai=p_ai,
            points_to="ai" if p_ai >= 0.5 else "real",
            details={
                "frames": len(frames),
                "duration_s": round(dur, 3),
                "top25_p_ai": round(_sigmoid(agg["top25_mean"]), 4),
                "timeline": timeline,
            },
            calibration=Calibration(set="eval_v2"),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return frames, Evidence(
            **base,
            status="error",
            details={"error": str(exc)},
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


async def collect(routed: RoutedInput) -> list[Evidence]:
    if routed.modality != "video" or routed.data is None or routed.out_of_domain:
        return []
    if os.environ.get("AI_DETECTOR_SKIP_ML"):
        return []

    frames, head = _frames_evidence(routed.data, routed.duration_s)
    items = [head]
    if frames:
        items.append(_mark_evidence(frames))

    wav = extract_soundtrack(routed.data)
    if wav:
        stem = Path(routed.filename or "clip.mp4").stem
        audio_routed = RoutedInput(
            modality="audio",
            out_of_domain=False,
            reason=None,
            filename=f"{stem}.wav",
            mime="audio/wav",
            size_bytes=len(wav),
            duration_s=None,
            token_count=None,
            sha256=routed.sha256,
            data=wav,
        )
        soundtrack = await audio_mod.collect(audio_routed)
        for e in soundtrack:
            e.weight = 0.0  # fusion ignores soundtrack scores on video; show the weight that was applied
        items.extend(soundtrack)
    return items
