"""Frame sampling + aggregation. Soundtrack goes through the audio path."""

from __future__ import annotations

import io
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path

from ai_detector.detectors import audio as audio_mod
from ai_detector.detectors.image import HEAD_VERSION, score_head
from ai_detector.router import RoutedInput, ffmpeg_exe, probe_duration_s
from ai_detector.schemas import Calibration, Evidence

N_FRAMES = 16
EDGE_SKIP_S = 0.5
FFMPEG = ffmpeg_exe()


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


def _frames_evidence(data: bytes, duration_s: float | None) -> Evidence:
    started = time.perf_counter()
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
        return Evidence(
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
        return Evidence(
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

    items = [_frames_evidence(routed.data, routed.duration_s)]

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
        items.extend(await audio_mod.collect(audio_routed))
    return items
