import asyncio
import subprocess

from ai_detector.detectors import video as video_mod
from ai_detector.fusion import fuse
from ai_detector.router import RoutedInput, ffmpeg_exe
from ai_detector.schemas import Evidence


def _routed(data: bytes = b"not-a-video", *, ood: bool = False) -> RoutedInput:
    return RoutedInput(
        modality="video",
        out_of_domain=ood,
        reason="only mp4 video is in scope" if ood else None,
        filename="clip.mp4",
        mime="video/mp4",
        size_bytes=len(data),
        duration_s=2.0,
        token_count=None,
        sha256="x",
        data=data,
    )


def test_frame_times_skip_edges():
    times = video_mod.frame_times(10.0, n=5, edge=0.5)
    assert times[0] == 0.5 and times[-1] == 9.5
    assert len(times) == 5
    assert video_mod.frame_times(0.8, n=16, edge=0.5)[0] == 0.4  # short clip: edge collapses to mid


def test_frame_times_single():
    assert video_mod.frame_times(5.0, n=1) == [2.5]


def test_collect_skips_ml(monkeypatch):
    monkeypatch.setenv("AI_DETECTOR_SKIP_ML", "1")
    assert asyncio.run(video_mod.collect(_routed())) == []


def test_collect_wrong_modality():
    routed = _routed()
    routed.modality = "image"
    assert asyncio.run(video_mod.collect(routed)) == []


def _tiny_mp4(seconds: float = 3.0) -> bytes:
    """Short colour-bars mp4 via ffmpeg (no audio). frag_keyframe lets mp4 go to a pipe."""
    out = subprocess.run(
        [
            ffmpeg_exe(),
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c=blue:s=320x240:d={seconds}",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "frag_keyframe+empty_moov",
            "-f",
            "mp4",
            "pipe:1",
        ],
        capture_output=True,
        timeout=30,
        check=True,
    )
    return out.stdout


def test_collect_vid_head_with_mocked_scorer(monkeypatch):
    monkeypatch.delenv("AI_DETECTOR_SKIP_ML", raising=False)
    mp4 = _tiny_mp4(3.0)
    monkeypatch.setattr(video_mod, "score_head", lambda image: (2.0, 0.88))
    monkeypatch.setattr(video_mod, "extract_soundtrack", lambda data: None)
    monkeypatch.setattr(video_mod, "read_text", lambda image: [])
    routed = _routed(mp4)
    routed.duration_s = 3.0

    items = asyncio.run(video_mod.collect(routed))
    by_id = {e.id: e for e in items}
    assert "vid.head" in by_id
    head = by_id["vid.head"]
    assert head.status == "ok" and head.modality == "video"
    assert head.raw_score == 2.0 and head.p_ai is not None and head.p_ai > 0.8
    assert head.details["frames"] == video_mod.N_FRAMES
    assert len(head.details["timeline"]) == video_mod.N_FRAMES
    assert all("t" in row and "p_ai" in row for row in head.details["timeline"])


def test_vid_head_votes_in_fusion():
    evidence = [
        Evidence(
            id="vid.head",
            detector="commfor-screenshot-head",
            version="t",
            modality="video",
            signal="learned",
            status="ok",
            raw_score=3.0,
            p_ai=0.95,
            weight=1.0,
        )
    ]
    verdict = fuse(_routed(), evidence, request_id="v")
    assert verdict.modality == "video" and abs(verdict.p_ai - 0.9526) < 1e-3  # sigmoid(raw_score)
    assert verdict.verdict == "likely_ai"


def test_soundtrack_does_not_vote_on_video():
    """Soundtrack scores are shown on a video but only vid.head votes (fusion.py weight_of)."""
    evidence = [
        Evidence(
            id="vid.head",
            detector="h",
            version="t",
            modality="video",
            signal="learned",
            status="ok",
            raw_score=-1.3863,
            p_ai=0.2,
            weight=1.0,
        ),
        Evidence(
            id="aud.w2vbert",
            detector="w",
            version="t",
            modality="audio",
            signal="learned",
            status="ok",
            p_ai=0.99,
            weight=1.0,
        ),
    ]
    verdict = fuse(_routed(), evidence, request_id="v")
    assert abs(verdict.p_ai - 0.2) < 1e-3
    # re-encoded Kling clips score 0.13-0.34 too: a low frame score never clears a video
    assert verdict.verdict == "inconclusive"


def test_find_visible_mark_tolerates_ocr_misreads():
    # OCR output read from real Kling 3.0/4.0 frames, originals and 848p re-encodes
    frames = [[" KingAI 3.0"], ["my mona", "KlingAl 4.0"], ["KlingAI3.04K"], ["KlngAI40"], ["OKIingAI3.04K"], []]
    assert video_mod.find_visible_mark(frames) == ("Kling AI", 5)


def test_find_visible_mark_needs_two_frames_and_whole_words():
    assert video_mod.find_visible_mark([["KlingAI 3.0"], ["sunset"]]) == (None, 1)  # one frame of scene text
    # OCR text read from real (non-AI) clips, plus near-miss words
    real = [["PrizePool", "18400"], ["Water saved:O Liters", "HELLO"], ["making a list"], ["the King Al"], ["kingdom"]]
    assert video_mod.find_visible_mark(real) == (None, 0)


def test_visible_mark_decides_video(monkeypatch):
    monkeypatch.delenv("AI_DETECTOR_SKIP_ML", raising=False)
    monkeypatch.setattr(video_mod, "score_head", lambda image: (0.3, 0.57))
    monkeypatch.setattr(video_mod, "extract_soundtrack", lambda data: None)
    monkeypatch.setattr(video_mod, "read_text", lambda image: ["KlingAI 3.0"])
    routed = _routed(_tiny_mp4(3.0))
    routed.duration_s = 3.0
    items = asyncio.run(video_mod.collect(routed))
    mark = next(e for e in items if e.id == "vid.watermark")
    assert mark.points_to == "ai" and mark.details["frames_read"] == video_mod.N_FRAMES
    assert fuse(routed, items, request_id="v").verdict == "declared_ai"
