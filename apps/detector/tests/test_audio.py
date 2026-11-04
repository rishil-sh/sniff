import asyncio
from pathlib import Path

import numpy as np

from ai_detector.detectors import audio as audio_mod
from ai_detector.fusion import fuse
from ai_detector.router import RoutedInput

SR = audio_mod.SR


def _routed() -> RoutedInput:
    return RoutedInput(
        modality="audio", out_of_domain=False, reason=None, filename="clip.wav", mime="audio/wav",
        size_bytes=4, duration_s=10.0, token_count=None, sha256="x", data=b"RIFF",
    )


def _speechlike(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.standard_normal(int(seconds * SR)) * 0.1).astype(np.float32)  # white: full band


def test_windows_cover_clip_and_tile_short():
    wins = audio_mod.windows(_speechlike(10), 64_600)
    starts = [t for t, _ in wins]
    assert starts[0] == 0.0 and all(c.size == 64_600 for _, c in wins)
    assert starts[-1] * SR + 64_600 == 10 * SR  # tail window ends at the clip end
    short = audio_mod.windows(_speechlike(1), 64_600)
    assert len(short) == 1 and short[0][1].size == 64_600


def test_windows_drop_silence():
    wav = np.concatenate([_speechlike(5), np.zeros(10 * SR, np.float32)])
    assert all(t < 5 for t, _ in audio_mod.windows(wav, 64_600))
    assert audio_mod.windows(np.zeros(10 * SR, np.float32), 64_600)  # all silent: keep them, don't return nothing


def test_aggregate():
    agg = audio_mod.aggregate([-3, -2, -1, 4])
    assert agg["median"] == -1.5 and agg["top25_mean"] == 4


def test_narrowband():
    assert not audio_mod.narrowband(_speechlike(3))
    t = np.arange(3 * SR) / SR
    assert audio_mod.narrowband((0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32))


def test_segments_min_length():
    mask = np.zeros(10 * SR, bool)
    mask[2 * SR : 4 * SR] = True
    mask[6 * SR : 6 * SR + 100] = True  # 6 ms blip: ignored
    assert audio_mod.segments(mask, SR, 1.0) == [{"start": 2.0, "end": 4.0}]


def _patch(monkeypatch, wav, mambo, dfa, seal, probe=None):
    """Stub every model. probe=None means no trained probe file; a float is its per-window logit."""
    monkeypatch.delenv("AI_DETECTOR_SKIP_ML", raising=False)
    monkeypatch.setattr(audio_mod, "decode", lambda data, suffix: wav)
    monkeypatch.setattr(audio_mod, "score_mambo", lambda chunks: [mambo] * len(chunks))
    monkeypatch.setattr(audio_mod, "score_dfarena", lambda chunks: [dfa] * len(chunks))
    monkeypatch.setattr(audio_mod, "detect_audioseal", lambda w: seal(w))
    if probe is None:
        monkeypatch.setattr(audio_mod, "PROBE_PATH", Path("/nonexistent/probe.npz"))
    else:
        monkeypatch.setattr(audio_mod, "PROBE_PATH", Path(__file__))  # exists; contents never read
        monkeypatch.setattr(audio_mod, "load_probe", lambda: {"tag": np.array("test")})
        monkeypatch.setattr(audio_mod, "score_w2vbert", lambda chunks: [probe] * len(chunks))


def _silent(w):
    return np.zeros(w.size, np.float32)


def test_probe_decides_tts_clip(monkeypatch):
    _patch(monkeypatch, _speechlike(10), 3.0, 2.5, _silent, probe=4.0)
    items = asyncio.run(audio_mod.collect(_routed()))
    by_id = {e.id: e for e in items}
    assert by_id["aud.audioseal"].signal == "watermark" and by_id["aud.audioseal"].points_to == "none"
    probe = by_id["aud.w2vbert"]
    assert probe.status == "ok" and probe.p_ai > 0.9 and probe.weight == 1.0
    assert len(probe.details["timeline"]) == probe.details["windows"]
    verdict = fuse(_routed(), items, request_id="t")
    assert verdict.verdict == "likely_ai" and verdict.decided_by == "fusion"


def test_evidence_only_detectors_cannot_veto_the_probe(monkeypatch):
    # DF Arena and MamBo say AI with confidence; the probe says human. Weight 0 = shown, no vote, no veto.
    _patch(monkeypatch, _speechlike(10), 4.0, 4.0, _silent, probe=-4.0)
    verdict = fuse(_routed(), asyncio.run(audio_mod.collect(_routed())), request_id="t")
    assert verdict.verdict == "no_ai_signal"
    by_id = {e.id: e for e in verdict.evidence}
    assert by_id["aud.dfarena"].p_ai > 0.9 and by_id["aud.dfarena"].weight == 0.0


def test_without_probe_weights_audio_is_inconclusive(monkeypatch):
    _patch(monkeypatch, _speechlike(10), 3.0, 2.5, _silent)
    items = asyncio.run(audio_mod.collect(_routed()))
    assert next(e for e in items if e.id == "aud.w2vbert").status == "skipped"
    assert fuse(_routed(), items, request_id="t").verdict == "inconclusive"


def test_collect_watermark_declares_ai_separately(monkeypatch):
    def seal(w):
        p = np.zeros(w.size, np.float32)
        p[2 * SR : 5 * SR] = 0.9
        return p

    _patch(monkeypatch, _speechlike(10), -3.0, -3.0, seal, probe=-3.0)
    items = asyncio.run(audio_mod.collect(_routed()))
    wm = next(e for e in items if e.id == "aud.audioseal")
    assert wm.details["segments"] == [{"start": 2.0, "end": 5.0}]
    verdict = fuse(_routed(), items, request_id="t")
    assert verdict.verdict == "declared_ai" and verdict.decided_by == "watermark"
    assert next(e for e in items if e.id == "aud.w2vbert").points_to == "real"  # classifier result kept apart


def test_collect_narrowband_is_out_of_domain(monkeypatch):
    t = np.arange(10 * SR) / SR
    _patch(monkeypatch, (0.1 * np.sin(2 * np.pi * 300 * t)).astype(np.float32), 3.0, 3.0, _silent, probe=3.0)
    items = asyncio.run(audio_mod.collect(_routed()))
    assert {e.status for e in items if e.signal == "learned"} == {"out_of_domain"}
    assert fuse(_routed(), items, request_id="t").verdict == "inconclusive"


def test_probe_score_uses_saved_layer_and_scaling(monkeypatch):
    feats = np.zeros((2, 25, 4), np.float32)
    feats[0, 7] = [3, 1, 1, 1]
    feats[1, 7] = [-1, 1, 1, 1]
    monkeypatch.setattr(audio_mod, "embed_w2vbert", lambda chunks: feats)
    monkeypatch.setitem(audio_mod._models, "probe", {
        "layer": np.int64(7), "mean": np.array([1, 1, 1, 1], np.float32), "scale": np.array([2, 1, 1, 1], np.float32),
        "coef": np.array([1, 0, 0, 0], np.float32), "intercept": np.float32(0.5), "tag": np.array("t"),
    })
    assert audio_mod.score_w2vbert([None, None]) == [1.5, -0.5]


def test_detector_weights_match_fusion_policy():
    from ai_detector.fusion import load_policy

    weights = load_policy()["weights"]
    assert {k: weights[k] for k in audio_mod.WEIGHTS} == audio_mod.WEIGHTS


def test_probe_features_pooling():
    emb = np.random.default_rng(0).standard_normal((3, 25, 4)).astype(np.float32)
    one = audio_mod.probe_features(emb, {"pool": np.array("one"), "layers": np.array([7])})
    cat = audio_mod.probe_features(emb, {"pool": np.array("cat"), "layers": np.array([6, 12, 18])})
    avg = audio_mod.probe_features(emb, {"pool": np.array("avg"), "layers": np.arange(1, 25)})
    assert np.array_equal(one, emb[:, 7])
    assert np.array_equal(cat, np.concatenate([emb[:, 6], emb[:, 12], emb[:, 18]], axis=1))
    assert np.allclose(avg, emb[:, 1:].mean(1))
    assert np.array_equal(audio_mod.probe_features(emb, {"layer": np.int64(7)}), emb[:, 7])  # en-v2 file format


def test_hot_ranges_merge_overlapping_windows():
    from ai_detector.muse.client import hot_ranges

    timeline = [{"t": 0.0, "p_ai": 0.9}, {"t": 2.0, "p_ai": 0.95}, {"t": 4.0, "p_ai": 0.1}, {"t": 8.0, "p_ai": 0.85}]
    assert hot_ranges(timeline, 4.0) == [(0.0, 6.0), (8.0, 12.0)]


def test_parse_ffmpeg_duration():
    from ai_detector.router import parse_ffmpeg_duration

    assert parse_ffmpeg_duration("  Duration: 00:01:02.50, start: 0.000000, bitrate: 128 kb/s") == 62.5
    assert parse_ffmpeg_duration("no banner") is None


def test_audio_report_reasons(monkeypatch):
    from ai_detector.muse.client import explain_verdict

    _patch(monkeypatch, _speechlike(10), -2.0, 3.0, _silent, probe=4.0)
    verdict = fuse(_routed(), asyncio.run(audio_mod.collect(_routed())), request_id="t")
    verdict.report = explain_verdict(verdict, send_to_muse=False)
    assert verdict.report.summary.startswith("Likely AI-generated speech")
    texts = [r.text for r in verdict.report.reasons]
    assert texts[0].startswith("Deciding detector, w2v-BERT 2.0 probe: p(AI) 0.98") and "0-10 s" in texts[0]
    assert any("No AudioSeal watermark" in t for t in texts)
    assert any(t.startswith("Context only, not used in the verdict: DF Arena 0.95, XLSR-MamBo 0.12") for t in texts)


def test_audio_report_explains_unscored_clip(monkeypatch):
    from ai_detector.muse.client import explain_verdict

    _patch(monkeypatch, _speechlike(10), 3.0, 3.0, _silent)  # no probe weights
    verdict = fuse(_routed(), asyncio.run(audio_mod.collect(_routed())), request_id="t")
    report = explain_verdict(verdict, send_to_muse=False)
    assert verdict.verdict == "inconclusive" and report.summary.startswith("Inconclusive. The speech detector could not")


def test_audio_stacker_is_pinned_to_the_shipped_probe():
    # A retrained probe changes its tag; the stacker must be refit (or re-pinned) or it silently switches off.
    from ai_detector.fusion import load_policy

    stacker = load_policy()["stackers"]["audio"]
    tag = str(audio_mod.load_probe()["tag"])
    assert stacker["versions"]["aud.w2vbert"] == f"{audio_mod.W2VBERT_REPO}@{audio_mod.W2VBERT_REV[:7]}+probe:{tag}"
