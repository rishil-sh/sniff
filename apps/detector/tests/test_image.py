import asyncio
import io

from PIL import Image

from ai_detector.analyze import analyze
from ai_detector.detectors import image as image_mod
from ai_detector.router import route


def _png_bytes(size: int = 256) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), (12, 80, 160)).save(buf, format="PNG")
    return buf.getvalue()


def test_collect_skips_non_image():
    routed = route(text=" ".join(["token"] * 200))
    assert asyncio.run(image_mod.collect(routed)) == []


def test_collect_scores(monkeypatch):
    monkeypatch.setattr(image_mod, "score_head", lambda _im: (1.0, 0.73))
    monkeypatch.setattr(image_mod, "score_commfor", lambda _im: (2.0, 0.88))
    monkeypatch.setattr(image_mod, "score_clip", lambda _im: (0.4, 0.60))
    routed = route(data=_png_bytes(), filename="shot.png")
    items = asyncio.run(image_mod.collect(routed))
    by_id = {item.id: item for item in items}
    assert by_id["img.commfor"].status == "ok"
    assert by_id["img.commfor"].p_ai == 0.88
    assert by_id["img.clip"].p_ai == 0.60


def test_analyze_image_fusion(monkeypatch):
    monkeypatch.setattr(image_mod, "score_head", lambda _im: (2.5, 0.92))
    monkeypatch.setattr(image_mod, "score_commfor", lambda _im: (3.0, 0.92))
    monkeypatch.setattr(image_mod, "score_clip", lambda _im: (1.2, 0.81))
    verdict = asyncio.run(analyze(data=_png_bytes(), filename="shot.png"))
    assert verdict.modality == "image"
    assert verdict.verdict == "likely_ai"
    assert verdict.decided_by == "fusion"
    assert any(item.id == "img.c2pa" and item.status == "ok" for item in verdict.evidence)
    assert verdict.report is not None
    assert any(r.evidence_id == "img.head" for r in verdict.report.reasons)


def test_iptc_declares_ai(monkeypatch):
    monkeypatch.setattr(image_mod, "score_commfor", lambda _im: (0.0, 0.1))
    monkeypatch.setattr(image_mod, "score_clip", lambda _im: (0.0, 0.1))
    monkeypatch.setattr(
        "ai_detector.detectors.provenance._iptc_source",
        lambda _data: "trainedAlgorithmicMedia",
    )
    verdict = asyncio.run(analyze(data=_png_bytes(), filename="ai.png"))
    assert verdict.verdict == "declared_ai"
    assert verdict.decided_by == "provenance"


def test_domain_issue():
    assert image_mod.domain_issue(Image.new("RGB", (512, 512))) is None
    assert "palette" in image_mod.domain_issue(Image.new("RGB", (512, 512)).quantize(256))
    assert "transparent" in image_mod.domain_issue(Image.new("RGBA", (512, 512), (0, 0, 0, 0)))
    assert image_mod.domain_issue(Image.new("RGBA", (512, 512), (0, 0, 0, 255))) is None
    assert "256 px" in image_mod.domain_issue(Image.new("RGB", (512, 171)))


def test_out_of_domain_image_is_inconclusive(monkeypatch):
    monkeypatch.setattr(image_mod, "score_commfor", lambda _im: (-12.0, 0.0001))
    buf = io.BytesIO()
    Image.new("RGB", (512, 171), (90, 40, 120)).quantize(256).save(buf, format="PNG")
    verdict = asyncio.run(analyze(data=buf.getvalue(), filename="banner.png"))
    assert verdict.verdict == "inconclusive"
    assert {e.id: e.status for e in verdict.evidence}["img.commfor"] == "out_of_domain"


def test_spai_skipped_when_not_installed(monkeypatch):
    monkeypatch.setattr(image_mod, "score_commfor", lambda _im: (0.0, 0.1))
    monkeypatch.setattr(image_mod, "score_clip", lambda _im: (0.0, 0.1))
    items = asyncio.run(image_mod.collect(route(data=_png_bytes(), filename="shot.png")))
    spai = {item.id: item for item in items}["img.spai"]
    assert spai.status == "skipped" and "setup_spai.sh" in spai.details["reason"]


def test_spai_scored_when_installed(monkeypatch):
    monkeypatch.setattr(image_mod, "spai_ready", lambda: True)
    monkeypatch.setattr(image_mod, "score_head", lambda _im: (2.0, 0.9))
    monkeypatch.setattr(image_mod, "score_commfor", lambda _im: (2.0, 0.88))
    monkeypatch.setattr(image_mod, "score_clip", lambda _im: (1.0, 0.73))
    monkeypatch.setattr(image_mod, "score_spai", lambda _im: (3.0, 0.95))
    verdict = asyncio.run(analyze(data=_png_bytes(), filename="shot.png"))
    spai = {item.id: item for item in verdict.evidence}["img.spai"]
    # p_ai is Platt-calibrated from fusion.yaml, so only check SPAI ran (calibration: test_eval.py)
    assert spai.status == "ok" and spai.p_ai is not None and spai.details["input_size"] == "native"
    assert verdict.verdict == "likely_ai"


def test_shipped_head_matches_commfor_features():
    import numpy as np

    head = np.load(image_mod.HEAD_PATH)
    assert head["w"].shape == head["mean"].shape == head["std"].shape == (384,)
    assert np.isfinite(head["w"]).all() and float(head["b"]) == float(head["b"])
