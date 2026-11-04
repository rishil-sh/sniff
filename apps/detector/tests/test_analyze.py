import asyncio

from ai_detector.analyze import analyze
from ai_detector.router import TEXT_MIN_TOKENS


def test_stub_text_returns_inconclusive():
    text = " ".join(["token"] * TEXT_MIN_TOKENS)
    verdict = asyncio.run(analyze(text=text, send_to_muse=False))
    assert verdict.verdict == "inconclusive"
    assert verdict.modality == "text"
    assert verdict.report is not None
    assert verdict.report.model == "stub"
    dumped = verdict.model_dump(mode="json")
    assert dumped["decided_by"] == "fusion"
    assert "evidence" in dumped


def test_short_text_ood_evidence():
    verdict = asyncio.run(analyze(text="too short", send_to_muse=False))
    assert verdict.verdict == "inconclusive"
    assert any(item.status == "out_of_domain" for item in verdict.evidence)
