import json
from types import SimpleNamespace

from ai_detector.muse import client as muse
from ai_detector.schemas import Evidence, FusionTrace, Verdict


def _verdict(label, p_ai, reason=None):
    evidence = [
        Evidence(id="txt.desklib", detector="d", version="v", modality="text", signal="learned", status="ok",
                 raw_score=0.1, p_ai=0.52, details={"window_logits": [2.0, -1.5]}),
        Evidence(id="txt.gltr", detector="g", version="v", modality="text", signal="learned", status="ok",
                 details={"shares": {"top-10": 0.6}, "tokens": [["secret", "top-10"]]}),
    ]
    if reason:
        evidence.append(Evidence(id="sys.router", detector="r", version="v", modality="text", signal="learned",
                                 status="out_of_domain", details={"reason": reason}))
    return Verdict(request_id="r", modality="text", sha256="s", verdict=label, p_ai=p_ai, evidence=evidence,
                   fusion=FusionTrace(policy="t"))


def _fake_muse(monkeypatch, reply=None, error=None):
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        if error:
            raise error
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply)))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(muse, "muse_client", lambda: client)
    return sent


WHY = "The opening reads like AI writing, while the ending reads like a person wrote it."


def test_whatsapp_message_headline_is_set_by_code():
    msg = muse.explain_verdict(_verdict("likely_ai", 0.9), send_to_muse=False).answer
    assert msg.startswith("🤖 *Written by AI*\n\n") and msg.endswith(muse.AI_FOOTER)
    msg = muse.explain_verdict(_verdict("no_ai_signal", 0.01), send_to_muse=False).answer
    assert msg.startswith("✍️ *Written by a human*") and muse.AI_FOOTER not in msg
    report = muse.explain_verdict(_verdict("inconclusive", 0.3), send_to_muse=False)
    assert report.lean == "human" and report.answer.startswith("🤔 *Can't say for sure. Best guess: human*")
    report = muse.explain_verdict(_verdict("inconclusive", None, reason="text shorter than 150 tokens"),
                                  send_to_muse=False)
    assert report.lean == "unclear" and "at least a full paragraph" in report.answer


def test_muse_writes_the_why_and_the_inconclusive_lean(monkeypatch):
    sent = _fake_muse(monkeypatch, {"why": WHY, "lean": "ai"})
    report = muse.explain_verdict(_verdict("inconclusive", 0.45), send_to_muse=True)
    assert report.model == muse.MUSE_MODEL and report.lean == "ai"
    assert report.answer == f"🤔 *Can't say for sure. Best guess: AI*\n\n{WHY}\n\n{muse.AI_FOOTER}"
    payload = sent["messages"][1]["content"]
    assert "window_logits" in payload and "secret" not in payload  # GLTR tokens (the text) are never sent


def test_muse_cannot_overturn_a_decided_verdict(monkeypatch):
    _fake_muse(monkeypatch, {"why": "This is clearly AI.", "lean": "ai"})  # about a no_ai_signal verdict
    report = muse.explain_verdict(_verdict("no_ai_signal", 0.01), send_to_muse=True)
    assert report.lean == "human" and "clearly AI" not in report.answer
    assert report.answer.endswith("none of the patterns that AI writing tools leave behind.")


def test_muse_failure_falls_back_to_the_template(monkeypatch):
    _fake_muse(monkeypatch, error=TimeoutError("504"))
    report = muse.explain_verdict(_verdict("inconclusive", 0.7), send_to_muse=True)
    assert report.model == "stub" and report.lean == "ai" and any("call failed" in c for c in report.caveats)
    assert "Best guess: AI" in report.answer


def test_whatsapp_reply_names_the_right_kind_of_ai_per_modality():
    for modality, words in (("image", "AI image generators"), ("audio", "AI voice generators")):
        verdict = _verdict("likely_ai", 0.95).model_copy(update={"modality": modality})
        assert words in muse.explain_verdict(verdict, send_to_muse=False).answer
    video = Verdict(request_id="r", modality="video", sha256="s", verdict="inconclusive", p_ai=None, evidence=[],
                    fusion=FusionTrace(policy="t"))
    assert "can't check videos yet" in muse.explain_verdict(video, send_to_muse=False).answer


def test_stub_reasons_are_plain_voting_first():
    v = _verdict("likely_ai", 0.9)
    v.evidence.append(Evidence(id="txt.binoculars", detector="b", version="v", modality="text", signal="zero_shot",
                               status="ok", p_ai=0.7, points_to="ai"))
    v.fusion = FusionTrace(policy="t", weights={"txt.desklib": 0.5, "txt.binoculars": 0})
    report = muse.explain_verdict(v, send_to_muse=False)
    texts = [r.text for r in report.reasons]
    assert texts[0].startswith("Voting detector, Desklib classifier: p(AI) 0.52")
    assert texts[1] == "Context only, not used in the verdict: Binoculars 0.70."
    assert not any("status=" in t for t in texts)
    assert report.summary == "🤖 Written by AI"


def test_palette_image_says_why_once():
    ev = [Evidence(id=i, detector="d", version="v", modality="image", signal="learned", status="out_of_domain",
                   details={"reason": "P-mode palette image"}) for i in ("img.head", "img.clip")]
    v = Verdict(request_id="r", modality="image", sha256="s", verdict="inconclusive", p_ai=None, evidence=ev,
                fusion=FusionTrace(policy="t"))
    report = muse.explain_verdict(v, send_to_muse=False)
    assert [r.text for r in report.reasons] == ["Out of scope: P-mode palette image."]
    assert "reduced colour palette" in report.answer
