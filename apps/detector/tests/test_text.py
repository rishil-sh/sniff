import asyncio
import math
from types import SimpleNamespace

import torch

from ai_detector.analyze import analyze
from ai_detector.detectors import text as text_mod

ESSAY = "The river was high that spring, and it carried the old bridge away with it. " * 10


def _peaked_logits(y, vocab=50, peak=2.0):
    """Both models favour token y[t] at every step."""
    logits = torch.zeros(len(y), vocab)
    logits[torch.arange(len(y)), y] = peak
    return logits


def test_pair_scores_machine_like_vs_surprising():
    y = torch.arange(24) % 50
    lb, li = _peaked_logits(y), _peaked_logits(y)
    # Text the models would have written: Binoculars below 1, Fast-DetectGPT well above 0, all rank 0.
    bino, fast, ranks = text_mod.pair_scores(lb, li, y)
    assert bino < 0.7 and fast > 5 and ranks == [0] * 24
    # Same models, tokens they did not favour: Binoculars above 1, Fast-DetectGPT negative.
    bino2, fast2, ranks2 = text_mod.pair_scores(lb, li, (y + 25) % 50)
    assert bino2 > 1 and fast2 < -1 and all(r >= 1 for r in ranks2)


def test_gltr_bucket():
    assert [text_mod.gltr_bucket(r) for r in (0, 9, 10, 99, 999, 1000)] == [
        "top-10", "top-10", "top-100", "top-100", "top-1000", "rest",
    ]


# Zero-shot fakes use the class medians measured on RAID abstracts with the Llama-3.2-3B pair.
def _fake_scores(binoculars, fast_detect):
    return lambda _text: {
        "tokens": 160, "windows": 1, "truncated": False, "binoculars": binoculars, "fast_detect": fast_detect,
        "gltr": [[" the", "top-10"]] * 150 + [[" river", "rest"]] * 10,
    }


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _fake_classifiers(fakespot, desklib):
    """One window per classifier; a list of p values gives several windows."""
    as_windows = lambda p: [_logit(x) for x in (p if isinstance(p, list) else [p])]  # noqa: E731
    return lambda _text: {"txt.fakespot": as_windows(fakespot), "txt.desklib": as_windows(desklib)}


def _run(monkeypatch, zero_shot, classifiers):
    monkeypatch.setattr(text_mod, "score_text", _fake_scores(*zero_shot))
    monkeypatch.setattr(text_mod, "score_classifiers", _fake_classifiers(*classifiers))
    verdict = asyncio.run(analyze(text=ESSAY))
    return verdict, {e.id: e for e in verdict.evidence}


def test_machine_like_text_is_likely_ai(monkeypatch):
    # Raw sigmoids are overconfident: on eval_text a raw fakespot 0.99 calibrates to ~0.3, so use AI-typical logits.
    verdict, by_id = _run(monkeypatch, (0.60, 6.1), (0.99999, 0.9999))
    assert by_id["txt.binoculars"].p_ai > 0.8 and by_id["txt.fastdetect"].p_ai > 0.8
    assert by_id["txt.gltr"].p_ai is None and by_id["txt.gltr"].details["shares"]["top-10"] == 150 / 160
    assert verdict.verdict == "likely_ai"
    assert any("Paraphrasing" in c for c in verdict.report.caveats)


def test_human_like_text_has_no_ai_signal(monkeypatch):
    verdict, _ = _run(monkeypatch, (0.98, 0.4), (0.01, 0.02))
    assert verdict.verdict == "no_ai_signal"


def test_classifiers_flag_modern_essay_the_zero_shot_pair_misses(monkeypatch):
    # The car essay: Binoculars 1.04 / Fast-DetectGPT -0.59 (human-like), both classifiers ~1.0.
    verdict, by_id = _run(monkeypatch, (1.04, -0.59), (1.0, 1.0))
    assert by_id["txt.binoculars"].p_ai < 0.05  # still reported, but weight 0: no veto
    assert verdict.verdict == "likely_ai"


def test_one_classifier_misfiring_is_not_likely_ai(monkeypatch):
    # Wikipedia articles fakespot flagged while desklib did not. The stacker weighs desklib's confident "human"
    # above fakespot's misfire (no_ai_signal); the weighted rule's veto said inconclusive. Never likely_ai.
    verdict, _ = _run(monkeypatch, (0.98, 0.4), (0.96, 0.01))
    assert verdict.fusion.method == "stacker" and verdict.verdict != "likely_ai"


def test_classifier_failure_keeps_zero_shot_evidence(monkeypatch):
    monkeypatch.setattr(text_mod, "score_text", _fake_scores(0.98, 0.4))  # classifiers refused by conftest
    verdict = asyncio.run(analyze(text=ESSAY))
    status = {e.id: e.status for e in verdict.evidence}
    assert status["txt.fakespot"] == "error" and status["txt.binoculars"] == "ok"
    assert verdict.verdict == "inconclusive"


def test_model_load_failure_is_inconclusive():
    # conftest makes both loaders raise, as missing weights would
    verdict = asyncio.run(analyze(text=ESSAY))
    assert {e.id: e.status for e in verdict.evidence}["txt.binoculars"] == "error"
    assert verdict.verdict == "inconclusive"


def test_window_logits_average_into_the_document_score(monkeypatch):
    # An AI-written opening followed by human text: the mean window logit decides, every window is kept.
    _, by_id = _run(monkeypatch, (0.98, 0.4), ([0.99, 0.01, 0.01], 0.02))
    fakespot = by_id["txt.fakespot"]
    assert fakespot.details["windows"] == 3 and len(fakespot.details["window_logits"]) == 3
    assert abs(fakespot.raw_score - sum(_logit(p) for p in (0.99, 0.01, 0.01)) / 3) < 1e-9
    assert fakespot.p_ai < 0.5


def test_hidden_characters_are_reported_and_removed(monkeypatch):
    seen = []
    monkeypatch.setattr(text_mod, "score_text", _fake_scores(0.98, 0.4))
    monkeypatch.setattr(text_mod, "score_classifiers", lambda t: seen.append(t) or _fake_classifiers(0.01, 0.02)(t))
    verdict = asyncio.run(analyze(text=ESSAY.replace("river", "riv​er")))
    tamper = {e.id: e for e in verdict.evidence}["txt.tamper"]
    assert tamper.details["hidden"] == 10 and tamper.p_ai is None
    assert "​" not in seen[0] and verdict.verdict == "no_ai_signal"  # shown, never votes


def test_clean_text_matches_fakespot_preprocessing():
    md = "# Title\n\n**Bold** and _it_ with [a link](http://x.y) , `code` and &amp; done."
    assert text_mod.clean_text(md) == "Title Bold and it with a link, and & done."


def test_pair_scores_long_text_in_windows(monkeypatch):
    vocab = 50

    def tokenizer(_text, return_tensors):
        return SimpleNamespace(input_ids=torch.cat([torch.tensor([[0]]), torch.arange(1200)[None] % vocab], 1))

    tokenizer.decode = lambda ids: str(ids[0])
    model = lambda ids: SimpleNamespace(logits=torch.zeros(1, ids.shape[1], vocab))  # noqa: E731
    monkeypatch.setattr(text_mod, "load_pair", lambda: (tokenizer, model, model, "cpu"))
    s = text_mod.score_text("x")
    assert (s["windows"], s["tokens"], s["truncated"]) == (3, 1200, False)  # 512 + 512 + 176, nothing dropped
    monkeypatch.setattr(text_mod, "MAX_PAIR_WINDOWS", 2)
    s = text_mod.score_text("x")
    assert (s["windows"], s["tokens"], s["truncated"]) == (2, 1024, True)
    # uniform logits: every token is as likely as expected, so Binoculars is exactly 1
    assert abs(s["binoculars"] - 1) < 1e-6 and len(s["gltr"]) == 1024
