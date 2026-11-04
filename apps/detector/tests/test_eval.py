import numpy as np

from ai_detector.fusion import fuse
from ai_detector.router import route
from ai_detector.schemas import Evidence
from scripts.eval_images import auroc, fit_logistic, fit_platt, in_test, tpr_at_fpr, tpr_ci, wilson


def test_auroc_and_tpr():
    labels = np.array([1, 1, 1, 0, 0, 0])
    assert auroc(np.array([3.0, 2.0, 1.0, 0.0, -1.0, -2.0]), labels) == 1.0
    assert auroc(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), labels) == 0.5
    assert tpr_at_fpr(np.array([3.0, 2.0, -5.0, 0.0, -1.0, -2.0]), labels, 0.01) == 2 / 3


def test_platt_recovers_known_slope():
    rng = np.random.default_rng(0)
    raw = rng.normal(0, 2, 20000)
    labels = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 * raw - 1.0)))).astype(int)
    a, b = fit_platt(raw, labels)
    assert abs(a - 0.5) < 0.05 and abs(b + 1.0) < 0.1


def test_fusion_applies_platt():
    ev = Evidence(
        id="img.commfor",
        detector="c",
        version="0",
        modality="image",
        signal="learned",
        status="ok",
        raw_score=-12.9,
        p_ai=2.4e-6,
    )
    policy = {"weights": {"img.commfor": 1.0}, "platt": {"img.commfor": {"a": 0.1, "b": 0.0, "set": "t"}}}
    verdict = fuse(
        route(text="It was the best of times, it was the worst of times. " * 15), [ev], request_id="x", policy=policy
    )
    assert abs(verdict.p_ai - 1 / (1 + np.exp(1.29))) < 1e-9
    assert verdict.evidence[0].calibration.set == "t"


def test_logistic_stacker_recovers_two_coefficients():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 2, (20000, 2))
    labels = (rng.random(20000) < 1 / (1 + np.exp(-(1.0 * x[:, 0] - 0.5 * x[:, 1] + 0.3)))).astype(int)
    w = fit_logistic(x, labels, l2=1e-3)
    assert np.allclose(w, [1.0, -0.5, 0.3], atol=0.08)


def test_interval_helpers():
    lo, hi = wilson(0, 300)
    assert lo == 0.0 and 0.011 < hi < 0.014  # rule of three: about 3/300
    labels = np.array([1] * 200 + [0] * 200)
    scores = np.concatenate([np.linspace(1, 3, 200), np.linspace(0, 2, 200)])
    lo, hi = tpr_ci(scores, labels, 0.05)
    assert lo <= tpr_at_fpr(scores, labels, 0.05) <= hi


def test_split_is_stable_and_near_the_share():
    paths = [f"real/g/{i}.txt" for i in range(4000)]
    share = sum(in_test(p, 0.5) for p in paths) / len(paths)
    assert 0.47 < share < 0.53 and [in_test(p, 0.5) for p in paths[:50]] == [in_test(p, 0.5) for p in paths[:50]]
