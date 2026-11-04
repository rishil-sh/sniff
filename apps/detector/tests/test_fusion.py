import math

from ai_detector.fusion import fuse
from ai_detector.router import RoutedInput
from ai_detector.schemas import Evidence


def _routed(**kwargs) -> RoutedInput:
    defaults = dict(
        modality="image",
        out_of_domain=False,
        reason=None,
        filename="x.png",
        mime="image/png",
        size_bytes=10,
        duration_s=None,
        token_count=None,
        sha256="a" * 64,
    )
    defaults.update(kwargs)
    return RoutedInput(**defaults)


def test_ood_is_inconclusive():
    verdict = fuse(
        _routed(out_of_domain=True, reason="too small"),
        [],
        request_id="abcd1234",
    )
    assert verdict.verdict == "inconclusive"
    assert verdict.p_ai is None
    assert verdict.decided_by == "fusion"


def test_no_scores_is_inconclusive():
    verdict = fuse(_routed(), [], request_id="abcd1234")
    assert verdict.verdict == "inconclusive"
    assert verdict.fusion.policy.startswith("fusion.yaml@")


def test_openai_c2pa_claim_declares_ai_even_if_signature_invalid():
    evidence = [
        Evidence(
            id="img.c2pa",
            detector="c2pa-python",
            version="0",
            modality="image",
            signal="provenance",
            status="ok",
            points_to="ai",
            details={
                "manifest_present": True,
                "signature_valid": False,
                "claim_generator": "OpenAI Media Service API",
                "digital_source_type": "trainedAlgorithmicMedia",
            },
        )
    ]
    verdict = fuse(_routed(), evidence, request_id="abcd1234")
    assert verdict.verdict == "declared_ai"
    assert verdict.decided_by == "provenance"


def test_watermark_declares_ai():
    evidence = [
        Evidence(
            id="img.videoseal",
            detector="videoseal",
            version="0",
            modality="image",
            signal="watermark",
            status="ok",
            points_to="ai",
            details={"scheme": "videoseal", "detected": True},
        )
    ]
    verdict = fuse(_routed(), evidence, request_id="abcd1234")
    assert verdict.verdict == "declared_ai"
    assert verdict.decided_by == "watermark"


def test_weighted_bands():
    evidence = [
        Evidence(
            id="img.commfor",
            detector="community-forensics",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=0.91,
            points_to="ai",
        )
    ]
    policy = {"weights": {"img.commfor": 1.0}, "bands": {"likely_ai_above": 0.9}}
    verdict = fuse(_routed(), evidence, request_id="abcd1234", policy=policy)
    assert verdict.verdict == "likely_ai"
    assert abs((verdict.p_ai or 0) - 0.91) < 1e-9


def test_policy_weighted_mean():
    evidence = [
        Evidence(
            id="img.commfor",
            detector="community-forensics",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=0.95,
        ),
        Evidence(
            id="img.clip",
            detector="clip",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=0.90,
        ),
    ]
    policy = {"weights": {"img.commfor": 0.6, "img.clip": 0.4}, "bands": {"likely_ai_above": 0.9}}
    verdict = fuse(_routed(), evidence, request_id="abcd1234", policy=policy)
    assert verdict.verdict == "likely_ai"
    assert abs((verdict.p_ai or 0) - 0.93) < 1e-9


def test_near_chance_clip_does_not_block_no_ai_signal():
    evidence = [
        Evidence(
            id="img.commfor",
            detector="community-forensics",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=2.4e-6,
            points_to="real",
        ),
        Evidence(
            id="img.clip",
            detector="clip",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=0.532,
            points_to="ai",
        ),
    ]
    policy = {"weights": {"img.commfor": 0.6, "img.clip": 0.4}, "bands": {"no_ai_signal_below": 0.5}}
    verdict = fuse(_routed(), evidence, request_id="7d4f750b", policy=policy)
    assert verdict.verdict == "no_ai_signal"
    assert (verdict.p_ai or 1) < 0.01


def test_disagreement_is_inconclusive():
    evidence = [
        Evidence(
            id="a",
            detector="a",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=0.95,
        ),
        Evidence(
            id="b",
            detector="b",
            version="0",
            modality="image",
            signal="learned",
            status="ok",
            p_ai=0.20,
        ),
    ]
    verdict = fuse(_routed(), evidence, request_id="abcd1234")
    assert verdict.verdict == "inconclusive"


def test_zero_weight_detector_does_not_vote_or_veto():
    def learned(id_, p):
        return Evidence(id=id_, detector=id_, version="1", modality="image", signal="learned", status="ok", p_ai=p)

    policy = {
        "weights": {"img.commfor": 1.0, "img.spai": 0},
        "bands": {"likely_ai_above": 0.9},
        "max_disagreement": 0.6,
    }
    verdict = fuse(
        _routed(), [learned("img.commfor", 0.95), learned("img.spai", 0.05)], request_id="abcd1234", policy=policy
    )
    assert verdict.verdict == "likely_ai" and verdict.p_ai == 0.95


def test_platt_recalibration_updates_points_to():
    # raw -0.12 -> raw sigmoid 0.47 ("real"), Platt p ~0.72 ("ai"): label must follow p_ai
    ev = Evidence(
        id="img.clip",
        detector="clip",
        version="0",
        modality="image",
        signal="learned",
        status="ok",
        raw_score=-0.12,
        p_ai=0.47,
        points_to="real",
    )
    policy = {"platt": {"img.clip": {"a": 0.564935, "b": 0.993031}}, "weights": {"img.clip": 1}}
    verdict = fuse(_routed(), [ev], request_id="abcd1234", policy=policy)
    assert verdict.evidence[0].p_ai > 0.7
    assert verdict.evidence[0].points_to == "ai"


def _txt(eid, raw, p, version="v1"):
    return Evidence(
        id=eid, detector=eid, version=version, modality="text", signal="learned", status="ok", raw_score=raw, p_ai=p
    )


STACKER_POLICY = {
    "weights": {"txt.a": 0.5, "txt.b": 0.5},
    "stackers": {
        "text": {
            "intercept": -1.0,
            "coef": {"txt.a": 1.0, "txt.b": 0.5},
            "versions": {"txt.a": "v1"},
            "bands": {"no_ai_signal_below": 0.1, "likely_ai_above": 0.9},
        }
    },
}


def test_stacker_decides_with_its_own_bands_and_no_veto():
    # The weighted rule would call this inconclusive (0.99 vs 0.05 disagree); the stacker gives sigmoid(4 - 1 - 1).
    evidence = [_txt("txt.a", 4.0, 0.99), _txt("txt.b", -2.0, 0.05)]
    verdict = fuse(_routed(modality="text"), evidence, request_id="s", policy=STACKER_POLICY)
    assert verdict.fusion.method == "stacker" and abs(verdict.p_ai - 1 / (1 + math.exp(-2.0))) < 1e-9
    assert verdict.bands.likely_ai_above == 0.9 and verdict.verdict == "inconclusive"
    verdict = fuse(
        _routed(modality="text"),
        [_txt("txt.a", 8.0, 0.99), _txt("txt.b", 0.0, 0.5)],
        request_id="s",
        policy=STACKER_POLICY,
    )
    assert verdict.verdict == "likely_ai"


def test_stacker_falls_back_when_an_input_is_missing_or_from_another_model():
    for evidence in ([_txt("txt.a", 4.0, 0.95)], [_txt("txt.a", 4.0, 0.95, version="v2"), _txt("txt.b", 4.0, 0.95)]):
        verdict = fuse(_routed(modality="text"), evidence, request_id="s", policy=STACKER_POLICY)
        assert verdict.fusion.method == "weighted_mean" and verdict.bands.likely_ai_above == 0.80
        assert verdict.verdict == "likely_ai"
