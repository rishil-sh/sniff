from ai_detector.schemas import Bands, FusionTrace, Verdict


def test_verdict_json_shape():
    verdict = Verdict(
        request_id="9f2c0001",
        modality="image",
        sha256="b" * 64,
        verdict="inconclusive",
        p_ai=None,
        bands=Bands(),
        decided_by="fusion",
        evidence=[],
        fusion=FusionTrace(policy="fusion.yaml@v0.1", weights={}),
    )
    dumped = verdict.model_dump(mode="json")
    assert dumped["verdict"] == "inconclusive"
    assert dumped["bands"]["likely_ai_above"] == 0.80
    assert "generated_at" in dumped
    roundtrip = Verdict.model_validate(dumped)
    assert roundtrip.request_id == "9f2c0001"
