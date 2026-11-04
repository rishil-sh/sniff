from ai_detector.detectors.provenance import interpret_manifest


def test_no_manifest():
    details = interpret_manifest(None, None)
    assert details["manifest_present"] is False
    assert details["points_to"] == "none"


def test_valid_ai_action():
    store = {
        "active_manifest": "urn:1",
        "validation_state": "Valid",
        "manifests": {
            "urn:1": {
                "claim_generator": "Muse Image/1.0",
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                                    ),
                                }
                            ]
                        },
                    }
                ],
            }
        },
    }
    details = interpret_manifest(store, "Valid")
    assert details["manifest_present"] is True
    assert details["signature_valid"] is True
    assert details["points_to"] == "ai"
    assert details["digital_source_type"] == "trainedAlgorithmicMedia"
    assert details["claim_generator"] == "Muse Image/1.0"


def test_valid_capture():
    store = {
        "active_manifest": "urn:1",
        "manifests": {
            "urn:1": {
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
                                    ),
                                }
                            ]
                        },
                    }
                ],
            }
        },
    }
    details = interpret_manifest(store, "Valid")
    assert details["points_to"] == "real"


def test_invalid_signature_ai_claim_still_points_to_ai():
    store = {
        "active_manifest": "urn:1",
        "manifests": {
            "urn:1": {
                "claim_generator": "OpenAI Media Service API",
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"
                                    ),
                                },
                                {"action": "c2pa.converted", "digitalSourceType": None},
                            ]
                        },
                    }
                ],
            }
        },
    }
    details = interpret_manifest(store, "Invalid")
    assert details["signature_valid"] is False
    assert details["points_to"] == "ai"
    assert details["claim_generator"] == "OpenAI Media Service API"


def test_invalid_signature_capture_is_not_trusted():
    store = {
        "active_manifest": "urn:1",
        "manifests": {
            "urn:1": {
                "assertions": [
                    {
                        "label": "c2pa.actions",
                        "data": {
                            "actions": [
                                {
                                    "action": "c2pa.created",
                                    "digitalSourceType": (
                                        "http://cv.iptc.org/newscodes/digitalsourcetype/digitalCapture"
                                    ),
                                }
                            ]
                        },
                    }
                ],
            }
        },
    }
    details = interpret_manifest(store, "Invalid")
    assert details["points_to"] == "none"


def test_generator_signature_rules():
    from ai_detector.detectors.provenance import generator_signature

    assert generator_signature({"parameters": "a cat\nSteps: 20, Sampler: Euler a"}) == "AUTOMATIC1111/Forge parameters"
    assert generator_signature({"prompt": '{"3": {"class_type": "KSampler"}}'}) == "ComfyUI workflow"
    assert generator_signature({"Software": "NovelAI"}) == "novelai"
    assert generator_signature({"CreatorTool": "Adobe Firefly"}) == "firefly"
    assert generator_signature({"Software": "Adobe Photoshop 25.0", "parameters": "no steps here"}) is None
    assert generator_signature({"Description": "painted in the style of midjourney"}) is None
