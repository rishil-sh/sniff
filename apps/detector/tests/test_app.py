from fastapi.testclient import TestClient

from ai_detector.app import app
from ai_detector.router import TEXT_MIN_TOKENS


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_analyze_text_endpoint():
    client = TestClient(app)
    text = " ".join(["token"] * TEXT_MIN_TOKENS)
    r = client.post("/v1/analyze/text", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["verdict"] == "inconclusive"
    assert body["modality"] == "text"
    assert body["report"]["model"] == "stub"


def test_analyze_png_upload():
    client = TestClient(app)
    png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    r = client.post(
        "/v1/analyze",
        files={"file": ("shot.png", png, "image/png")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["modality"] == "image"
    assert body["verdict"] == "inconclusive"
