import pytest

from ai_detector.detectors import image as image_mod
from ai_detector.detectors import text as text_mod


@pytest.fixture(autouse=True)
def _no_spai_subprocess(monkeypatch):
    # Tests never launch the external SPAI venv; tests that need it stub score_spai.
    monkeypatch.setattr(image_mod, "spai_ready", lambda: False)
    # Nor do they load commfor through the head; 0.5 sits in the chance band and never votes.
    monkeypatch.setattr(image_mod, "score_head", lambda _im: (0.0, 0.5))


@pytest.fixture(autouse=True)
def _no_llama_download(monkeypatch):
    # Tests never load the real Llama pair; tests that need scores stub score_text.
    def refuse():
        raise RuntimeError("tests must stub text_mod.score_text")

    monkeypatch.setattr(text_mod, "load_pair", refuse)
    monkeypatch.setattr(text_mod, "load_classifiers", refuse)


@pytest.fixture(autouse=True)
def _no_muse_calls(monkeypatch):
    # .env may hold a real key; tests never spend credits. Tests that need Muse stub muse_client.
    from ai_detector.muse import client as muse

    monkeypatch.setattr(muse, "muse_client", lambda: None)
