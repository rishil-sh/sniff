"""Download every model the detector uses, so the first request doesn't, and images stay lean.

    uv run ai-detector-download            # locally, into AI_DETECTOR_CACHE (default data/models)
    docker compose run --rm models         # in Docker, into the `models` volume

Each step calls the detector's own loader, so this always fetches exactly what the code loads. Re-running is cheap:
cached files are skipped. AI_DETECTOR_TEXT_PAIR=0 skips the ~13 GB Llama pair (weight 0, supporting evidence only).
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable


def _text_pair() -> None:
    # Download only: loading two 3B models just to cache them would need ~13 GB of RAM.
    from huggingface_hub import snapshot_download

    from ai_detector.detectors.image import HF_CACHE
    from ai_detector.detectors.text import OBSERVER, PERFORMER

    for repo in dict.fromkeys((OBSERVER, PERFORMER)):
        snapshot_download(repo, cache_dir=HF_CACHE, allow_patterns=["*.json", "*.safetensors", "tokenizer*"])


def _steps() -> list[tuple[str, Callable[[], object]]]:
    from ai_detector.detectors import audio, image, text

    steps: list[tuple[str, Callable[[], object]]] = [
        ("image: Community Forensics (commfor)", image.load_commfor),
        ("image: Cozzolino CLIP probe + ViT-L-14", image.load_clip),
        ("text: Fakespot + Desklib classifiers", text.load_classifiers),
        ("audio: w2v-BERT 2.0", audio.load_w2vbert),
        ("audio: DF Arena", audio.load_dfarena),
        ("audio: XLSR-MamBo", audio.load_mambo),
        ("audio: AudioSeal detector", audio.load_audioseal),
    ]
    if os.environ.get("AI_DETECTOR_TEXT_PAIR") != "0":
        steps.append((f"text: zero-shot pair {text.OBSERVER} + {text.PERFORMER}", _text_pair))
    return steps


def main() -> None:
    failed = []
    for name, step in _steps():
        start = time.monotonic()
        print(f"→ {name}", flush=True)
        try:
            step()
        except Exception as exc:  # keep going: one unreachable host shouldn't block the other models
            failed.append(name)
            print(f"  ✗ {type(exc).__name__}: {exc}", flush=True)
            continue
        print(f"  ✓ {time.monotonic() - start:.0f}s", flush=True)
    if failed:
        print(f"\n{len(failed)} model(s) failed: {', '.join(failed)}. Re-run to retry.", flush=True)
        sys.exit(1)
    print("\nAll models ready.", flush=True)


if __name__ == "__main__":
    main()
