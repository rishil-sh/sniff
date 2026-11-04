"""Community Forensics + Cozzolino CLIP linear probe + SPAI (spectral, any resolution)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path

from ai_detector import CACHE_DIR, HF_CACHE
from ai_detector.router import RoutedInput
from ai_detector.schemas import Calibration, Evidence

COMMFOR_REPO = "OwensLab/commfor-model-384"
COMMFOR_VERSION = f"{COMMFOR_REPO}@main"
CLIP_VERSION = "clipdet_latent10k_plus"
CLIP_WEIGHTS_URL = "https://www.grip.unina.it/download/prog/DMimageDetection/weights_clipdet.zip"
SPAI_DIR = CACHE_DIR / "spai"
SPAI_VERSION = "mever-team/spai@8ff7b3b"
SPAI_RUNNER = Path(__file__).with_name("spai_runner.py")

_lock = threading.Lock()
_commfor = None
_clip = None


def _device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _open_rgb(data: bytes):
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def load_commfor():
    global _commfor
    if _commfor is not None:
        return _commfor
    with _lock:
        if _commfor is not None:
            return _commfor
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from torchvision import transforms as T

        from ai_detector.detectors.commfor import ViTClassifier

        device = _device()
        model = ViTClassifier(model_size="small", input_size=384, patch_size=16)
        weights = load_file(hf_hub_download(COMMFOR_REPO, "model.safetensors", cache_dir=HF_CACHE))
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        transform = T.Compose(
            [
                T.Resize(440),
                T.CenterCrop(384),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        _commfor = (model, transform, device)
        return _commfor


def score_commfor(image) -> tuple[float, float]:
    import torch

    model, transform, device = load_commfor()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logit = model(tensor).reshape(-1)[0].float().cpu()
        p_ai = torch.sigmoid(logit).item()
    return float(logit), float(p_ai)


HEAD_PATH = Path(__file__).with_name("commfor_head.npz")
HEAD_VERSION = "commfor-head@eval_v2-faces-l2_3-rw_2"
_head = None


def score_head(image) -> tuple[float, float]:
    """Logistic head on frozen commfor features, trained on originals + screenshot copies
    (scripts/train_head.py, in git history at bde0082) so zoomed screenshots of real photos stop reading as AI."""
    import numpy as np
    import torch

    global _head
    if _head is None:
        _head = dict(np.load(HEAD_PATH))
    model, transform, device = load_commfor()
    with torch.inference_mode():
        x = transform(image).unsqueeze(0).to(device)
        feats = model.vit.forward_head(model.vit.forward_features(x), pre_logits=True)
    f = feats.reshape(-1).float().cpu().numpy()
    z = float(((f - _head["mean"]) / _head["std"]) @ _head["w"] + _head["b"])
    return z, float(1 / (1 + np.exp(-z)))


def _clip_weights_path() -> Path:
    dest = CACHE_DIR / "clipdet_latent10k_plus" / "weights.pth"
    if dest.exists() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    zip_path = CACHE_DIR / "weights_clipdet.zip"
    if not zip_path.exists() or zip_path.stat().st_size < 1000:
        urllib.request.urlretrieve(CLIP_WEIGHTS_URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = [
            n
            for n in zf.namelist()
            if n.endswith("clipdet_latent10k_plus/weights.pth")
            or n.endswith("weights.pth")
            and "clipdet_latent10k_plus" in n
        ]
        if not names:
            names = [n for n in zf.namelist() if n.endswith("weights.pth")]
        if not names:
            raise FileNotFoundError("clipdet_latent10k_plus/weights.pth missing from GRIP zip")
        dest.write_bytes(zf.read(names[0]))
    return dest


def load_clip():
    global _clip
    if _clip is not None:
        return _clip
    with _lock:
        if _clip is not None:
            return _clip
        import open_clip
        import torch
        import torch.nn as nn
        from huggingface_hub import hf_hub_download
        from torchvision import transforms as T

        device = _device()
        # Cozzolino clipdet_latent10k_plus: opencliplinearnext_clipL14commonpool
        weight = hf_hub_download(
            "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K",
            "open_clip_pytorch_model.bin",
            cache_dir=HF_CACHE,
        )
        backbone = open_clip.create_model("ViT-L-14", pretrained=weight)
        num_features = backbone.visual.proj.shape[0]
        backbone.visual.proj = None
        head = nn.Linear(num_features, 1)
        ckpt = torch.load(_clip_weights_path(), map_location="cpu")
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()}
        head_state = {
            k.removeprefix("fc."): v
            for k, v in state.items()
            if "fc" in k or k.startswith("weight") or k.startswith("bias")
        }
        if "weight" not in head_state:
            # keys like fc.weight
            head_state = {k.split(".")[-1]: v for k, v in state.items() if k.endswith("weight") or k.endswith("bias")}
        head.load_state_dict({"weight": head_state["weight"], "bias": head_state["bias"]})
        backbone.to(device).eval()
        head.to(device).eval()
        transform = T.Compose(
            [
                T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )
        _clip = (backbone, head, transform, device)
        return _clip


def score_clip(image) -> tuple[float, float]:
    import torch

    backbone, head, transform, device = load_clip()
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        features = backbone.encode_image(tensor, normalize=True)
        logit = head(features).reshape(-1)[0].float().cpu()
        p_ai = torch.sigmoid(logit).item()
    return float(logit), float(p_ai)


def spai_ready() -> bool:
    return (SPAI_DIR / "weights" / "spai.pth").exists() and (SPAI_DIR / ".venv" / "bin" / "python").exists()


def score_spai(image) -> tuple[float, float]:
    # shortcut: one subprocess + model load per image (~seconds); keep a worker alive if latency matters
    with tempfile.NamedTemporaryFile(suffix=".png") as fh:
        image.save(fh, format="PNG")
        fh.flush()
        out = subprocess.run(
            [str(SPAI_DIR / ".venv" / "bin" / "python"), "-P", str(SPAI_RUNNER), fh.name],
            cwd=SPAI_DIR,
            env={**os.environ, "PYTHONPATH": str(SPAI_DIR)},
            capture_output=True,
            text=True,
            timeout=300,
            check=True,
        )
    row = json.loads(out.stdout.strip().splitlines()[-1])
    return float(row["logit"]), float(row["p_ai"])


def _one(
    *,
    evidence_id: str,
    detector: str,
    version: str,
    routed: RoutedInput,
    score_fn,
    weight: float,
    started: float,
) -> Evidence:
    try:
        image = _open_rgb(routed.data or b"")
        raw, p_ai = score_fn(image)
        return Evidence(
            id=evidence_id,
            detector=detector,
            version=version,
            modality="image",
            signal="learned",
            status="ok",
            raw_score=raw,
            p_ai=p_ai,
            points_to="ai" if p_ai >= 0.5 else "real",
            weight=weight,
            details={"input_size": {"img.head": 384, "img.commfor": 384, "img.clip": 224}.get(evidence_id, "native")},
            # the head is a logistic fit on eval_v2, so its sigmoid is already its calibration
            calibration=Calibration(set="eval_v2" if evidence_id == "img.head" else "uncalibrated-sigmoid"),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return Evidence(
            id=evidence_id,
            detector=detector,
            version=version,
            modality="image",
            signal="learned",
            status="error",
            weight=weight,
            details={"error": str(exc)},
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


MIN_SIDE = 256


def domain_issue(image) -> str | None:
    """Why the photo-trained detectors can't judge this image, if they can't. Pure; unit-tested."""
    if image.mode in ("P", "PA", "1"):
        return f"{image.mode}-mode palette image: quantization erased the pixel traces the detectors read"
    if "transparency" in image.info or (image.mode in ("RGBA", "LA") and image.getchannel("A").getextrema()[0] < 255):
        return "transparent pixels: a cut-out graphic, not a photo-like frame"
    if min(image.size) < MIN_SIDE:
        return f"smallest side {min(image.size)} px < {MIN_SIDE} px"
    return None


async def collect(routed: RoutedInput) -> list[Evidence]:
    if routed.modality != "image" or routed.data is None or routed.out_of_domain:
        return []
    if os.environ.get("AI_DETECTOR_SKIP_ML"):
        return []
    from PIL import Image

    try:
        issue = domain_issue(Image.open(io.BytesIO(routed.data)))
    except Exception:  # noqa: BLE001
        issue = None  # _one reports the decode error per detector
    if issue:
        return [
            Evidence(
                id=evidence_id,
                detector=detector,
                version=version,
                modality="image",
                signal="learned",
                status="out_of_domain",
                weight=weight,
                details={"reason": issue},
            )
            for evidence_id, detector, version, weight in (
                ("img.head", "commfor-screenshot-head", HEAD_VERSION, 1.0),
                ("img.commfor", "community-forensics-vits-384", COMMFOR_VERSION, 0.6),
                ("img.clip", "cozzolino-clip-linear", CLIP_VERSION, 0.4),
                ("img.spai", "spai-spectral", SPAI_VERSION, 0.5),
            )
        ]
    head = _one(
        evidence_id="img.head",
        detector="commfor-screenshot-head",
        version=HEAD_VERSION,
        routed=routed,
        score_fn=score_head,
        weight=1.0,
        started=time.perf_counter(),
    )
    commfor = _one(
        evidence_id="img.commfor",
        detector="community-forensics-vits-384",
        version=COMMFOR_VERSION,
        routed=routed,
        score_fn=score_commfor,
        weight=0.6,
        started=time.perf_counter(),
    )
    clip = _one(
        evidence_id="img.clip",
        detector="cozzolino-clip-linear",
        version=CLIP_VERSION,
        routed=routed,
        score_fn=score_clip,
        weight=0.4,
        started=time.perf_counter(),
    )
    if not spai_ready():
        spai = Evidence(
            id="img.spai",
            detector="spai-spectral",
            version=SPAI_VERSION,
            modality="image",
            signal="learned",
            status="skipped",
            details={"reason": "not installed; run scripts/setup_spai.sh"},
        )
    else:
        spai = _one(
            evidence_id="img.spai",
            detector="spai-spectral",
            version=SPAI_VERSION,
            routed=routed,
            score_fn=score_spai,
            weight=0.5,
            started=time.perf_counter(),
        )
    return [head, commfor, clip, spai]
