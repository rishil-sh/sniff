"""Speech (English): AudioSeal watermark, then three learned detectors over 4 s windows.

  aud.w2vbert  frozen w2v-BERT 2.0 (Meta, MIT) + our linear probe (ai_detector/weights/w2vbert_probe.npz),
               trained by scripts/train_audio_probe.py on ElevenLabs, OpenAI and open TTS vs human speech.
               The only detector that votes (fusion.yaml weight 1.0). Shipped probe: en-L18-C0.01 (v6).
  aud.dfarena  DF Arena 500M (non-commercial licence). Evidence only (weight 0): flagged 37% of
               Common Voice humans as AI and missed half of ElevenLabs v4 in scripts/eval_audio.py.
  aud.mambo    XLSR-MamBo (MIT). Evidence only (weight 0): 17-37% false positives on human speech.
AudioSeal reports only its own watermark and is never fused as a score. Results: docs/audio-detection.md.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
import time
import types
from pathlib import Path
from unittest import mock

import numpy as np

from ai_detector.router import RoutedInput, ffmpeg_exe
from ai_detector.schemas import Calibration, Evidence

SR = 16_000
FFMPEG = ffmpeg_exe()
HOP_S = 2.0
MIN_RMS = 1e-3  # windows quieter than this are silence; scoring them is noise
NARROWBAND_HZ = 4_000
NARROWBAND_MAX_ENERGY = 1e-4  # energy share above 4 kHz below this = 8 kHz phone-band audio
WM_MIN_SEGMENT_S = 1.0

MAMBO_REPO = "SpeechAntiSpoofingBenchmarks/XLSR-MamBo"
MAMBO_REV = "cfc2af1eb666b4a1742117cf80a0a77c5b0364d1"
MAMBO_VERSION = f"{MAMBO_REPO}@{MAMBO_REV[:7]}"
MAMBO_WINDOW = 66_800  # 4.175 s, the model's training crop
DFA_REPO = "Speech-Arena-2025/DF_Arena_500M_V_1"
DFA_REV = "8258fa8e74ff9b8ad20d4c939c1a7f694a6e4080"
DFA_VERSION = f"{DFA_REPO}@{DFA_REV[:7]}"
DFA_WINDOW = 64_600  # 4.04 s
SEAL_VERSION = "audioseal@0.2.0/audioseal_detector_16bits"
W2VBERT_REPO = "facebook/w2v-bert-2.0"
W2VBERT_REV = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
W2VBERT_WINDOW = 64_000  # 4 s
PROBE_PATH = Path(__file__).resolve().parent.parent / "weights" / "w2vbert_probe.npz"
# Mirror fusion.yaml so the evidence shows the weight that was applied (as the image detectors do).
WEIGHTS = {"aud.w2vbert": 1.0, "aud.dfarena": 0.0, "aud.mambo": 0.0}

_lock = threading.Lock()
_models: dict[str, object] = {}


def _device():
    import torch

    # ponytail: MPS skipped; MamBo's pure-torch SSD path is only parity-checked on CPU/CUDA
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(data: bytes, suffix: str = ".wav") -> np.ndarray:
    """Any ffmpeg-readable audio -> float32 mono 16 kHz. Temp file, not a pipe: m4a needs seeking."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        out = subprocess.run(
            [FFMPEG, "-v", "error", "-i", tmp.name, "-ac", "1", "-ar", str(SR), "-f", "f32le", "pipe:1"],
            capture_output=True,
            timeout=120,
            check=True,
        )
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def to_wav16(data: bytes, suffix: str = ".wav") -> bytes:
    """16-bit PCM mono 16 kHz WAV, the format Muse Voice Transcribe takes. Metadata stripped."""
    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
        tmp.write(data)
        tmp.flush()
        out = subprocess.run(
            [FFMPEG, "-v", "error", "-i", tmp.name, "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le",
             "-map_metadata", "-1", "-f", "wav", "pipe:1"],
            capture_output=True,
            timeout=120,
            check=True,
        )
    return out.stdout


def windows(wav: np.ndarray, size: int, hop: int = int(HOP_S * SR)) -> list[tuple[float, np.ndarray]]:
    """(start_s, chunk) pairs; short clips tile-repeat to `size` as both models were trained. Pure; unit-tested."""
    if wav.size == 0:
        return []
    if wav.size < size:
        return [(0.0, np.tile(wav, size // wav.size + 1)[:size])]
    starts = list(range(0, wav.size - size + 1, hop))
    if starts[-1] + size < wav.size:
        starts.append(wav.size - size)  # cover the tail
    voiced = [(s / SR, wav[s : s + size]) for s in starts if np.sqrt(np.mean(wav[s : s + size] ** 2)) >= MIN_RMS]
    return voiced or [(s / SR, wav[s : s + size]) for s in starts]


def aggregate(logits: list[float]) -> dict[str, float]:
    """Clip score = median window logit; top-quartile mean flags partial fakes. Pure; unit-tested."""
    arr = np.sort(np.asarray(logits, dtype=np.float64))
    top = arr[-max(1, len(arr) // 4) :]
    return {"median": float(np.median(arr)), "top25_mean": float(top.mean())}


def narrowband(wav: np.ndarray) -> bool:
    """True for phone-band audio (nothing above 4 kHz), where both detectors are untested. Pure; unit-tested."""
    if wav.size < SR // 10:
        return False
    power = np.abs(np.fft.rfft(wav)) ** 2
    freqs = np.fft.rfftfreq(wav.size, 1 / SR)
    total = power.sum()
    return bool(total > 0 and power[freqs > NARROWBAND_HZ].sum() / total < NARROWBAND_MAX_ENERGY)


def segments(mask: np.ndarray, rate: int, min_s: float) -> list[dict[str, float]]:
    """Runs of True at least `min_s` long, as [{start, end}] seconds. Pure; unit-tested."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [
        {"start": round(a / rate, 2), "end": round(b / rate, 2)}
        for a, b in zip(edges[::2], edges[1::2], strict=True)
        if (b - a) / rate >= min_s
    ]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def load_mambo():
    if "mambo" in _models:
        return _models["mambo"]
    with _lock:
        if "mambo" in _models:
            return _models["mambo"]
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            MAMBO_REPO,
            revision=MAMBO_REV,
            allow_patterns=["_net.py", "_hydra/*", "mamba_ssm/**", "*.pt", "LICENSE.source"],
        )
        stubbed = False
        try:
            import triton  # noqa: F401
        except ImportError:
            # ponytail: macOS has no triton; the vendored mamba_ssm only touches it at import time,
            # and pure_torch=True never calls a kernel. Stub just what import-time code reads.
            stub = mock.MagicMock(name="triton")
            stub.__version__ = "3.0.0"
            stub.Config = lambda *a, **kw: types.SimpleNamespace(**kw)
            sys.modules["triton"], sys.modules["triton.language"] = stub, stub.language
            stubbed = True
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            import _net  # the repo's network module
        finally:
            if stubbed:  # a lingering fake triton breaks transformers' import checks (DF Arena)
                sys.modules.pop("triton", None)
                sys.modules.pop("triton.language", None)

        device = _device()
        net = _net.XLSRMamBo(pure_torch=True)
        net.load_state_dict(_net.load_state(os.path.join(path, "mambo3_da0_best0.pt")), strict=True)
        _models["mambo"] = (net.to(device).eval(), device)
        return _models["mambo"]


def score_mambo(chunks: list[np.ndarray]) -> list[float]:
    """Logit that each window is synthetic (spoof minus bona fide)."""
    import torch

    net, device = load_mambo()
    out: list[float] = []
    with torch.inference_mode():
        for i in range(0, len(chunks), 8):
            x = torch.from_numpy(np.stack(chunks[i : i + 8])).to(device)
            logits = net(x).float().cpu()
            out.extend((logits[:, 0] - logits[:, 1]).tolist())
    return out


def load_dfarena():
    if "dfarena" in _models:
        return _models["dfarena"]
    with _lock:
        if "dfarena" in _models:
            return _models["dfarena"]
        from transformers import AutoModel

        device = _device()
        # Remote code reviewed at this revision: plain torch/einops, no I/O beyond the XLS-R config.
        model = AutoModel.from_pretrained(DFA_REPO, revision=DFA_REV, trust_remote_code=True)
        _models["dfarena"] = (model.to(device).eval(), device)
        return _models["dfarena"]


def score_dfarena(chunks: list[np.ndarray]) -> list[float]:
    """Logit that each window is synthetic. The model's forward takes one 1-D waveform (id 0 = spoof)."""
    import torch

    model, device = load_dfarena()
    out: list[float] = []
    with torch.inference_mode():
        for chunk in chunks:
            logits = model(input_values=torch.from_numpy(chunk).to(device))["logits"].float().cpu()[0]
            out.append(float(logits[0] - logits[1]))
    return out


def load_audioseal():
    if "seal" in _models:
        return _models["seal"]
    with _lock:
        if "seal" not in _models:
            from audioseal import AudioSeal

            _models["seal"] = AudioSeal.load_detector("audioseal_detector_16bits").eval()
        return _models["seal"]


def detect_audioseal(wav: np.ndarray) -> np.ndarray:
    """Per-sample watermark probability. 30 s chunks keep memory flat on 10-minute clips."""
    import torch

    det = load_audioseal()
    parts = []
    with torch.inference_mode():
        for i in range(0, wav.size, 30 * SR):
            x = torch.from_numpy(wav[i : i + 30 * SR]).reshape(1, 1, -1)
            probs, _ = det(x, SR)
            parts.append(probs[0, 1].numpy())
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _embed_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # features only; 3.5x faster than CPU on Apple silicon
        return torch.device("mps")
    return torch.device("cpu")


def load_w2vbert():
    if "w2vbert" in _models:
        return _models["w2vbert"]
    with _lock:
        if "w2vbert" not in _models:
            from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

            device = _embed_device()
            fe = AutoFeatureExtractor.from_pretrained(W2VBERT_REPO, revision=W2VBERT_REV)
            model = Wav2Vec2BertModel.from_pretrained(W2VBERT_REPO, revision=W2VBERT_REV).to(device).eval()
            _models["w2vbert"] = (fe, model, device)
        return _models["w2vbert"]


def embed_w2vbert(chunks: list[np.ndarray], batch: int = 8) -> np.ndarray:
    """Frozen w2v-BERT 2.0: time-mean of every hidden layer -> float32 [n, 25, 1024].
    Shared by scripts/train_audio_probe.py and the detector, so train and serve features match."""
    import torch

    fe, model, device = load_w2vbert()
    out = []
    with torch.inference_mode():
        for i in range(0, len(chunks), batch):
            x = fe(list(chunks[i : i + batch]), sampling_rate=SR, return_tensors="pt")
            hidden = model(**{k: v.to(device) for k, v in x.items()}, output_hidden_states=True).hidden_states
            out.append(torch.stack(hidden, 1).mean(2).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 25, 1024), np.float32)


def load_probe() -> dict:
    if "probe" not in _models:
        with np.load(PROBE_PATH) as z:
            _models["probe"] = {k: z[k] for k in z.files}
    return _models["probe"]


def probe_features(emb: np.ndarray, probe: dict) -> np.ndarray:
    """[n, 25, 1024] embeddings -> the probe's input: one layer, a concatenation ("cat"), or a mean ("avg").
    Must match pooled() in scripts/train_audio_probe.py."""
    if "layers" not in probe:  # en-v2 format: a single `layer`
        return emb[:, int(probe["layer"])]
    sel = emb[:, [int(i) for i in probe["layers"]]]
    return sel.mean(1) if str(probe["pool"]) == "avg" else sel.reshape(len(emb), -1)


def score_w2vbert(chunks: list[np.ndarray]) -> list[float]:
    """Logit that each window is synthetic: standardised probe features . coef + intercept."""
    probe = load_probe()
    feats = probe_features(embed_w2vbert(chunks), probe)
    return (((feats - probe["mean"]) / probe["scale"]) @ probe["coef"] + probe["intercept"]).tolist()


def _learned(evidence_id, detector, version, score_fn, wav, size, *, issue: str | None,
             calibration: Calibration | None = None) -> Evidence:
    started = time.perf_counter()
    base = {"id": evidence_id, "detector": detector, "version": version, "modality": "audio",
            "signal": "learned", "weight": WEIGHTS[evidence_id]}
    if issue:
        return Evidence(**base, status="out_of_domain", details={"reason": issue})
    try:
        wins = windows(wav, size)
        logits = score_fn([chunk for _, chunk in wins])
        agg = aggregate(logits)
        p_ai = _sigmoid(agg["median"])
        return Evidence(
            **base,
            status="ok",
            raw_score=agg["median"],
            p_ai=p_ai,
            points_to="ai" if p_ai >= 0.5 else "real",
            details={
                "seconds": round(wav.size / SR, 2),
                "windows": len(wins),
                "window_s": round(size / SR, 3),
                "hop_s": HOP_S,
                "top25_p_ai": round(_sigmoid(agg["top25_mean"]), 4),
                "timeline": [{"t": round(t, 2), "p_ai": round(_sigmoid(x), 4)} for (t, _), x in zip(wins, logits)],
            },
            calibration=calibration or Calibration(set="uncalibrated-sigmoid"),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:  # noqa: BLE001
        return Evidence(**base, status="error", details={"error": str(exc)},
                        latency_ms=int((time.perf_counter() - started) * 1000))


def _watermark(wav: np.ndarray) -> Evidence:
    started = time.perf_counter()
    base = {"id": "aud.audioseal", "detector": "audioseal", "version": SEAL_VERSION, "modality": "audio",
            "signal": "watermark"}
    try:
        probs = detect_audioseal(wav)
        segs = segments(probs > 0.5, SR, WM_MIN_SEGMENT_S)
        details = {
            "scheme": "AudioSeal",
            "detected": bool(segs),
            "score": round(float((probs > 0.5).mean()) if probs.size else 0.0, 4),
            "segments": segs,
            "note": "Detects only Meta AudioSeal marks. No mark says nothing about other generators.",
        }
        return Evidence(**base, status="ok", points_to="ai" if segs else "none", details=details,
                        latency_ms=int((time.perf_counter() - started) * 1000))
    except Exception as exc:  # noqa: BLE001
        return Evidence(**base, status="error", details={"error": str(exc)},
                        latency_ms=int((time.perf_counter() - started) * 1000))


SUFFIX = {"audio/wav": ".wav", "audio/flac": ".flac", "audio/mpeg": ".mp3", "audio/mp4": ".m4a"}


async def collect(routed: RoutedInput) -> list[Evidence]:
    if routed.modality != "audio" or routed.data is None or routed.out_of_domain:
        return []
    if os.environ.get("AI_DETECTOR_SKIP_ML"):
        return []
    try:
        wav = decode(routed.data, SUFFIX.get(routed.mime or "", ".wav"))
    except Exception as exc:  # noqa: BLE001
        return [Evidence(id="aud.decode", detector="ffmpeg", version="system", modality="audio",
                         signal="learned", status="error", details={"error": str(exc)})]
    issue = "narrowband (phone-quality, <= 8 kHz) audio; the detectors are untested there" if narrowband(wav) else None
    # ponytail: sequential; ~300 windows on a 10-minute clip is slow on CPU, fine on the L40S
    return [
        _watermark(wav),
        _learned("aud.mambo", "xlsr-mambo", MAMBO_VERSION, score_mambo, wav, MAMBO_WINDOW, issue=issue),
        _learned("aud.dfarena", "df-arena-500m", DFA_VERSION, score_dfarena, wav, DFA_WINDOW, issue=issue),
        _probe(wav, issue),
    ]


def _probe(wav: np.ndarray, issue: str | None) -> Evidence:
    if not PROBE_PATH.exists():
        return Evidence(id="aud.w2vbert", detector="w2v-bert-2.0-probe", version="untrained", modality="audio",
                        signal="learned", status="skipped", details={"reason": "run scripts/train_audio_probe.py"})
    tag = str(load_probe()["tag"])
    version = f"{W2VBERT_REPO}@{W2VBERT_REV[:7]}+probe:{tag}"
    # Class-balanced logistic output, threshold picked by leave-one-generator-out CV; not Platt-scaled.
    calibration = Calibration(set=f"probe {tag}: English, leave-one-generator-out CV (docs/audio-detection.md)")
    return _learned("aud.w2vbert", "w2v-bert-2.0-probe", version, score_w2vbert, wav, W2VBERT_WINDOW,
                    issue=issue, calibration=calibration)
