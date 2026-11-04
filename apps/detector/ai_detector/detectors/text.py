"""Text evidence: a zero-shot Llama pair plus two supervised classifiers.

Zero-shot: Binoculars (Hans et al., ICML 2024) + Fast-DetectGPT (Bao et al., ICLR 2024) +
GLTR ranks. One forward pass each through a base (observer / sampling) and Instruct
(performer / scoring) model that share a tokenizer gives all three. This has to run
locally: the Meta Model API rejects `logprobs`, and both scores need full-vocabulary
distributions.

Supervised: RoBERTa and DeBERTa classifiers trained on recent chat-model text. They catch
modern essays that the small zero-shot pair reads as human (see fusion.yaml weights).
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from html import unescape

from ai_detector.detectors.image import HF_CACHE, _device
from ai_detector.router import RoutedInput
from ai_detector.schemas import Calibration, Evidence

# Ungated bf16 mirrors of Llama-3.2-3B (base + Instruct, ~13 GB in bf16). For the original 8B pair (~32 GB, gated,
# needs HF_TOKEN) set both to meta-llama/Llama-3.1-8B(-Instruct). AI_DETECTOR_TEXT_PAIR=0 skips the pair entirely.
OBSERVER = os.environ.get("AI_DETECTOR_TEXT_OBSERVER", "mlx-community/Llama-3.2-3B-bf16")
PERFORMER = os.environ.get("AI_DETECTOR_TEXT_PERFORMER", "mlx-community/Llama-3.2-3B-Instruct")
MAX_TOKENS = 512  # per forward pass of the pair (Binoculars' max_token_observed)
# ponytail: the pair scores the first 8 x 512 tokens; raise if long documents need the zero-shot scores
MAX_PAIR_WINDOWS = 8
WINDOW_BATCH = 8  # classifier windows per forward pass

# Uncalibrated priors until scripts/eval_images.py fits Platt on held-out text (fusion.yaml `platt:`).
# Binoculars: lower = machine; centred on the Falcon-7B accuracy threshold 0.9015 (Hans et al.).
# Fast-DetectGPT: higher = machine; a z-score, centred at 1.
BINOCULARS_PRIOR = (-20.0, 0.9015)
FAST_DETECT_PRIOR = (2.0, 1.0)

# evidence id -> (repo, detector name, max tokens). Both output p(AI); licenses in NOTICE.
CLASSIFIERS = {
    "txt.fakespot": ("fakespot-ai/roberta-base-ai-text-detection-v1", "fakespot-roberta-base", 512),
    "txt.desklib": ("desklib/ai-text-detector-v1.01", "desklib-deberta-v3-large", 768),
}

_lock = threading.Lock()
_pair = None
_classifiers = None


def load_pair():
    global _pair
    if _pair is not None:
        return _pair
    with _lock:
        if _pair is not None:
            return _pair
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = _device()
        tokenizer = AutoTokenizer.from_pretrained(OBSERVER, cache_dir=HF_CACHE)
        observer, performer = (
            AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, cache_dir=HF_CACHE).to(device).eval()
            for name in (OBSERVER, PERFORMER)
        )
        _pair = (tokenizer, observer, performer, device)
        return _pair


def load_classifiers():
    global _classifiers
    if _classifiers is not None:
        return _classifiers
    with _lock:
        if _classifiers is not None:
            return _classifiers
        import torch.nn as nn
        from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel
        from transformers.modeling_outputs import SequenceClassifierOutput

        class Desklib(PreTrainedModel):  # architecture from the model card: mean-pooled DeBERTa + linear head
            config_class = AutoConfig

            def __init__(self, config):
                super().__init__(config)
                self.model = AutoModel.from_config(config)
                self.classifier = nn.Linear(config.hidden_size, 1)
                self.post_init()

            def forward(self, input_ids, attention_mask):
                h = self.model(input_ids, attention_mask=attention_mask)[0]
                m = attention_mask.unsqueeze(-1).to(h.dtype)
                return SequenceClassifierOutput(logits=self.classifier((h * m).sum(1) / m.sum(1).clamp(min=1e-9)))

        device = _device()
        models = {}
        for eid, (repo, _name, _max) in CLASSIFIERS.items():
            cls = Desklib if eid == "txt.desklib" else AutoModelForSequenceClassification
            models[eid] = (
                AutoTokenizer.from_pretrained(repo, cache_dir=HF_CACHE),
                cls.from_pretrained(repo, cache_dir=HF_CACHE).to(device).eval(),
            )
        _classifiers = (models, device)
        return _classifiers


def clean_text(t: str) -> str:
    """Fakespot's own preprocessing, verbatim from the model repo's utils.py (Apache-2.0).

    Its card: "For better performance, you should apply the `clean_text` function".
    """
    t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)
    t = re.sub(r"`[^`]*`", "", t)
    t = re.sub(r"!\[.*?\]\(.*?\)", "", t)
    t = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", t)
    t = re.sub(r"(\*\*|__)(.*?)\1", r"\2", t)
    t = re.sub(r"(\*|_)(.*?)\1", r"\2", t)
    t = re.sub(r"#+ ", "", t)
    t = re.sub(r"^>.*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"^(\s*[-*+]|\d+\.)\s+", "", t, flags=re.MULTILINE)
    t = re.sub(r"^\s*[-*_]{3,}\s*$", "", t, flags=re.MULTILINE)
    t = re.sub(r"\|.*?\|", "", t)
    t = re.sub(r"<.*?>", "", t)
    t = unescape(t)
    for ch in ("\n", "\t", "^M", "\r"):
        t = t.replace(ch, " ")
    return re.sub(" +", " ", t.replace(" ,", ","))


def score_classifiers(text: str) -> dict[str, list[float]]:
    """evidence id -> one logit per window. Windows overlap by half, so the whole text is read."""
    import torch

    models, device = load_classifiers()
    out = {}
    for eid, (tokenizer, model) in models.items():
        max_len = CLASSIFIERS[eid][2]
        enc = tokenizer(clean_text(text) if eid == "txt.fakespot" else text, truncation=True, max_length=max_len,
                        stride=max_len // 2, return_overflowing_tokens=True, padding=True, return_tensors="pt",
                        return_token_type_ids=False)
        enc.pop("overflow_to_sample_mapping", None)
        logits: list[float] = []
        with torch.inference_mode():
            for i in range(0, enc["input_ids"].shape[0], WINDOW_BATCH):
                lg = model(**{k: v[i : i + WINDOW_BATCH].to(device) for k, v in enc.items()}).logits.float()
                # fakespot: 2-way softmax {0: Human, 1: AI}; desklib: one sigmoid logit
                logits += (lg[:, 1] - lg[:, 0] if lg.shape[-1] == 2 else lg[:, 0]).tolist()
        out[eid] = logits
    return out


def token_stats(lb, li, y):
    """Per-token pieces of Binoculars, Fast-DetectGPT and GLTR. Pure.

    lb, li: [T, V] next-token logits from the observer (base) and performer (Instruct).
    y: [T] the tokens that actually follow.
    """
    logp_i = li.float().log_softmax(-1)
    p_b = lb.float().softmax(-1)
    ll = logp_i.gather(-1, y[:, None]).squeeze(-1)  # log-lik of observed tokens
    mu = (p_b * logp_i).sum(-1)  # expected log-lik under the base's distribution
    var = (p_b * logp_i**2).sum(-1) - mu**2
    ranks = (lb > lb.gather(-1, y[:, None])).sum(-1)  # GLTR: rank under the base model
    return ll, mu, var, ranks


def combine_stats(ll, mu, var, ranks):
    """Document scores from per-token stats, which concatenate across windows."""
    binoculars = (ll.mean() / mu.mean()).item()  # log PPL / log X-PPL
    fast_detect = ((ll.sum() - mu.sum()) / var.sum().sqrt()).item()
    return binoculars, fast_detect, ranks.tolist()


def pair_scores(lb, li, y):
    """Binoculars, Fast-DetectGPT and per-token GLTR ranks for one window. Pure; unit-tested."""
    return combine_stats(*token_stats(lb, li, y))


def gltr_bucket(rank: int) -> str:
    for k in (10, 100, 1000):
        if rank < k:
            return f"top-{k}"
    return "rest"


def score_text(text: str) -> dict:
    import torch

    tokenizer, observer, performer, device = load_pair()
    all_ids = tokenizer(text, return_tensors="pt").input_ids[0]  # adds BOS, as Binoculars does
    bos, body = all_ids[:1], all_ids[1:]
    # Non-overlapping 512-token windows, each restarted from BOS; per-token stats concatenate exactly.
    starts = range(0, min(len(body), MAX_TOKENS * MAX_PAIR_WINDOWS), MAX_TOKENS)
    stats, ys = [], []
    with torch.inference_mode():
        for s in starts:
            ids = torch.cat([bos, body[s : s + MAX_TOKENS]])[None].to(device)
            y = ids[0, 1:]
            stats.append(token_stats(observer(ids).logits[0, :-1], performer(ids).logits[0, :-1], y))
            ys.append(y)
    y = torch.cat(ys)
    binoculars, fast_detect, ranks = combine_stats(*(torch.cat(parts) for parts in zip(*stats)))
    return {
        "tokens": int(y.numel()),
        "windows": len(starts),
        "truncated": len(body) > y.numel(),
        "binoculars": binoculars,
        "fast_detect": fast_detect,
        "gltr": [[tokenizer.decode([t]), gltr_bucket(r)] for t, r in zip(y.tolist(), ranks)],
    }


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


async def collect(routed: RoutedInput) -> list[Evidence]:
    if routed.modality != "text" or routed.out_of_domain or not routed.text:
        return []
    tamper = _tamper_evidence(routed.text_changes)
    if os.environ.get("AI_DETECTOR_SKIP_ML"):
        return tamper
    # AI_DETECTOR_TEXT_PAIR=0 skips the Llama pair (weight 0): machines without ~32 GB, or classifier-only evals
    pair = [] if os.environ.get("AI_DETECTOR_TEXT_PAIR") == "0" else _zero_shot_evidence(routed.text)
    return tamper + pair + _classifier_evidence(routed.text)


def _tamper_evidence(changes: dict[str, int]) -> list[Evidence]:
    """Hidden or lookalike characters in prose are a known detector-evasion trick. Shown, never votes."""
    if not (changes.get("hidden") or changes.get("lookalike")):
        return []
    return [Evidence(id="txt.tamper", detector="unicode-normalizer", version="router.normalize_text", modality="text",
                     signal="rule", status="ok", points_to="none", details=changes)]


def _classifier_evidence(text: str) -> list[Evidence]:
    started = time.perf_counter()
    try:
        scores, error = score_classifiers(text), None
    except Exception as exc:  # noqa: BLE001 — e.g. offline on first run
        scores, error = {}, str(exc)
    ms = int((time.perf_counter() - started) * 1000)
    items = []
    for eid, (repo, name, max_tokens) in CLASSIFIERS.items():
        common = dict(id=eid, detector=name, version=repo, modality="text", signal="learned", weight=0.5, latency_ms=ms)
        if error:
            items.append(Evidence(status="error", details={"error": error}, **common))
            continue
        windows = scores[eid]
        logit = sum(windows) / len(windows)  # mean window logit = the document score
        p = _sigmoid(logit)
        details = {"max_tokens": max_tokens, "windows": len(windows), "window_logits": [round(w, 3) for w in windows]}
        items.append(Evidence(status="ok", raw_score=logit, p_ai=p, points_to="ai" if p >= 0.5 else "real",
                              details=details, calibration=Calibration(set="uncalibrated-sigmoid"), **common))
    return items


def _zero_shot_evidence(text: str) -> list[Evidence]:
    common = dict(modality="text", signal="learned")
    ids = (
        ("txt.binoculars", "binoculars", 0.5),
        ("txt.fastdetect", "fast-detectgpt-analytic", 0.5),
        ("txt.gltr", "gltr-token-rank", 0.0),
    )
    version = f"{OBSERVER}+{PERFORMER}"
    started = time.perf_counter()
    try:
        s = score_text(text)
    except Exception as exc:  # noqa: BLE001 — e.g. gated weights without HF_TOKEN
        ms = int((time.perf_counter() - started) * 1000)
        return [
            Evidence(id=i, detector=d, version=version, status="error", weight=w,
                     details={"error": str(exc)}, latency_ms=ms, **common)
            for i, d, w in ids
        ]
    ms = int((time.perf_counter() - started) * 1000)
    details = {k: s[k] for k in ("tokens", "windows", "truncated", "binoculars", "fast_detect")}
    details |= {"observer": OBSERVER, "performer": PERFORMER}
    p_bino = _sigmoid(BINOCULARS_PRIOR[0] * (s["binoculars"] - BINOCULARS_PRIOR[1]))
    p_fast = _sigmoid(FAST_DETECT_PRIOR[0] * (s["fast_detect"] - FAST_DETECT_PRIOR[1]))
    shares = {b: sum(t[1] == b for t in s["gltr"]) / len(s["gltr"]) for b in ("top-10", "top-100", "top-1000", "rest")}
    scored = [
        (ids[0], s["binoculars"], p_bino, details),
        (ids[1], s["fast_detect"], p_fast, details),
        # GLTR is for the visual only, not a score (build guide §5)
        (ids[2], None, None, {"ranked_by": OBSERVER, "shares": shares, "tokens": s["gltr"]}),
    ]
    return [
        Evidence(
            id=i, detector=d, version=version, status="ok", raw_score=raw, p_ai=p,
            points_to="none" if p is None else ("ai" if p >= 0.5 else "real"),
            weight=w, details=det,
            calibration=None if p is None else Calibration(set="uncalibrated-sigmoid"),
            latency_ms=ms, **common,
        )
        for (i, d, w), raw, p, det in scored
    ]
