from __future__ import annotations

import math
from pathlib import Path

import yaml

from ai_detector.router import RoutedInput
from ai_detector.schemas import Bands, Calibration, Evidence, FusionTrace, Verdict

POLICY_PATH = Path(__file__).with_name("fusion.yaml")


def load_policy(path: Path | None = None) -> dict:
    with (path or POLICY_PATH).open() as fh:
        return yaml.safe_load(fh) or {}


def _sigmoid(x: float) -> float:
    return 0.5 * (1.0 + math.tanh(0.5 * x))  # = 1/(1+e^-x), without overflow for large |x|


def stacked_p_ai(stacker: dict, evidence: list[Evidence]) -> float | None:
    """Logistic stacker over raw detector scores (fusion.yaml `stackers:`, fit by scripts/eval_images.py).

    None when an input is missing, failed, or comes from a model version the stacker wasn't fit on:
    coefficients fit on one Llama pair are meaningless for another.
    """
    by_id = {e.id: e for e in evidence if e.status == "ok" and e.raw_score is not None}
    versions = stacker.get("versions") or {}
    z = float(stacker["intercept"])
    for eid, coef in stacker["coef"].items():
        e = by_id.get(eid)
        if e is None or (eid in versions and e.version != versions[eid]):
            return None
        z += float(coef) * float(e.raw_score)
    return _sigmoid(z)


def fuse(
    routed: RoutedInput,
    evidence: list[Evidence],
    *,
    request_id: str,
    policy: dict | None = None,
) -> Verdict:
    policy = policy if policy is not None else load_policy()
    version = str(policy.get("version", "0.1"))
    stacker = (policy.get("stackers") or {}).get(routed.modality)
    p_stack = stacked_p_ai(stacker, evidence) if stacker else None
    # A stacker carries its own bands, set at a fixed FPR on held-out human text; they only fit its p_ai.
    bands_raw = (stacker.get("bands") if p_stack is not None else None) or policy.get("bands") or {}
    # Short inputs are noisier, so the stacker may set stricter bands below a token count.
    short = (stacker or {}).get("short_bands") if p_stack is not None else None
    if short and routed.token_count is not None and routed.token_count < int(short["below_tokens"]):
        bands_raw = short
    bands = Bands(
        no_ai_signal_below=float(bands_raw.get("no_ai_signal_below", 0.20)),
        likely_ai_above=float(bands_raw.get("likely_ai_above", 0.80)),
    )
    weights = {str(k): float(v) for k, v in (policy.get("weights") or {}).items()}
    max_disagreement = float(policy.get("max_disagreement", 0.6))
    # shortcut: uncalibrated CLIP sits near 0.5 on real photos; ignore until Platt
    chance_band = float(policy.get("chance_band", 0.15))

    # Platt fits from scripts/eval_images.py replace raw sigmoids: p = sigmoid(a * raw + b).
    for e in evidence:
        fit = (policy.get("platt") or {}).get(e.id)
        if fit and e.status == "ok" and e.raw_score is not None:
            e.p_ai = _sigmoid(float(fit["a"]) * e.raw_score + float(fit["b"]))
            e.points_to = "ai" if e.p_ai >= 0.5 else "real"  # detector set it from the raw sigmoid
            e.calibration = Calibration(
                set=str(fit.get("set")), auroc=fit.get("auroc"), tpr_at_1pct_fpr=fit.get("tpr_at_1pct_fpr")
            )

    declared = [
        e for e in evidence if e.status == "ok" and e.signal in ("provenance", "watermark") and e.points_to == "ai"
    ]
    capture = [e for e in evidence if e.status == "ok" and e.signal == "provenance" and e.points_to == "real"]

    def weight_of(e: Evidence) -> float:
        # A video soundtrack is usually music, effects or non-English speech: the speech probe is measured on clean
        # English speech only (docs/audio-detection.md) and called Kling's music track "real" (0.06). Shown, no vote.
        if routed.modality == "video" and e.modality == "audio":
            return 0.0
        w = weights.get(e.id)
        return (e.weight if e.weight > 0 else 1.0) if w is None else w

    scored = [e for e in evidence if e.status == "ok" and e.p_ai is not None]
    # weight 0 in fusion.yaml = shown as evidence, never votes or vetoes
    voters = [
        e for e in scored if weight_of(e) > 0 and not (e.signal == "learned" and abs(float(e.p_ai) - 0.5) < chance_band)
    ]
    scores = [float(e.p_ai) for e in voters if e.p_ai is not None]

    method = "weighted_mean"
    if routed.out_of_domain:
        label, p_ai, decided = "inconclusive", None, "fusion"
    elif declared:
        label, p_ai, decided = "declared_ai", 1.0, declared[0].signal  # type: ignore[assignment]
    elif capture:
        label, p_ai, decided = "declared_capture", 0.0, "provenance"
    elif p_stack is not None:
        # the stacker learned how much each detector's disagreement matters, so no veto here
        p_ai, decided, method = p_stack, "fusion", "stacker"
        label = _band(p_ai, bands)
    elif not scores:
        label, p_ai, decided = "inconclusive", None, "fusion"
    else:
        num = 0.0
        den = 0.0
        for e in voters:
            w = weight_of(e)
            num += float(e.p_ai) * w
            den += w
        p_ai = num / den if den else sum(scores) / len(scores)
        decided = "fusion"
        label = "inconclusive" if max(scores) - min(scores) > max_disagreement else _band(p_ai, bands)

    return Verdict(
        request_id=request_id,
        modality=routed.modality,
        sha256=routed.sha256,
        verdict=label,  # type: ignore[arg-type]
        p_ai=p_ai,
        bands=bands,
        decided_by=decided,  # type: ignore[arg-type]
        evidence=evidence,
        fusion=FusionTrace(policy=f"fusion.yaml@v{version}", weights=weights, method=method),
    )


def _band(p_ai: float, bands: Bands) -> str:
    if p_ai >= bands.likely_ai_above:
        return "likely_ai"
    if p_ai <= bands.no_ai_signal_below:
        return "no_ai_signal"
    return "inconclusive"
