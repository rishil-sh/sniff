from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["image", "video", "audio", "text", "unknown"]
Signal = Literal["provenance", "watermark", "learned", "zero_shot", "vlm", "rule"]
EvidenceStatus = Literal["ok", "skipped", "error", "out_of_domain"]
PointsTo = Literal["ai", "real", "none"]
VerdictLabel = Literal[
    "declared_ai",
    "declared_capture",
    "likely_ai",
    "inconclusive",
    "no_ai_signal",
]
DecidedBy = Literal["provenance", "watermark", "fusion"]


class Artifact(BaseModel):
    kind: str
    uri: str


class Calibration(BaseModel):
    set: str | None = None
    auroc: float | None = None
    tpr_at_1pct_fpr: float | None = None


class Evidence(BaseModel):
    id: str
    detector: str
    version: str
    modality: Modality
    signal: Signal
    status: EvidenceStatus
    raw_score: float | None = None
    p_ai: float | None = None
    points_to: PointsTo = "none"
    weight: float = 0.0
    details: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[Artifact] = Field(default_factory=list)
    calibration: Calibration | None = None
    latency_ms: int = 0


class Bands(BaseModel):
    no_ai_signal_below: float = 0.20
    likely_ai_above: float = 0.80


class FusionTrace(BaseModel):
    policy: str
    weights: dict[str, float] = Field(default_factory=dict)
    method: Literal["weighted_mean", "stacker"] = "weighted_mean"


class ReportReason(BaseModel):
    evidence_id: str
    text: str


Lean = Literal["ai", "human", "unclear"]


class Report(BaseModel):
    model: str
    summary: str
    # Plain-language reply to "is this AI?". `lean` matches the verdict; only an inconclusive verdict gets a
    # reasoned lean, which is a weak guess, never a verdict.
    answer: str = ""
    lean: Lean = "unclear"
    reasons: list[ReportReason] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    request_id: str
    modality: Modality
    sha256: str
    verdict: VerdictLabel
    p_ai: float | None = None
    bands: Bands = Field(default_factory=Bands)
    decided_by: DecidedBy = "fusion"
    evidence: list[Evidence] = Field(default_factory=list)
    fusion: FusionTrace
    report: Report | None = None
    # Context for the report only. Never scored: wording says nothing about who or what voiced it.
    transcript: str | None = None
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
