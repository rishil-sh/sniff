from __future__ import annotations

import uuid

from ai_detector.detectors import gather_evidence
from ai_detector.detectors.audio import SUFFIX, to_wav16
from ai_detector.fusion import fuse
from ai_detector.muse.client import explain_verdict, transcribe
from ai_detector.router import route
from ai_detector.schemas import Verdict


async def analyze(
    *,
    data: bytes | None = None,
    filename: str | None = None,
    text: str | None = None,
    send_to_muse: bool = False,
) -> Verdict:
    routed = route(data=data, filename=filename, text=text)
    evidence = await gather_evidence(routed)
    verdict = fuse(routed, evidence, request_id=uuid.uuid4().hex[:8])
    if send_to_muse and routed.modality == "audio" and routed.data and not routed.out_of_domain:
        try:
            verdict.transcript = transcribe(to_wav16(routed.data, SUFFIX.get(routed.mime or "", ".wav")))
        except Exception:  # noqa: BLE001 — a transcript is optional context
            verdict.transcript = None
    verdict.report = explain_verdict(verdict, send_to_muse=send_to_muse)
    return verdict
