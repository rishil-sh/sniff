from __future__ import annotations

import json
import os

from ai_detector.schemas import Lean, Report, ReportReason, Verdict

MUSE_BASE_URL = "https://api.meta.ai/v1"
# The strongest Muse text model on the Model API (2026-11-03). muse-spark-1.3 is ~4x faster and ~10x cheaper.
MUSE_MODEL = os.environ.get("AI_DETECTOR_REPORT_MODEL", "muse-wonder-1.0")
MUSE_TIMEOUT_S = 90
TRANSCRIBE_MODEL = "muse-voice-transcribe-1.0"
ELEVENLABS_CLASSIFIER = "https://elevenlabs.io/ai-speech-classifier"

# Each checks only its own vendor's watermark/C2PA; none has a public API we can call.
VERIFY_LINKS = (
    ("OpenAI (SynthID + C2PA):", "https://openai.com/verify"),
    ("Google SynthID Detector:", "https://deepmind.google/models/synthid/"),
    ("Meta Content Seal:", "https://www.meta.ai/identification"),
    ("Adobe Content Credentials:", "https://contentcredentials.org/verify"),
)

# ponytail: measured numbers are pasted from fusion.yaml's stacker comment; update both when you refit
SYSTEM_PROMPT = """\
You write the explanation line of a WhatsApp reply from an AI-content checker for text, images, voice notes and \
video. You receive the verdict JSON our code produced; `modality` says what was sent. Our code writes the bold \
headline (the verdict) and any footer; you write only `why`, plus `lean`.

`why`:
- One or two short sentences, at most 35 words, that a 12-year-old understands. Speak as "we".
- Say what we found in the text, picture, voice or video, in everyday words. No numbers, percentages, field names or \
technical words (classifier, model, score, probability, token, logit, calibrated, detector names).
- For likely_ai, declared_ai, no_ai_signal and declared_capture: state the finding confidently, as a fact about \
what we found. No hedging words (likely, fairly, may, might, seems, probably, not certain, hint) and no disclaimers.
- For inconclusive: say plainly what points each way, in one confident sentence about the signals, without \
claiming it is AI-made or human-made.
- Never contradict `verdict`.

`lean`: "ai" for likely_ai or declared_ai, "human" for no_ai_signal or declared_capture. Only when `verdict` is \
inconclusive, weigh the evidence and pick "ai", "human", or "unclear" when nothing points either way.

Reading text evidence:
- The verdict's `p_ai` is the combined, calibrated probability from two classifiers, txt.fakespot and \
txt.desklib. Their own `p_ai` values are calibrated too.
- `details.window_logits` score successive parts of the text, positive meaning AI-like. A mix of high and low \
windows suggests text that is partly AI-written or heavily edited.
- `txt.tamper`: invisible or look-alike characters were removed before scoring. People rarely type them and they \
are a known trick to fool detectors, so they point toward AI; copy-paste can add a few.
- txt.binoculars and txt.fastdetect have weight 0 and are weak on modern chat models: supporting detail only.
- Texts near the 150-token minimum are less reliable.

Reading image evidence:
- img.c2pa: Content Credentials or IPTC metadata. A valid signed claim of an AI source decides the verdict \
(declared_ai); a missing manifest says nothing either way.
- img.commfor and img.clip are calibrated learned detectors that vote; img.spai has weight 0 and is supporting \
detail only.

Reading audio evidence:
- aud.w2vbert decides. Its `details` say how many four-second windows sounded synthetic; AI voices usually sound \
synthetic throughout, so only a few hot windows is weaker evidence.
- aud.audioseal: a Meta AudioSeal watermark found in the audio means a Meta tool made it (declared_ai).
- aud.dfarena and aud.mambo have weight 0 and often flag real voices: supporting detail only.
- A `transcript`, if present, is context only. Never judge whether a voice is synthetic from its wording.

Reading video evidence:
- vid.watermark: a generator's name (such as KlingAI) printed in the corner of the picture, read in several frames. \
It decides the verdict (declared_ai), like a label the maker put on it.
- img.genmeta on a video: the file's own metadata names the AI tool that made it (declared_ai).
- vid.head scores sampled frames with our image detector and is the only score that votes. A high score points to \
AI; a low one says nothing, because compressed AI videos score as low as real ones. Soundtrack (aud.*) scores are \
shown but do not vote: music, effects and non-English speech are outside what they were tested on.

Measured on 1,056 held-out texts (2026-11-03): the tool calls 2.4% of human texts "likely AI" (16% of academic \
abstracts) and catches 95.6% of AI texts; Llama-3.1-8B (62%) and paraphrased AI text (86%) are caught less often. \
When it says no_ai_signal, 97.8% of texts were human. Only 2.8% of texts were inconclusive, 24 of those 30 human; \
a lean there is a weak guess.
"""

# Strict structured output needs every property required and no extras, so this is written out, not generated.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"why": {"type": "string"}, "lean": {"type": "string", "enum": ["ai", "human", "unclear"]}},
    "required": ["why", "lean"],
    "additionalProperties": False,
}

FIXED_LEAN: dict[str, Lean] = {
    "likely_ai": "ai",
    "declared_ai": "ai",
    "no_ai_signal": "human",
    "declared_capture": "human",
}

# WhatsApp formatting: *bold*, _italic_. The headline is code's, so its confidence matches the measured verdict:
# on held-out text 96.5% of likely_ai and 97.8% of no_ai_signal calls were right; the 2.8% left inconclusive
# get a "best guess" headline, because a lean inside that band is weak.
HEADLINES = {
    ("likely_ai", "audio"): "🤖 *AI-generated voice*",
    ("likely_ai", "text"): "🤖 *Written by AI*",
    ("no_ai_signal", "text"): "✍️ *Written by a human*",
    "likely_ai": "🤖 *AI-generated*",
    "declared_ai": "🤖 *AI-generated*",
    "no_ai_signal": "✅ *No signs of AI*",
    "declared_capture": "📷 *Captured by a real camera or recorder*",
    ("inconclusive", "ai"): "🤔 *Can't say for sure. Best guess: AI*",
    ("inconclusive", "human"): "🤔 *Can't say for sure. Best guess: human*",
    ("inconclusive", "unclear"): "🤔 *Can't say for sure*",
    "unchecked": "⚠️ *Couldn't check this*",
}
TEMPLATE_WHY = {
    "likely_ai": "We found the clear patterns that {tools} leave behind.",
    "declared_ai": "The file itself carries a label saying AI made or edited it.",
    "no_ai_signal": "We found none of the patterns that {tools} leave behind.",
    "declared_capture": "The file's built-in record says it came straight from a camera or recording device.",
    "ai": "Most of the signals point to {tools}, but a few point the other way.",
    "human": "Most of the signals point to a real person, but a few point the other way.",
    "unclear": "The signals point both ways in equal measure.",
}
TOOLS = {
    "text": "AI writing tools",
    "image": "AI image generators",
    "audio": "AI voice generators",
    "video": "AI video generators",
}
# shown under AI verdicts: a wrong call can cost a real person, and 2.4% of human texts are flagged (16% academic)
AI_FOOTER = "_Automated check: please don't use it as the only evidence against anyone._"


def transcribe(wav16: bytes) -> str | None:
    """Muse Voice Transcribe on 16 kHz PCM WAV. Only called with the user's consent; result is report context."""
    import httpx

    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        return None
    request = {"mode": "ENDPOINTING", "model": TRANSCRIBE_MODEL, "audioEncoding": "WAV"}
    resp = httpx.post(
        f"{MUSE_BASE_URL}/asr/transcribe",
        headers={"Authorization": f"Bearer {key}"},
        files={
            "request": (None, json.dumps(request), "application/json"),
            "audio": ("clip.wav", wav16, "audio/wav"),
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("transcript")


def muse_client():
    from openai import OpenAI

    key = os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")
    if not key:
        return None
    return OpenAI(base_url=MUSE_BASE_URL, api_key=key, timeout=MUSE_TIMEOUT_S, max_retries=1)


def lean_for(verdict: Verdict, muse_lean: Lean | None = None) -> Lean:
    """Code decides: a decided verdict fixes the lean. Only an inconclusive verdict with a score gets a lean,
    Muse's if it gave one, else the side of 0.5 the calibrated p_ai falls on."""
    if verdict.verdict in FIXED_LEAN:
        return FIXED_LEAN[verdict.verdict]
    if verdict.p_ai is None:
        return "unclear"
    if verdict.modality == "video" and verdict.p_ai < 0.5:
        return "unclear"  # re-encoded AI video scores as low as real video (fusion.yaml stackers.video)
    return muse_lean or ("ai" if verdict.p_ai >= 0.5 else "human")


def _unchecked_why(verdict: Verdict) -> str:
    reason = next((e.details.get("reason") or "" for e in verdict.evidence if e.id == "sys.router"), "")
    if "shorter than" in reason:
        return "Please send a longer text, at least a full paragraph (about 110 words), so we can check it properly."
    if "English" in reason:
        return "Right now we can only check English text."
    if "longer than" in reason and verdict.modality == "text":
        return "That text is too long. Please send up to about 15,000 words at a time."
    if "longer than" in reason and verdict.modality in ("audio", "video"):
        clip = "voice note" if verdict.modality == "audio" else "video"
        return f"That {clip} is too long. Please send a shorter clip."
    if verdict.modality == "image" and any("palette" in (e.details.get("reason") or "") for e in verdict.evidence):
        return (
            "This picture was saved with a reduced colour palette, which wipes out the traces we look for. "
            "Please send the original file, or a JPEG or full-colour PNG."
        )
    return "We couldn't read this file. Please try a different file or format."


def whatsapp_message(verdict: Verdict, lean: Lean, why: str | None = None) -> str:
    """The reply the WhatsApp bot sends: code's headline, a plain why (Muse's or a template), a footer on AI calls."""
    if verdict.p_ai is None and verdict.verdict == "inconclusive":
        return f"{HEADLINES['unchecked']}\n\n{_unchecked_why(verdict)}"
    key = verdict.verdict if verdict.verdict != "inconclusive" else lean
    headline = (
        HEADLINES.get((verdict.verdict, verdict.modality))
        or HEADLINES.get(verdict.verdict)
        or HEADLINES[("inconclusive", lean)]
    )
    message = f"{headline}\n\n{why or TEMPLATE_WHY[key].format(tools=TOOLS.get(verdict.modality, 'AI tools'))}"
    return f"{message}\n\n{AI_FOOTER}" if lean == "ai" else message


def _payload(verdict: Verdict) -> dict:
    payload = verdict.model_dump(mode="json", exclude={"report"})
    for e in payload["evidence"]:
        # GLTR's token list is the text itself, and thousands of tokens the explanation doesn't need
        e["details"].pop("tokens", None)
        e["details"].pop("timeline", None)  # per-window audio scores: long, and the summary fields carry the signal
    return payload


def hot_ranges(timeline: list[dict], window_s: float, threshold: float = 0.8) -> list[tuple[float, float]]:
    """Merge overlapping windows scoring >= threshold into (start, end) second ranges. Pure; unit-tested."""
    ranges: list[list[float]] = []
    for point in timeline:
        if point["p_ai"] < threshold:
            continue
        start, end = point["t"], point["t"] + window_s
        if ranges and start <= ranges[-1][1]:
            ranges[-1][1] = max(ranges[-1][1], end)
        else:
            ranges.append([start, end])
    return [(a, b) for a, b in ranges]


def _fmt_ranges(ranges: list[tuple[float, float]]) -> str:
    return ", ".join(f"{a:.0f}-{b:.0f} s" for a, b in ranges[:6]) + (" and more" if len(ranges) > 6 else "")


def _audio_reasons(verdict: Verdict) -> tuple[str, list[ReportReason]]:
    """Plain-language summary and reasons for a speech verdict, deciding evidence first."""
    by_id = {e.id: e for e in verdict.evidence}
    reasons: list[ReportReason] = []
    probe = by_id.get("aud.w2vbert")
    summary = f"Fusion returned {verdict.verdict}."
    if probe and probe.status == "ok" and probe.p_ai is not None:
        timeline = probe.details.get("timeline", [])
        hot = sum(point["p_ai"] >= 0.8 for point in timeline)
        ranges = hot_ranges(timeline, probe.details.get("window_s", 4.0))
        where = f"; synthetic-sounding at {_fmt_ranges(ranges)}" if ranges else ""
        reasons.append(
            ReportReason(
                evidence_id=probe.id,
                text=(
                    f"Deciding detector, w2v-BERT 2.0 probe: p(AI) {probe.p_ai:.2f}. "
                    f"{hot} of {len(timeline)} four-second windows scored 0.8 or higher{where}."
                ),
            )
        )
        summary = {
            "likely_ai": f"Likely AI-generated speech. The probe scored {probe.p_ai:.2f}"
            + (" across most of the clip." if timeline and hot >= 0.75 * len(timeline) else "."),
            "no_ai_signal": f"No AI-voice signal found (probe {probe.p_ai:.2f}). This is not proof the voice is human.",
            "inconclusive": f"Inconclusive. The probe scored {probe.p_ai:.2f}, "
            "between the no-signal and likely-AI bands.",
        }.get(verdict.verdict, summary)
    elif probe and probe.status in {"out_of_domain", "skipped", "error"}:
        why = probe.details.get("reason") or probe.details.get("error") or probe.status
        reasons.append(ReportReason(evidence_id=probe.id, text=f"Deciding detector did not score this clip: {why}."))
        summary = f"Inconclusive. The speech detector could not score this clip ({why})."
    seal = by_id.get("aud.audioseal")
    if seal and seal.status == "ok":
        segs = seal.details.get("segments") or []
        text = (
            "AudioSeal watermark found at " + ", ".join(f"{g['start']:.1f}-{g['end']:.1f} s" for g in segs) + "."
            if segs
            else "No AudioSeal watermark (this checks Meta's watermark only)."
        )
        reasons.append(ReportReason(evidence_id=seal.id, text=text))
        if segs:
            summary = "Declared AI: a Meta AudioSeal watermark is embedded in this audio."
    c2pa = by_id.get("img.c2pa")
    if c2pa and c2pa.status == "ok":
        if c2pa.details.get("manifest_present"):
            gen = c2pa.details.get("claim_generator") or "an unnamed tool"
            src = c2pa.details.get("digital_source_type") or "no source type"
            reasons.append(ReportReason(evidence_id=c2pa.id, text=f"C2PA Content Credentials from {gen} ({src})."))
        else:
            reasons.append(ReportReason(evidence_id=c2pa.id, text="No C2PA Content Credentials in the file."))
    context = [
        f"{name} {e.p_ai:.2f}"
        for eid, name in (("aud.dfarena", "DF Arena"), ("aud.mambo", "XLSR-MamBo"))
        if (e := by_id.get(eid)) and e.status == "ok" and e.p_ai is not None
    ]
    if context:
        reasons.append(
            ReportReason(
                evidence_id="aud.dfarena",
                text=(
                    "Context only, not used in the verdict: "
                    + ", ".join(context)
                    + " (both flag many real human voices as AI in our tests)."
                ),
            )
        )
    router = by_id.get("sys.router")
    if router and router.details.get("reason"):
        reasons.append(ReportReason(evidence_id=router.id, text=f"Out of scope: {router.details['reason']}."))
        summary = f"Inconclusive. Out of scope: {router.details['reason']}."
    return summary, reasons


NAMES = {
    "img.head": "Screenshot-robust Community Forensics head",
    "img.commfor": "Community Forensics",
    "img.clip": "CLIP linear probe",
    "img.spai": "SPAI spectral detector",
    "txt.fakespot": "Fakespot classifier",
    "txt.desklib": "Desklib classifier",
    "txt.binoculars": "Binoculars",
    "txt.fastdetect": "Fast-DetectGPT",
}
POINTS = {"ai": "points to AI", "real": "points to human-made", "none": "points neither way"}


def _plain_reasons(verdict: Verdict) -> list[ReportReason]:
    """Image/text evidence in plain words: provenance, voting detectors, then weight-0 context on one line."""
    reasons: list[ReportReason] = []
    by_id = {e.id: e for e in verdict.evidence}
    c2pa = by_id.get("img.c2pa")
    if c2pa and c2pa.status == "ok":
        if c2pa.details.get("manifest_present"):
            gen = c2pa.details.get("claim_generator") or "an unnamed tool"
            src = c2pa.details.get("digital_source_type") or "no source type"
            reasons.append(ReportReason(evidence_id=c2pa.id, text=f"Content Credentials from {gen} ({src})."))
        else:
            reasons.append(ReportReason(evidence_id=c2pa.id, text="No C2PA or IPTC metadata in the file."))
    for e in verdict.evidence:  # a generator's own label decides the verdict: say which one
        if e.id in ("img.genmeta", "vid.watermark") and e.status == "ok" and e.points_to == "ai":
            where = "printed on the picture" if e.id == "vid.watermark" else "in the file's metadata"
            text = f"Generator label {where}: {e.details.get('generator')}."
            reasons.append(ReportReason(evidence_id=e.id, text=text))
    if tamper := by_id.get("txt.tamper"):
        reasons.append(
            ReportReason(
                evidence_id=tamper.id,
                text=(
                    f"Removed {tamper.details.get('hidden', 0)} invisible and {tamper.details.get('lookalike', 0)} "
                    "look-alike characters before scoring (a known trick to fool detectors)."
                ),
            )
        )
    weights = verdict.fusion.weights

    def votes(e) -> bool:  # same rule as fusion.weight_of: unlisted ids vote, a video's soundtrack never does
        if verdict.modality == "video" and e.modality == "audio":
            return False
        w = weights.get(e.id)
        return (w if w is not None else 1.0) > 0

    scored = [
        e
        for e in verdict.evidence
        if e.status == "ok" and e.p_ai is not None and e.signal != "provenance" and e.id != "txt.tamper"
    ]
    for e in (e for e in scored if votes(e)):
        reasons.append(
            ReportReason(
                evidence_id=e.id,
                text=(f"Voting detector, {NAMES.get(e.id, e.detector)}: p(AI) {e.p_ai:.2f}, {POINTS[e.points_to]}."),
            )
        )
    context = [e for e in scored if not votes(e)]
    if context:
        reasons.append(
            ReportReason(
                evidence_id=context[0].id,
                text=(
                    "Context only, not used in the verdict: "
                    + ", ".join(f"{NAMES.get(e.id, e.detector)} {e.p_ai:.2f}" for e in context)
                    + "."
                ),
            )
        )
    seen = set()
    for e in verdict.evidence:  # detectors often share one reason (e.g. a palette image): say it once
        why = e.details.get("reason") if e.status == "out_of_domain" else None
        if why and why not in seen:
            seen.add(why)
            reasons.append(ReportReason(evidence_id=e.id, text=f"Out of scope: {why}."))
    return reasons


def _stub_report(verdict: Verdict, *, send_to_muse: bool) -> Report:
    caveats = []
    # Only evidence that can move the verdict matters here; weight-0 context scores are labelled in the reasons.
    if any(e.calibration and e.calibration.set == "uncalibrated-sigmoid" and e.weight > 0 for e in verdict.evidence):
        caveats.append("Learned detector scores are uncalibrated sigmoids until a held-out Platt fit.")
    c2pa = next((item for item in verdict.evidence if item.id == "img.c2pa"), None)
    if verdict.modality == "text":
        caveats.append(
            "Paraphrasing defeats text detectors, and they over-flag non-native English writers. "
            "Never use this verdict alone to discipline anyone."
        )
    elif c2pa is None or not c2pa.details.get("manifest_present"):
        caveats.append("Missing C2PA or IPTC metadata is not evidence of authenticity.")
    elif c2pa.points_to == "ai" and not c2pa.details.get("signature_valid"):
        generator = c2pa.details.get("claim_generator") or "the generator"
        caveats.append(
            f"C2PA from {generator} records an AI source type, but the signature "
            "did not validate (a convert or re-save often breaks it). "
            "The claim is treated like unsigned IPTC, not a tamper-evident proof."
        )
    if verdict.verdict == "likely_ai" and verdict.decided_by == "fusion" and verdict.modality == "image":
        caveats.append(
            "In our held-out eval this cutoff flagged 0 of 1717 real photos (incl. screenshots and "
            "faces), which still allows a true rate up to ~0.2%. Treat it as strong evidence, not proof."
        )
    if verdict.verdict == "no_ai_signal":
        caveats.append('"No AI signal found" is never a claim that the content is real.')
    if verdict.modality == "image" and verdict.verdict != "declared_ai":
        caveats.append(
            "Stripped or re-encoded files lose C2PA. Vendor watermarks only their makers can read: "
            + "; ".join(f"{name} {url}" for name, url in VERIFY_LINKS)
        )
    if verdict.modality == "audio":
        caveats.append(
            "Audio verdicts are advisory and calibrated on English speech only. The deciding detector "
            "(aud.w2vbert) was trained on ElevenLabs, OpenAI and open-source TTS; unfamiliar voice "
            "generators, reverberation, speech over music and heavy compression can still fool it."
        )
        if verdict.verdict != "declared_ai":
            caveats.append(f"ElevenLabs checks its own output at {ELEVENLABS_CLASSIFIER}")
    if send_to_muse and not (os.environ.get("MODEL_API_KEY") or os.environ.get("META_API_KEY")):
        caveats.append(f"MODEL_API_KEY is unset; {MUSE_MODEL} was not called.")
    reasons = _plain_reasons(verdict)
    summary = whatsapp_message(verdict, lean_for(verdict)).split("\n")[0].replace("*", "")
    if verdict.modality == "audio":
        summary, reasons = _audio_reasons(verdict)
    if c2pa and c2pa.details.get("claim_generator") and verdict.verdict == "declared_ai":
        summary = (
            f"{c2pa.details['claim_generator']} declared this "
            f"{c2pa.details.get('digital_source_type') or 'AI-generated'}."
        )
    lean = lean_for(verdict)
    return Report(
        model="stub",
        summary=summary,
        answer=whatsapp_message(verdict, lean),
        lean=lean,
        reasons=reasons,
        caveats=caveats,
    )


def explain_verdict(verdict: Verdict, *, send_to_muse: bool) -> Report:
    if not send_to_muse:
        report = _stub_report(verdict, send_to_muse=False)
        report.caveats.append(
            "Nothing was sent to the Meta Model API. Tick the consent box for a Muse-written explanation."
        )
        return report

    if verdict.p_ai is None:  # nothing was scored (too short, not English, ...): the template says all there is
        return _stub_report(verdict, send_to_muse=False)
    client = muse_client()
    if client is None:
        return _stub_report(verdict, send_to_muse=True)

    try:
        # shortcut: media bytes are not attached; Muse reasons from the verdict JSON only
        resp = client.chat.completions.create(
            model=MUSE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(_payload(verdict))},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "Answer", "schema": ANSWER_SCHEMA, "strict": True},
            },
            reasoning_effort="low",  # a chat reply: ~7 s instead of ~20 s on Wonder
        )
        data = json.loads(resp.choices[0].message.content or "{}")
    except Exception as exc:  # noqa: BLE001 — surface API errors in the report
        report = _stub_report(verdict, send_to_muse=True)
        report.caveats.append(f"{MUSE_MODEL} call failed: {exc}")
        return report

    lean = lean_for(verdict, data.get("lean") if data.get("lean") in ("ai", "human", "unclear") else None)
    # Muse contradicting a decided verdict means its sentence can't be trusted either: use the template line
    why = (data.get("why") or "").strip() if lean == data.get("lean") else None
    report = _stub_report(verdict, send_to_muse=True)  # deterministic reasons and caveats for JSON consumers
    report.model, report.lean, report.answer = MUSE_MODEL, lean, whatsapp_message(verdict, lean, why or None)
    return report
