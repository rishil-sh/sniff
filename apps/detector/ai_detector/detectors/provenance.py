"""C2PA + IPTC/XMP Digital Source Type."""

from __future__ import annotations

import io
import json
import re
import subprocess
import tempfile
import time
from typing import Any

from ai_detector.router import RoutedInput, ffmpeg_exe
from ai_detector.schemas import Evidence

VERSION = "c2pa-python@0.38.2"

AI_SOURCES = {
    "trainedAlgorithmicMedia",
    "compositeWithTrainedAlgorithmicMedia",
    "trainedAlgorithmicData",
    "algorithmicMedia",
}
CAPTURE_SOURCES = {
    "digitalCapture",
    "computationalCapture",
    "negativeFilm",
    "positiveFilm",
    "print",
}


def _source_token(value: str) -> str:
    return value.rstrip("/").split("/")[-1]


def _walk_actions(manifest: dict) -> list[dict]:
    actions: list[dict] = []
    for assertion in manifest.get("assertions") or []:
        label = str(assertion.get("label") or "")
        if not label.startswith("c2pa.actions"):
            continue
        data = assertion.get("data") or {}
        actions.extend(data.get("actions") or [])
    return actions


def _collect_source_types(manifest: dict) -> list[str]:
    found: list[str] = []
    for action in _walk_actions(manifest):
        raw = action.get("digitalSourceType") or action.get("digital_source_type")
        if raw:
            found.append(str(raw))
    for ingredient in manifest.get("ingredients") or []:
        raw = ingredient.get("digitalSourceType") or ingredient.get("digital_source_type")
        if raw:
            found.append(str(raw))
        for assertion in ingredient.get("assertions") or []:
            data = assertion.get("data") or {}
            for action in data.get("actions") or []:
                raw = action.get("digitalSourceType") or action.get("digital_source_type")
                if raw:
                    found.append(str(raw))
    return found


def interpret_manifest(store: dict[str, Any] | None, validation_state: str | None) -> dict[str, Any]:
    """Turn a C2PA store into the evidence `details` fields. Pure; unit-tested."""
    if not store:
        return {
            "manifest_present": False,
            "signature_valid": False,
            "claim_generator": None,
            "digital_source_type": None,
            "actions": [],
            "points_to": "none",
        }

    manifests = store.get("manifests") or {}
    active_id = store.get("active_manifest")
    manifest = manifests.get(active_id) if active_id else None
    if manifest is None and manifests:
        manifest = next(iter(manifests.values()))
    manifest = manifest or {}

    valid = (validation_state or store.get("validation_state") or "").lower() in {
        "valid",
        "trusted",
    }
    sources = _collect_source_types(manifest)
    tokens = [_source_token(s) for s in sources]
    actions = [
        {
            "action": a.get("action"),
            "digital_source_type": a.get("digitalSourceType") or a.get("digital_source_type"),
        }
        for a in _walk_actions(manifest)
    ]
    ai = any(t in AI_SOURCES for t in tokens)
    capture = any(t in CAPTURE_SOURCES for t in tokens) and not ai
    # AI self-declaration is treated like IPTC: present even if the
    # signature broke after a convert/re-save. Capture still needs a valid sig.
    if ai:
        points = "ai"
    elif valid and capture:
        points = "real"
    else:
        points = "none"

    generator = None
    info = manifest.get("claim_generator_info")
    if isinstance(info, list) and info:
        generator = info[0].get("name")
    generator = generator or manifest.get("claim_generator")

    return {
        "manifest_present": True,
        "signature_valid": valid,
        "claim_generator": generator,
        "digital_source_type": tokens[0] if tokens else None,
        "digital_source_types": tokens,
        "actions": actions,
        "validation_state": validation_state or store.get("validation_state"),
        "points_to": points,
    }


def _iptc_source(data: bytes) -> str | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        image = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001 - untrusted file: any parse failure means no metadata
        # not an image (mp4, mov): XMP sits in the container as plain text, so look for the AI tokens directly
        return next((token for token in AI_SOURCES if token.encode() in data), None)
    blob = ""
    try:
        xmp = image.getxmp()
        blob = json.dumps(xmp) if xmp else ""
    except Exception:  # noqa: BLE001 - untrusted file: any parse failure means no metadata
        blob = str(image.info.get("XML:com.adobe.xmp") or "")
    if not blob:
        blob = str(image.info.get("XML:com.adobe.xmp") or "")
    for token in list(AI_SOURCES) + list(CAPTURE_SOURCES):
        if token in blob:
            return token
    return None


# Matched against Software / CreatorTool only, never free-text captions.
GENERATOR_NAMES = (
    "novelai",
    "midjourney",
    "dall-e",
    "dall·e",
    "firefly",
    "stable diffusion",
    "comfyui",
    "invokeai",
    "fooocus",
    "draw things",
    "ideogram",
)


_AIGC_KEY = b"ContentProducer"
_AIGC_WINDOW = 2000  # bytes either side of the key; the label JSON is a few hundred


def embedded_aigc(data: bytes) -> str | None:
    """The AIGC implicit-label JSON (GB 45438-2025) wherever the file keeps it: mp4 tags, XMP, PNG text. Pure.

    Found by its ContentProducer key, not a container field, because every generator stores it differently.
    """
    i = data.find(_AIGC_KEY)
    while i != -1:
        start = data.rfind(b"{", max(0, i - _AIGC_WINDOW), i)
        end = data.find(b"}", i, i + _AIGC_WINDOW)
        if start != -1 and end != -1:
            blob = data[start : end + 1].decode("utf-8", "replace").replace("&quot;", '"')  # XMP attributes escape it
            try:
                if isinstance(json.loads(blob), dict):
                    return blob
            except ValueError:
                pass
        i = data.find(_AIGC_KEY, i + 1)
    return None


def generator_signature(fields: dict[str, str]) -> str | None:
    """Name the AI generator whose own metadata is in `fields`. Pure; unit-tested."""
    if "Steps:" in fields.get("parameters", "") or (
        "Steps:" in fields.get("UserComment", "") and "Sampler:" in fields.get("UserComment", "")
    ):
        return "AUTOMATIC1111/Forge parameters"
    if '"class_type"' in fields.get("prompt", "") or '"nodes"' in fields.get("workflow", ""):
        return "ComfyUI workflow"
    if {"sd-metadata", "invokeai_metadata", "Dream"} & fields.keys():
        return "InvokeAI metadata"
    try:
        aigc = json.loads(fields.get("AIGC") or "{}")
    except ValueError:
        aigc = {}
    # China's AIGC implicit label (GB 45438-2025), written by Kling, Jimeng, Hailuo...: Label 1 = AI-generated
    if isinstance(aigc, dict) and str(aigc.get("Label")) == "1":
        return f"AIGC label ({aigc.get('ContentProducer') or 'unknown producer'})"
    if "fooocus_scheme" in fields:
        return "Fooocus metadata"
    if "made with google ai" in fields.get("xmp", "").lower():
        return "Google AI credit line"
    tool = " ".join(fields.get(k, "") for k in ("Software", "Source", "CreatorTool")).lower()
    return next((name for name in GENERATOR_NAMES if name in tool), None)


def _generator_fields(data: bytes) -> dict[str, str]:
    from PIL import Image

    image = Image.open(io.BytesIO(data))
    fields = {str(k): str(v) for k, v in image.info.items() if isinstance(v, str)}
    exif = image.getexif()
    fields["Software"] = fields.get("Software", "") + " " + str(exif.get(0x0131) or "")
    comment = exif.get_ifd(0x8769).get(0x9286)
    if isinstance(comment, bytes):
        # UserComment is "UNICODE\0" + UTF-16 or "ASCII\0\0\0" + text; dropping NULs reads both.
        fields["UserComment"] = comment.replace(b"\x00", b"").decode("latin-1")
    elif comment:
        fields["UserComment"] = str(comment)
    try:
        xmp = json.dumps(image.getxmp() or {})
    except Exception:  # noqa: BLE001 - untrusted file: any parse failure means no metadata
        xmp = str(image.info.get("XML:com.adobe.xmp") or "")
    fields["xmp"] = xmp
    tool = re.search(r'CreatorTool"?\s*[:=>]\s*"?([^"<]+)', xmp)
    fields["CreatorTool"] = tool.group(1) if tool else ""
    fields["AIGC"] = fields.get("AIGC") or embedded_aigc(data) or ""
    return fields


def _video_fields(data: bytes) -> dict[str, str]:
    """Container tags (mp4 udta/meta) as key -> value, via ffmpeg's ffmetadata dump."""
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(data)
        tmp.flush()
        out = subprocess.run(
            [ffmpeg_exe(), "-v", "error", "-i", tmp.name, "-f", "ffmetadata", "-"],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
    pairs = (line.partition("=") for line in out.splitlines() if not line.startswith(";"))
    # ffmetadata backslash-escapes = ; # \ in values
    fields = {k: re.sub(r"\\(.)", r"\1", v) for k, sep, v in pairs if sep}
    fields["AIGC"] = fields.get("AIGC") or embedded_aigc(data) or ""  # some tools keep it in XMP, not a tag
    return fields


def _read_c2pa(data: bytes, mime: str | None) -> tuple[dict | None, str | None]:
    from c2pa import Reader

    stream = io.BytesIO(data)
    reader = None
    if mime:
        try:
            reader = Reader.try_create(mime, stream)
        except Exception:  # noqa: BLE001 - untrusted file: any parse failure means no metadata
            stream.seek(0)
            reader = None
    if reader is None:
        stream.seek(0)
        reader = Reader.try_create(stream)
    if reader is None:
        return None, None
    with reader:
        store = json.loads(reader.json())
        state = reader.get_validation_state()
    return store, state


async def collect(routed: RoutedInput) -> list[Evidence]:
    if routed.data is None or routed.modality not in {"image", "video", "audio"}:
        return []
    started = time.perf_counter()
    try:
        store, state = _read_c2pa(routed.data, routed.mime)
        details = interpret_manifest(store, state)
    except Exception as exc:  # noqa: BLE001
        return [
            Evidence(
                id="img.c2pa",
                detector="c2pa-python",
                version=VERSION,
                modality=routed.modality,
                signal="provenance",
                status="error",
                details={"error": str(exc)},
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
        ]

    iptc = _iptc_source(routed.data)
    details["iptc_digital_source_type"] = iptc
    points = details["points_to"]
    if iptc in AI_SOURCES:
        points = "ai"
        details["points_to"] = "ai"

    items = [
        Evidence(
            id="img.c2pa",
            detector="c2pa-python",
            version=VERSION,
            modality=routed.modality,
            signal="provenance",
            status="ok",
            points_to=points,
            details=details,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    ]
    if routed.modality in ("image", "video"):
        items.append(_genmeta(routed.data, routed.modality))
    return items


def _genmeta(data: bytes, modality: str = "image") -> Evidence:
    started = time.perf_counter()
    try:
        fields = _video_fields(data) if modality == "video" else _generator_fields(data)
        generator = generator_signature(fields)
        status, details = "ok", {"generator": generator}
    except Exception as exc:  # noqa: BLE001
        generator, status, details = None, "error", {"error": str(exc)}
    return Evidence(
        id="img.genmeta",
        detector="generator-metadata",
        version="0.2.0",
        modality=modality,  # type: ignore[arg-type]
        signal="provenance",
        status=status,  # type: ignore[arg-type]
        points_to="ai" if generator else "none",
        details=details,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
