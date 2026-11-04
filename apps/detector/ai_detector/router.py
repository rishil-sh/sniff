from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from ai_detector.schemas import Modality

IMAGE_MAX_BYTES = 20 * 1024 * 1024
AUDIO_MAX_BYTES = 32 * 1024 * 1024  # Muse Voice Transcribe limit
VIDEO_MAX_BYTES = 100 * 1024 * 1024
VIDEO_MAX_S = 60.0
AUDIO_MAX_S = 10 * 60.0
# ~85 words. Below this the text head wrongly flags 9-12% of human text (fusion.yaml stackers.text); 100-150 tokens
# get stricter short_bands.
TEXT_MIN_TOKENS = 100
TEXT_MAX_CHARS = 100_000

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_WORD_RE = re.compile(r"[^\W\d_]+")
# Most frequent English words; they make up about a third of running English prose.
_EN_FUNCTION_WORDS = frozenset(
    "the of and to a in is that it for was on with as be by this are at from or have an not "
    "but which you they we he she his her their has were been had i can will would there".split()
)
TEXT_MIN_ENGLISH_SHARE = 0.15

# Cyrillic/Greek letters that render like Latin ones: the homoglyph attack that cut Binoculars'
# RAID TPR from 0.79 to 0.36 (reports/AI text detection accuracy research.md §6).
# ponytail: hand-picked lookalikes, not Unicode confusables.txt; extend if an attack slips past
_LOOKALIKES = "аеорсухіјѕһԁԛԝАВЕКМНОРСТХУІЈЅαονρΑΒΕΖΗΙΚΜΝΟΡΤΥΧ"
_CONFUSABLES = str.maketrans(_LOOKALIKES, "aeopcyxijshdqwABEKMHOPCTXYIJSaovpABEZHIKMNOPTYX")
_CONFUSABLE_CHARS = frozenset(_LOOKALIKES)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus"}  # Ogg/Opus: WhatsApp voice notes


@dataclass
class RoutedInput:
    modality: Modality
    out_of_domain: bool
    reason: str | None
    filename: str | None
    mime: str | None
    size_bytes: int
    duration_s: float | None
    token_count: int | None
    sha256: str
    data: bytes | None = field(repr=False, default=None)
    text: str | None = None
    # what normalize_text changed; hidden/lookalike counts are reported as tamper evidence
    text_changes: dict[str, int] = field(default_factory=dict)


def normalize_text(text: str) -> tuple[str, dict[str, int]]:
    """Undo invisible-character, homoglyph and whitespace attacks before any detector sees the text.

    Returns (clean text, counts). `hidden` = format characters removed (zero-width, bidi, soft hyphen);
    `lookalike` = Cyrillic/Greek letters mapped to Latin inside words that are otherwise Latin.
    """

    def is_hidden(i: int) -> bool:
        if unicodedata.category(text[i]) != "Cf":
            return False
        # a zero-width joiner between emoji (👨‍💻, 🏳️‍🌈) is how those emoji are written, not an attack
        return not (
            text[i] == "‍"
            and 0 < i < len(text) - 1
            and unicodedata.category(text[i - 1]) in ("So", "Sk", "Mn")
            and unicodedata.category(text[i + 1]) == "So"
        )

    drop = {i for i in range(len(text)) if is_hidden(i)}
    hidden = len(drop)
    out = "".join(c for i, c in enumerate(text) if i not in drop) if hidden else text
    nfkc = unicodedata.normalize("NFKC", out)  # fullwidth, ligatures, exotic spaces -> plain
    lookalike = 0

    def fix(m: re.Match) -> str:
        nonlocal lookalike
        word = m.group(0)
        bad = sum(c in _CONFUSABLE_CHARS for c in word)
        # only words made of Latin + lookalike letters: real Cyrillic/Greek words keep their letters
        if bad and all(c in _CONFUSABLE_CHARS or unicodedata.name(c, "").startswith("LATIN") for c in word):
            lookalike += bad
            return word.translate(_CONFUSABLES)
        return word

    clean = _WORD_RE.sub(fix, nfkc)
    clean = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", clean)).strip()
    return clean, {"hidden": hidden, "lookalike": lookalike, "chars_removed": len(text) - len(clean)}


def estimate_tokens(text: str) -> int:
    # shortcut: Llama tokenizer replaces this on the text path
    return len(_TOKEN_RE.findall(text))


def english_share(text: str) -> float:
    # ponytail: function-word share, not a language-ID model; swap in lingua/fastText lid if it misroutes
    words = _WORD_RE.findall(text.lower())
    return sum(w in _EN_FUNCTION_WORDS for w in words) / max(len(words), 1)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sniff(data: bytes, filename: str | None) -> tuple[Modality, str | None]:
    name = (filename or "").lower()
    ext = Path(name).suffix

    if len(data) >= 3 and data[:3] == b"\xff\xd8\xff":
        return "image", "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image", "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image", "image/gif"
    if data.startswith(b"RIFF") and len(data) >= 12:
        kind = data[8:12]
        if kind == b"WEBP":
            return "image", "image/webp"
        if kind == b"WAVE":
            return "audio", "audio/wav"
    if data.startswith(b"fLaC"):
        return "audio", "audio/flac"
    if data.startswith(b"OggS"):
        return "audio", "audio/ogg"
    if data.startswith(b"ID3") or (len(data) >= 2 and data[0] == 0xFF and data[1] in (0xFB, 0xF3, 0xF2, 0xFA)):
        return "audio", "audio/mpeg"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in {b"M4A ", b"M4B "}:
            return "audio", "audio/mp4"
        if brand == b"qt  ":
            return "unknown", "video/quicktime"
        return "video", "video/mp4"

    if ext in IMAGE_EXTS:
        return "image", None
    if ext in VIDEO_EXTS:
        return "video", "video/mp4"
    if ext in AUDIO_EXTS:
        mime = {
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".m4a": "audio/mp4",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
            ".opus": "audio/ogg",
        }[ext]
        return "audio", mime
    return "unknown", None


def ffmpeg_exe() -> str:
    """FFMPEG env var, then ffmpeg on PATH, then the binary bundled with imageio-ffmpeg."""
    found = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        return "ffmpeg"


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


def parse_ffmpeg_duration(stderr: str) -> float | None:
    """Seconds from ffmpeg's "Duration: HH:MM:SS.ss" banner. Pure; unit-tested."""
    m = _DURATION_RE.search(stderr)
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3)) if m else None


def probe_duration_s(data: bytes, suffix: str) -> float | None:
    if not data:
        return None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(data)
        tmp.flush()
        if not shutil.which("ffprobe"):  # imageio-ffmpeg ships ffmpeg only
            proc = subprocess.run(
                [ffmpeg_exe(), "-hide_banner", "-i", tmp.name], capture_output=True, text=True, timeout=20, check=False
            )
            return parse_ffmpeg_duration(proc.stderr)
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                tmp.name,
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    if proc.returncode != 0:
        return None
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return None


def route(
    *,
    data: bytes | None = None,
    filename: str | None = None,
    text: str | None = None,
) -> RoutedInput:
    if text is not None and data is None:
        digest = sha256_bytes(text.encode("utf-8"))  # of what was submitted, before normalising
        submitted_bytes = len(text.encode("utf-8"))
        text, changes = normalize_text(text)
        tokens = estimate_tokens(text)
        ood = False
        reason = None
        if len(text) > TEXT_MAX_CHARS:
            ood, reason = True, f"text longer than {TEXT_MAX_CHARS} characters"
        elif tokens < TEXT_MIN_TOKENS:
            ood, reason = True, f"text shorter than {TEXT_MIN_TOKENS} tokens"
        elif english_share(text) < TEXT_MIN_ENGLISH_SHARE:
            ood, reason = True, "text does not look like English; the text detectors are English-only"
        return RoutedInput(
            modality="text",
            out_of_domain=ood,
            reason=reason,
            filename=filename,
            mime="text/plain",
            size_bytes=submitted_bytes,
            duration_s=None,
            token_count=tokens,
            sha256=digest,
            text=text,
            text_changes=changes,
        )

    if not data:
        return RoutedInput(
            modality="unknown",
            out_of_domain=True,
            reason="no file or text provided",
            filename=filename,
            mime=None,
            size_bytes=0,
            duration_s=None,
            token_count=None,
            sha256=sha256_bytes(b""),
        )

    modality, mime = sniff(data, filename)
    digest = sha256_bytes(data)
    size = len(data)
    duration = None
    ood = False
    reason = None

    if modality == "unknown":
        ood, reason = True, "unrecognized file type"

    elif modality == "image":
        if size > IMAGE_MAX_BYTES:
            ood, reason = True, f"image larger than {IMAGE_MAX_BYTES} bytes"

    elif modality == "video":
        if mime == "video/quicktime" or (filename or "").lower().endswith(".mov"):
            ood, reason = True, "only mp4 video is in scope"
        elif size > VIDEO_MAX_BYTES:
            ood, reason = True, f"video larger than {VIDEO_MAX_BYTES} bytes"
        else:
            duration = probe_duration_s(data, ".mp4")
            if duration is None:
                ood, reason = True, "could not read video duration"
            elif duration > VIDEO_MAX_S:
                ood, reason = True, f"video longer than {int(VIDEO_MAX_S)} seconds"

    elif modality == "audio":
        suffix = Path(filename or "clip.wav").suffix or ".wav"
        if size > AUDIO_MAX_BYTES:
            ood, reason = True, f"audio larger than {AUDIO_MAX_BYTES} bytes"
        else:
            duration = probe_duration_s(data, suffix)
            if duration is None:
                ood, reason = True, "could not read audio duration"
            elif duration > AUDIO_MAX_S:
                ood, reason = True, f"audio longer than {int(AUDIO_MAX_S)} seconds"

    return RoutedInput(
        modality=modality,
        out_of_domain=ood,
        reason=reason,
        filename=filename,
        mime=mime,
        size_bytes=size,
        duration_s=duration,
        token_count=None,
        sha256=digest,
        data=data,
    )
