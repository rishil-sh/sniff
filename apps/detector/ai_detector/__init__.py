"""Open-source AI-content detector."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # HF_TOKEN / MODEL_API_KEY for every entry point: app, scripts, eval

# Weights stay in the repo (git-ignored) unless AI_DETECTOR_CACHE is set, as Docker does.
# Set these before huggingface_hub or torch are imported; both read the variables at import time.
_CACHE = Path(os.environ.get("AI_DETECTOR_CACHE", Path(__file__).resolve().parent.parent / "data" / "models"))
os.environ["AI_DETECTOR_CACHE"] = str(_CACHE)
_HF = _CACHE / "hf"
os.environ.setdefault("HF_HOME", str(_HF))
os.environ.setdefault("HF_HUB_CACHE", str(_HF))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_HF))
os.environ.setdefault("HF_XET_CACHE", str(_HF / "xet"))
os.environ.setdefault("TORCH_HOME", str(_CACHE / "torch"))
os.environ.setdefault("AUDIOSEAL_CACHE_DIR", str(_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE / "cache"))

# A library imported earlier already copied the default (~/.cache) into its constants.
_hf_constants = sys.modules.get("huggingface_hub.constants")
if _hf_constants is not None:
    _hf_constants.HF_HOME = Path(os.environ["HF_HOME"])
    _hf_constants.HF_HUB_CACHE = Path(os.environ["HF_HUB_CACHE"])

CACHE_DIR = _CACHE
HF_CACHE = _HF

__version__ = "0.1.0"
