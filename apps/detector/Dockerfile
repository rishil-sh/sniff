# Detector API (FastAPI). Multi-stage on purpose: the final image is python:3.11-slim with the
# installed venv only. Alpine can't run the PyTorch wheels (they need glibc). No model weights
# are baked in; `docker compose` runs ai-detector-download into the `models` volume first.
# Linux resolves CPU-only PyTorch (see pyproject.toml). Build context: apps/detector.

FROM python:3.11-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends binutils \
    && rm -rf /var/lib/apt/lists/*
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project
COPY ai_detector ./ai_detector
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
# Compile to bytecode above, then drop files inference never loads and strip debug symbols.
RUN /app/.venv/bin/python - <<'PY'
import shutil
import sysconfig
from pathlib import Path

site = Path(sysconfig.get_path("purelib"))
torch = site / "torch"
# Headers and the C++/Python test trees are not used for inference.
for name in ("include", "test"):
    path = torch / name
    if path.is_dir():
        shutil.rmtree(path)
# torch/bin is ~50 MB of test executables. Startup only needs torch_shm_manager.
bin_dir = torch / "bin"
if bin_dir.is_dir():
    keep = {"torch_shm_manager", "upgrader_models", "script_module_v4.ptl"}
    for child in bin_dir.iterdir():
        if child.name in keep:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()
for path in site.rglob("*"):
    if path.is_file() and path.suffix in {".pyi", ".a"}:
        path.unlink()
PY
# Skip auditwheel-vendored *.libs/: they are patchelf'd, and stripping them corrupts them (SIGBUS on import).
RUN find /app/.venv -type f \( -name '*.so' -o -name '*.so.*' -o -name 'ffmpeg-*' \) -not -path '*.libs/*' -print0 \
    | xargs -0 -r strip --strip-unneeded || true

FROM python:3.11-slim
# opencv-python (via invisible-watermark, used by the watermark detector) links libxcb, libGL and glib even headless.
RUN apt-get update && apt-get install -y --no-install-recommends libxcb1 libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 detector && mkdir /models && chown detector /models
WORKDIR /app
COPY --from=build --chown=detector:detector /app/.venv /app/.venv
COPY --from=build --chown=detector:detector /app/ai_detector /app/ai_detector
USER detector
# Every cache the detectors and their libraries use lands in the one models volume.
ENV PATH="/app/.venv/bin:$PATH" \
    AI_DETECTOR_CACHE=/models \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    XDG_CACHE_HOME=/models/cache
VOLUME /models
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"
CMD ["ai-detector"]
