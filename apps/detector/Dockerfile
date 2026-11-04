# Build context: apps/detector. Models download into AI_DETECTOR_CACHE on first use (mount it as a volume).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY ai_detector ./ai_detector
RUN uv sync --frozen --no-dev
ENV AI_DETECTOR_CACHE=/models
EXPOSE 7860
CMD ["uv", "run", "--no-sync", "ai-detector"]
