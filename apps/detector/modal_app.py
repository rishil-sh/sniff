"""CPU hello-world deploy. GPU (L40S) and model weights come on Day 2."""

from pathlib import Path

import modal

APP_DIR = Path(__file__).parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "fastapi>=0.115",
        "uvicorn[standard]>=0.32",
        "pydantic>=2.9",
        "python-multipart>=0.0.9",
        "python-dotenv>=1.0",
        "openai>=1.55",
        "pyyaml>=6.0",
        "httpx>=0.27",
    )
    .add_local_dir(
        str(APP_DIR / "ai_detector"),
        remote_path="/pkg/ai_detector",
        copy=True,
    )
    .env(
        {
            "PYTHONPATH": "/pkg",
        }
    )
)

app = modal.App("ai-content-detector", image=image)


@app.function(cpu=1.0, memory=2048, timeout=120, min_containers=0)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    from ai_detector.app import app as fastapi_app

    return fastapi_app
