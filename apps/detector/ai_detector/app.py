from __future__ import annotations

from fastapi import FastAPI, Form, UploadFile
from pydantic import BaseModel

from ai_detector import __version__
from ai_detector.analyze import analyze
from ai_detector.schemas import Verdict


class TextRequest(BaseModel):
    text: str
    send_to_muse: bool = False


def create_app() -> FastAPI:
    """JSON API only; the UI lives in sniff's Next.js app (apps/web)."""
    api = FastAPI(
        title="AI Content Detector",
        version=__version__,
        description="Evidence-first verdicts for AI-generated media. Day 1 skeleton.",
    )

    @api.get("/")
    @api.get("/health")
    def health():
        return {"status": "ok", "version": __version__}

    @api.post("/v1/analyze", response_model=Verdict)
    async def analyze_upload(
        file: UploadFile | None = None,
        text: str | None = Form(None),
        send_to_muse: bool = Form(False),
    ):
        data = await file.read() if file is not None else None
        filename = file.filename if file is not None else None
        return await analyze(
            data=data,
            filename=filename,
            text=text,
            send_to_muse=send_to_muse,
        )

    @api.post("/v1/analyze/text", response_model=Verdict)
    async def analyze_text(body: TextRequest):
        return await analyze(text=body.text, send_to_muse=body.send_to_muse)

    return api


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("ai_detector.app:app", host="0.0.0.0", port=7860)
