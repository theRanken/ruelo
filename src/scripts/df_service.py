"""
df_service.py
FastAPI DeepFace service with:
- single-process model loading
- no temp-file deletion overhead
- base64 / URL / file path support
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

from deepface import DeepFace
from PIL import Image
from uvicorn.config import Config
from uvicorn.server import Server

import os, io, asyncio, base64, traceback, requests
import numpy as np  # <- IMPORTANT: use numpy arrays

app = FastAPI(title="DeepFace API Service")


# ---------- Models ----------
class VerifyPayload(BaseModel):
    img1: str
    img2: str
    model_name: str = "ArcFace"  # faster than Facenet512
    threshold: Optional[float] = None


class AnalyzePayload(BaseModel):
    img: str
    actions: Optional[List[str]] = None
    model_name: str = "ArcFace"


# ---------- Helpers ----------


def load_image(source: str) -> Image.Image:
    """
    Accepts:
    - data URLs: data:image/jpeg;base64,....
    - raw base64 (e.g. /9j/...)
    - http/https URLs
    - local filesystem paths
    """
    source = source.strip()

    # data URL
    if source.startswith("data:image"):
        header, b64data = source.split(",", 1)
        data = base64.b64decode(b64data)
        return Image.open(io.BytesIO(data))

    # raw base64 (common JPEG prefix)
    if source.startswith("/9j/"):
        data = base64.b64decode(source)
        return Image.open(io.BytesIO(data))

    # remote URL
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=5)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content))

    # local file path
    if os.path.exists(source):
        return Image.open(source)

    raise ValueError(f"Unsupported image source: {source[:50]}...")


def preprocess(img: Image.Image) -> np.ndarray:
    """
    Light resize to speed up inference, and return a NumPy array.
    DeepFace.verify / analyze are happy with np.ndarray inputs.
    """
    img = img.convert("RGB")
    img.thumbnail((1080, 1440))  # in-place
    return np.array(img)


# ---------- Routes ----------


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/verify")
async def verify(payload: VerifyPayload):
    try:
        # Convert both inputs to numpy arrays
        img1_arr = preprocess(load_image(payload.img1))
        img2_arr = preprocess(load_image(payload.img2))

        # Use positional args and np.ndarray images
        result = DeepFace.verify(
            img1_arr,
            img2_arr,
            model_name=payload.model_name,
            detector_backend="retinaface",
            enforce_detection=False,
        )

        if payload.threshold is not None:
            result["threshold_used"] = payload.threshold

        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )


@app.post("/analyze")
async def analyze(payload: AnalyzePayload):
    try:
        img_arr = preprocess(load_image(payload.img))

        actions = payload.actions or ["age", "gender", "emotion", "race"]

        # Also pass np.ndarray positionally here
        result = DeepFace.analyze(
            img_arr,
            actions=actions,
            enforce_detection=False,
            detector_backend="retinaface",
        )

        return JSONResponse(content={"result": result})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": traceback.format_exc(),
            },
        )


# ---------- Runner ----------


def run():
    port = int(os.getenv("DEEPFACE_PORT", "4800"))

    config = Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        reload=False,  # IMPORTANT: off in prod
        workers=1,  # single worker to avoid multiple model loads
    )
    server = Server(config)
    server.run()


if __name__ == "__main__":
    run()
