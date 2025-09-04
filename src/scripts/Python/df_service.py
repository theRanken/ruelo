"""
deepface_api_service.py
A FastAPI server for DeepFace operations with singleton model loading and robust error handling.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from deepface import DeepFace
import uvicorn, traceback, os, socket
from typing import Optional
from PIL import Image

app = FastAPI(title="DeepFace API Service")


class DeepFaceSingleton:
    _instance = None
    _models = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            try:
                # Load models once for all requests
                cls._models = DeepFace.build_model("Facenet512")
            except Exception as e:
                print(f"Error loading DeepFace models: {e}")
                cls._models = None
        return cls._instance

    def get_models(self):
        return self._models


def is_uvicorn_running(host="127.0.0.1", port=4800):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1)
            s.connect((host, port))
            return True
        except Exception:
            return False


# Instantiate singleton
deepface_singleton = DeepFaceSingleton()


@app.post("/verify")
async def verify(
    img1: str = Body(...),
    img2: str = Body(...),
    model_name: Optional[str] = "Facenet512",
):
    try:
        # Load images from file paths
        if not os.path.exists(img1) or not os.path.exists(img2):
            raise Exception(f"Image file(s) not found: {img1}, {img2}")
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=model_name,
            enforce_detection=False,
        )
        # Delete files after use
        if os.path.exists(img1):
            os.remove(img1)
        if os.path.exists(img2):
            os.remove(img2)
        return JSONResponse(content={"result": result})
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})


@app.post("/analyze")
async def analyze(
    img: str = Body(...),
    actions: Optional[str] = "['age', 'gender', 'emotion', 'race']",
    model_name: Optional[str] = "VGG-Face",
):
    try:
        # Load image from file path
        if not os.path.exists(img):
            raise Exception(f"Image file not found: {img}")
        actions_list = eval(actions) if isinstance(actions, str) else actions
        result = DeepFace.analyze(
            img_path=img,
            actions=actions_list,
            model_name=model_name,
            enforce_detection=False,
        )
        # Delete file after use
        if os.path.exists(img):
            os.remove(img)
        return JSONResponse(content={"result": result})
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace": tb})


@app.get("/health")
async def health():
    # Simple health check endpoint
    models_loaded = deepface_singleton.get_models() is not None
    return {"status": "ok", "models_loaded": models_loaded}


if __name__ == "__main__":

    if not is_uvicorn_running():
        log_config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../log_config.yaml")
        )
        uvicorn.run(app, host="0.0.0.0", port=4800, log_config=log_config_path)
