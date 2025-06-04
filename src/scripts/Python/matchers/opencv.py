import os
import cv2
import numpy as np
import base64
import tempfile
import time
from deepface import DeepFace


class OpenCVFaceMatcher:
    def __init__(self, detector_backend="mtcnn"):
        self.detector_backend = detector_backend.lower()
        self.recognizer_backend = "deepface"
        self.models_loaded = True
        self.detector_loaded = True

    def _load_image(self, image_source):
        if isinstance(image_source, str):
            if image_source.startswith("data:image/"):
                base64_string = image_source.split(",")[1]
                image_data = base64.b64decode(base64_string)
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not load image: {image_source}")
            return img
        raise TypeError("Invalid image source")

    def _extract_embedding_deepface(self, img):
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(tmp_file.name, img)
        tmp_file.close()
        try:
            embeddings = DeepFace.represent(
                img_path=tmp_file.name, model_name="Facenet512", enforce_detection=False
            )
            if isinstance(embeddings, list) and len(embeddings) > 0:
                return np.array(embeddings[0]["embedding"])
            else:
                raise ValueError(
                    "DeepFace.represent() failed to return a valid embedding."
                )
        finally:
            os.remove(tmp_file.name)

    def _calculate_similarity(self, embedding1, embedding2):
        norm1 = embedding1 / np.linalg.norm(embedding1)
        norm2 = embedding2 / np.linalg.norm(embedding2)
        return float(np.dot(norm1.flatten(), norm2.flatten()))

    def _get_confidence_label(self, confidence):
        if confidence >= 0.98:
            return "very_high"
        elif confidence >= 0.95:
            return "high"
        elif confidence >= 0.90:
            return "moderate"
        elif confidence >= 0.85:
            return "low"
        else:
            return "very_low"

    def compare_faces(self, image1_source, image2_source, threshold=None):
        start_time = time.perf_counter()
        try:
            if threshold is None:
                threshold = 0.45

            img1 = self._load_image(image1_source)
            img2 = self._load_image(image2_source)

            embedding1 = self._extract_embedding_deepface(img1)
            embedding2 = self._extract_embedding_deepface(img2)

            similarity = self._calculate_similarity(embedding1, embedding2)
            is_match = similarity >= threshold

            return {
                "message": (
                    "Result: Faces match!"
                    if is_match
                    else "Result: Faces do not match."
                ),
                "status": "success",
                "verified": bool(is_match),
                "confidence": round(similarity, 3),
                "confidence_level": self._get_confidence_label(similarity),
                "distance": round(1 - similarity, 6),
                "threshold": threshold,
                "model": f"DeepFace(Facenet512)",
                "recognizer_backend": "deepface",
                "match_time_seconds": round(time.perf_counter() - start_time, 4),
            }
        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "match_time_seconds": round(time.perf_counter() - start_time, 4),
            }

    def analyze_face(self, image_source):
        start_time = time.perf_counter()
        try:
            img = self._load_image(image_source)
            if img is None:
                raise ValueError("Could not load image for analysis.")

            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cv2.imwrite(tmp_file.name, img)
            tmp_file.close()

            analysis = DeepFace.analyze(
                img_path=tmp_file.name,
                actions=["age", "gender", "emotion", "race"],
                enforce_detection=True,
            )
            os.remove(tmp_file.name)

            result = analysis[0] if isinstance(analysis, list) else analysis
            result["status"] = "success"
            result["match_time_seconds"] = round(time.perf_counter() - start_time, 4)
            return result

        except Exception as e:
            return {
                "status": "error",
                "error_message": str(e),
                "match_time_seconds": round(time.perf_counter() - start_time, 4),
            }
