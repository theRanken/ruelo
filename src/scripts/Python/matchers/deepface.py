import cv2
import numpy as np
import base64
from deepface import DeepFace
import tempfile
import os

class DeepFaceMatcher:
    def __init__(self, model_name='Facenet512', distance_metric='cosine', analyze_actions=None):
        """
        Initialize DeepFaceMatcher.

        Args:
            model_name (str): DeepFace model to use for verification (e.g. Facenet).
            distance_metric (str): Distance metric to use (cosine, euclidean, etc.).
            analyze_actions (list): List of attributes to analyze, e.g. ['age', 'gender', 'race', 'emotion'].
                                    If None, defaults to ['age', 'gender', 'race', 'emotion'].
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.analyze_actions = analyze_actions or ['age', 'gender', 'race', 'emotion']

    def _base64_to_image(self, b64_string: str) -> np.ndarray:
        if b64_string.startswith('data:image'):
            b64_string = b64_string.split(',')[1]
        img_data = base64.b64decode(b64_string)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return img

    def _prepare_image(self, image) -> str:
        if os.path.isfile(image):
            return image

        img = self._base64_to_image(image)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        cv2.imwrite(tmp_file.name, img)
        tmp_file.close()
        return tmp_file.name

    def compare_faces(self, img1: str, img2: str, threshold:str = None) -> dict:
        temp_files = []
        try:
            if threshold is None:
                threshold = 0.45
            img1_path = self._prepare_image(img1)
            if not os.path.isfile(img1):
                temp_files.append(img1_path)
            img2_path = self._prepare_image(img2)
            if not os.path.isfile(img2):
                temp_files.append(img2_path)

            result = DeepFace.verify(
                img1_path,
                img2_path,
                model_name=self.model_name,
                distance_metric=self.distance_metric,
                enforce_detection=False,
                threshold=threshold
            )
            return result
        finally:
            for file in temp_files:
                try:
                    os.remove(file)
                except Exception:
                    pass

    def analyze_face(self, img: str) -> dict:
        """
        Analyze attributes of a face image (age, gender, race, emotion, etc.)

        Args:
            img (str): File path or base64 image.

        Returns:
            dict: Analysis results containing requested attributes.
        """
        temp_files = []
        try:
            img_path = self._prepare_image(img)
            if not os.path.isfile(img):
                temp_files.append(img_path)

            analysis = DeepFace.analyze(
                img_path,
                actions=self.analyze_actions,
                enforce_detection=True
            )
            return analysis
        finally:
            for file in temp_files:
                try:
                    os.remove(file)
                except Exception:
                    pass
