import cv2
import numpy as np
import json
from typing import Dict, Any, Union, Tuple

class OpenCVFaceMatcher:
    def __init__(self):
        # Initialize face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Initialize the SIFT feature detector
        self.sift = cv2.SIFT_create()
        
        # Parameters tuned for better accuracy
        self.min_neighbors = 5
        self.scale_factor = 1.1
        self.face_size = (150, 150)  # Normalized face size
        self.confidence_threshold = 65.0  # Lower means more strict matching
        
    def _load_and_prepare_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load, convert to grayscale and detect face in image."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(30, 30)
        )
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
            
        # Use the largest face found
        x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, self.face_size)
        
        # Apply histogram equalization for better recognition in different lighting
        face_roi = cv2.equalizeHist(face_roi)
        
        return face_roi, image    
    def compare_faces(self, img1_path: str, img2_path: str, threshold: float = 0.4) -> Dict[str, Any]:
        """Compare two faces and return similarity score."""
        try:
            # Process both images
            face1, _ = self._load_and_prepare_image(img1_path)
            face2, _ = self._load_and_prepare_image(img2_path)
            
            # Detect keypoints and compute descriptors
            keypoints1, descriptors1 = self.sift.detectAndCompute(face1, None)
            keypoints2, descriptors2 = self.sift.detectAndCompute(face2, None)
            
            if descriptors1 is None or descriptors2 is None:
                raise ValueError("Could not compute face features")
            
            # Create feature matcher
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(descriptors1, descriptors2, k=2)
            
            # Apply ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            # Calculate similarity score
            similarity = len(good_matches) / max(len(keypoints1), len(keypoints2))
            
            return {
                "verified": similarity >= threshold,
                "distance": 1 - similarity,
                "similarity": similarity,
                "threshold": threshold
            }
            
        except Exception as e:
            return {"error": str(e)}

    def analyze_face(self, img_path: str) -> Dict[str, Any]:
        """Analyze face in image and return facial attributes."""
        try:
            face_roi, image = self._load_and_prepare_image(img_path)
            
            # Basic face analysis
            height, width = image.shape[:2]
            face_analysis = {
                "success": True,
                "face_detected": True,
                "face_confidence": 100,  # OpenCV cascade doesn't provide confidence
                "dominant_gender": "Unknown",  # OpenCV basic doesn't do gender detection
                "dominant_gender_confidence": 0,
                "dominant_emotion": "Unknown",  # OpenCV basic doesn't do emotion detection
                "dominant_emotion_confidence": 0,
                "age": 0,  # OpenCV basic doesn't do age detection
                "age_confidence": 0,
                "region": {
                    "x": 0,
                    "y": 0,
                    "w": width,
                    "h": height
                }
            }
            
            return face_analysis
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
