from face_lib import face_lib
import cv2
import numpy as np
import time

class FaceLibMatcher:
    def __init__(self):
        self.FL = face_lib()

    def compare_faces(self, image1_source, image2_source, threshold=0.6):
        """
        Compare two faces using Face Library recognition pipeline.
        Returns dict with status, verified, distance, and other info.
        """
        start_time = time.perf_counter()
        try:
            img1 = cv2.imread(image1_source)
            img2 = cv2.imread(image2_source)
            if img1 is None:
                raise ValueError(f"Could not load image from path: {image1_source}")
            if img2 is None:
                raise ValueError(f"Could not load image from path: {image2_source}")

            face_exist, no_faces_detected = self.FL.recognition_pipeline(img1, img2, threshold=threshold)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            return {
                "status": "success",
                "verified": face_exist,
                "faces_detected": no_faces_detected,
                "threshold": threshold,
                "execution_time_seconds": execution_time
            }
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return {
                "status": "error",
                "error_message": str(e),
                "execution_time_seconds": execution_time
            }

    def analyze_face(self, image_source):
        """
        Analyze face attributes using Face Library.
        Returns dict with status and face count.
        """
        start_time = time.perf_counter()
        try:
            img = cv2.imread(image_source)
            if img is None:
                raise ValueError(f"Could not load image from path: {image_source}")

            no_of_faces, faces_coords = self.FL.faces_locations(img)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            return {
                "status": "success",
                "face_count": no_of_faces,
                "face_locations": faces_coords,
                "execution_time_seconds": execution_time
            }
        except Exception as e:
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            return {
                "status": "error",
                "error_message": str(e),
                "execution_time_seconds": execution_time
            }
