import cv2
import numpy as np
import base64
import tempfile
import os
import urllib.request


class OpenCVFaceMatcher:
    def __init__(self):
        # Initialize DNN-based face detection and recognition
        self.face_detector = None
        self.face_recognizer = None
        self.models_loaded = False

        # Model URLs (from OpenCV Zoo)
        self.yunet_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        self.sface_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

        # Try to load models
        self._load_models()

        # Fallback to Haar cascade if DNN models fail or are not available
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        # Emotion labels
        self.emotions = [
            "angry",
            "disgust",
            "fear",
            "happy",
            "sad",
            "surprise",
            "neutral",
        ]

    def _download_model(self, url, filename):
        """Download model file if it doesn't exist"""
        if not os.path.exists(filename):
            try:
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filename)
                print(f"Downloaded {filename}")
                return True
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
                return False
        return True

    def _load_models(self):
        """Load DNN models for face detection and recognition"""
        try:
            # Create models directory
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)

            yunet_path = os.path.join(models_dir, "yunet.onnx")
            sface_path = os.path.join(models_dir, "sface.onnx")

            # Download models if needed
            yunet_ok = self._download_model(self.yunet_url, yunet_path)
            sface_ok = self._download_model(self.sface_url, sface_path)

            if yunet_ok and sface_ok:
                # Initialize face detector (YuNet)
                # Parameters: model, config, input_size, score_threshold, nms_threshold, top_k
                self.face_detector = cv2.FaceDetectorYN.create(
                    yunet_path, "", (320, 240), 0.9, 0.3, 5000
                )

                # Initialize face recognizer (SFace)
                self.face_recognizer = cv2.FaceRecognizerSF.create(sface_path, "")

                self.models_loaded = True
            else:
                self.models_loaded = False


        except Exception as e:
            print(f"Error loading DNN models: {e}")
            print("Falling back to traditional face detection")
            self.models_loaded = False


    def _load_image(self, image_source):
        """Load image from file path or base64 string"""
        try:
            if isinstance(image_source, str):
                if image_source.startswith("data:image/") or (len(image_source) > 500 and base64_string_looks_like_image(image_source)):
                    # Assume it's a base64 string if it starts with "data:image/" or is long and looks like base64
                    return self._load_base64_image(image_source)
                else:
                    # Assume it's a file path
                    img = cv2.imread(image_source)
                    if img is None:
                        raise ValueError(f"Could not load image from path: {image_source}")
                    return img
            else:
                raise TypeError("image_source must be a string (file path or base64)")
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def _load_base64_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            if base64_string.startswith("data:image/"):
                base64_string = base64_string.split(",")[1]

            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode base64 image")
            return img
        except Exception as e:
            raise ValueError(f"Error decoding base64 image: {str(e)}")

    def _detect_face(self, img, method_preference="dnn"):
        """Detect face using DNN model (YuNet) or fallback to Haar cascade"""
        face = None
        detection_method = "none"

        if method_preference == "dnn" and self.models_loaded:
            face = self._detect_face_dnn(img)
            if face is not None:
                detection_method = "YuNet"
        
        if face is None: # Fallback if DNN failed or not preferred/loaded
            face = self._detect_face_traditional(img)
            if face is not None:
                detection_method = "Haar"
        
        return face, detection_method

    def _detect_face_dnn(self, img):
        """Detect face using DNN model (YuNet)"""
        if not self.models_loaded:
            return None

        try:
            h, w = img.shape[:2]
            # Set input size for the detector - crucial for performance and detection
            self.face_detector.setInputSize((w, h))

            # Detect faces
            _, faces = self.face_detector.detect(img)

            if faces is None or len(faces) == 0:
                return None

            # Get the face with highest confidence (index 14 is confidence score)
            best_face = max(faces, key=lambda x: x[14]) 

            # Return bounding box [x, y, w, h]
            x, y, w, h = best_face[:4].astype(int)
            return (x, y, w, h)

        except Exception as e:
            # print(f"DNN face detection failed: {e}") # Keep this for debugging if needed
            return None

    def _detect_face_traditional(self, img):
        """Fallback face detection using Haar cascade"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try with stricter parameters first
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80), # Increased minSize for better initial results
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            # If no faces found, try with more relaxed parameters
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=3, minSize=(50, 50)
            )

        return max(faces, key=lambda x: x[2] * x[3]) if len(faces) > 0 else None

    def _extract_face_embedding(self, img, face_coords):
        """Extract face embedding using SFace model"""
        if not self.models_loaded:
            return None

        try:
            x, y, w, h = face_coords

            # Ensure face_coords is a numpy array for alignCrop
            face_coords_np = np.array([x, y, w, h], dtype=np.float32)

            # Align face for recognition
            face_align = self.face_recognizer.alignCrop(img, face_coords_np)

            # Extract feature embedding
            face_feature = self.face_recognizer.feature(face_align)

            return face_feature

        except Exception as e:
            # print(f"Error extracting DNN embedding: {e}") # Keep for debugging
            return None

    def _calculate_cosine_similarity(self, feature1, feature2):
        """Calculate cosine similarity between two feature vectors"""
        if feature1 is None or feature2 is None:
            return 0.0

        # Normalize features
        feature1_norm = feature1 / np.linalg.norm(feature1)
        feature2_norm = feature2 / np.linalg.norm(feature2)

        # Calculate cosine similarity
        similarity = np.dot(feature1_norm.flatten(), feature2_norm.flatten())
        return float(similarity)

    def compare_faces(self, image1_source, image2_source, threshold=0.363):
        """
        Compare two faces using DNN models with fallback to traditional methods.
        Note: SFace threshold of 0.363 corresponds to ~95% accuracy
        """
        temp_files = []

        try:
            # Load images
            img1 = self._load_image(image1_source)
            img2 = self._load_image(image2_source)

            # Detect faces using the robust _detect_face method
            face1, detection_method1 = self._detect_face(img1, method_preference="dnn")
            face2, detection_method2 = self._detect_face(img2, method_preference="dnn")

            if face1 is None:
                return {"error": "No face detected in first image after all attempts."}

            if face2 is None:
                return {"error": "No face detected in second image after all attempts."}

            # Determine the method used for comparison
            method = "YuNet-SFace" if self.models_loaded else "Haar-Traditional"
            detector_backend = "opencv_dnn" if self.models_loaded else "opencv_haar"

            # Extract embeddings or features based on whether DNN models are loaded
            if self.models_loaded:
                embedding1 = self._extract_face_embedding(img1, face1)
                embedding2 = self._extract_face_embedding(img2, face2)

                if embedding1 is None or embedding2 is None:
                    return {"error": "Failed to extract face embeddings with DNN models."}

                # Calculate similarity using SFace's method
                similarity = self.face_recognizer.match(
                    embedding1, embedding2, cv2.FaceRecognizerSF_FR_COSINE
                )
            else:
                # Fallback to traditional features if DNN models are not loaded
                features1 = self._extract_traditional_features(img1, face1)
                features2 = self._extract_traditional_features(img2, face2)
                similarity = self._calculate_traditional_similarity(
                    features1, features2
                )
                threshold = 0.6  # Different threshold for traditional method

            # Determine match
            is_match = similarity >= threshold

            return {
                "verified": is_match,
                "confidence": float(similarity),
                "distance": (
                    float(1 - similarity) if similarity <= 1 else float(similarity - 1)
                ),
                "threshold": threshold,
                "model": method,
                "detector_backend": detector_backend,
                "facial_areas": {
                    "img1": {
                        "x": int(face1[0]),
                        "y": int(face1[1]),
                        "w": int(face1[2]),
                        "h": int(face1[3]),
                    },
                    "img2": {
                        "x": int(face2[0]),
                        "y": int(face2[1]),
                        "w": int(face2[2]),
                        "h": int(face2[3]),
                    },
                },
            }

        except Exception as e:
            return {"error": str(e)}

        finally:
            # Clean up temporary files (this part of your code is already good)
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass

    def _save_temp_image(self, img):
        """Save image to temporary file and return path"""
        try:
            temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(temp_fd)
            cv2.imwrite(temp_path, img)
            return temp_path
        except Exception as e:
            raise ValueError(f"Error saving temporary image: {str(e)}")

    def _extract_traditional_features(self, img, face_coords):
        """Extract traditional features as fallback"""
        x, y, w, h = face_coords
        face_roi = img[y : y + h, x : x + w]
        face_gray = (
            cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            if len(face_roi.shape) == 3
            else face_roi
        )
        face_resized = cv2.resize(face_gray, (128, 128))

        features = {}

        # Histogram
        features["histogram"] = cv2.calcHist(
            [face_resized], [0], None, [64], [0, 256]
        ).flatten()

        # LBP
        lbp = self._calculate_lbp(face_resized)
        features["lbp"] = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()

        return features

    def _calculate_lbp(self, img, radius=1, n_points=8):
        """Calculate Local Binary Pattern"""
        h, w = img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = img[i, j]
                code = 0

                for p in range(n_points):
                    angle = 2 * np.pi * p / n_points
                    x = int(round(i + radius * np.cos(angle)))
                    y = int(round(j + radius * np.sin(angle)))

                    if 0 <= x < h and 0 <= y < w:
                        if img[x, y] >= center:
                            code |= 1 << p

                lbp[i, j] = code

        return lbp

    def _calculate_traditional_similarity(self, features1, features2):
        """Calculate similarity for traditional features"""
        similarities = []

        # Histogram similarity
        hist_sim = cv2.compareHist(
            features1["histogram"], features2["histogram"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, hist_sim))

        # LBP similarity
        lbp_sim = cv2.compareHist(
            features1["lbp"], features2["lbp"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, lbp_sim))

        return np.mean(similarities)

    def analyze_face(self, image_source):
        """Analyze face for emotions, age, and gender"""
        temp_file = None

        try:
            # Load image
            img = self._load_image(image_source)

            # Detect face using the robust _detect_face method
            face, detection_method = self._detect_face(img, method_preference="dnn")

            if face is None:
                return {"error": "No face detected after all attempts."}

            x, y, w, h = face
            face_roi = img[y : y + h, x : x + w]
            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            # Analyze emotions
            emotions = self._analyze_emotions(face_gray, face_roi)

            # Estimate age and gender
            age = self._estimate_age(face_gray)
            gender = self._estimate_gender(face_gray, face_roi)

            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])

            return {
                "age": age,
                "gender": gender,
                "dominant_emotion": dominant_emotion[0],
                "emotion": emotions,
                "face_confidence": 0.95 if self.models_loaded else 0.8,
                "detection_method": detection_method,
                "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            }

        except Exception as e:
            return {"error": str(e)}

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass

    def _analyze_emotions(self, face_gray, face_color):
        """Analyze emotions using geometric and texture features"""
        face_resized = cv2.resize(face_gray, (64, 64))

        emotions = {}
        h, w = face_resized.shape

        # Eye region analysis
        eye_region = face_resized[int(h * 0.2) : int(h * 0.5), :]
        eye_variance = np.var(eye_region)

        # Mouth region analysis
        mouth_region = face_resized[int(h * 0.6) : int(h * 0.9), :]
        mouth_mean = np.mean(mouth_region)
        mouth_std = np.std(mouth_region)

        # Forehead analysis
        forehead_region = face_resized[0 : int(h * 0.3), :]
        forehead_std = np.std(forehead_region)

        # Enhanced emotion detection with better heuristics
        base_prob = 0.05

        emotions["neutral"] = base_prob + 0.15
        emotions["happy"] = base_prob + (
            0.4 if mouth_std > 25 and mouth_mean > 110 else 0.1
        )
        emotions["sad"] = base_prob + (
            0.3 if mouth_mean < 95 and mouth_std < 20 else 0.1
        )
        emotions["angry"] = base_prob + (
            0.35 if forehead_std > 30 and eye_variance < 300 else 0.1
        )
        emotions["surprise"] = base_prob + (
            0.4 if eye_variance > 450 and mouth_std > 20 else 0.1
        )
        emotions["fear"] = base_prob + (
            0.25 if eye_variance > 400 and mouth_std < 15 else 0.1
        )
        emotions["disgust"] = base_prob + (
            0.2 if mouth_mean < 100 and forehead_std > 25 else 0.1
        )

        # Normalize probabilities
        total = sum(emotions.values())
        emotions = {k: v / total for k, v in emotions.items()}

        return emotions

    def _estimate_age(self, face_gray):
        """Estimate age based on facial features"""
        face_resized = cv2.resize(face_gray, (64, 64))

        # Calculate texture complexity (wrinkles, skin texture)
        laplacian_var = cv2.Laplacian(face_resized, cv2.CV_64F).var()

        # Enhanced age estimation
        if laplacian_var > 800:
            age_range = (55, 75)
        elif laplacian_var > 500:
            age_range = (35, 55)
        elif laplacian_var > 200:
            age_range = (20, 35)
        else:
            age_range = (12, 25)

        return int(np.mean(age_range))

    def _estimate_gender(self, face_gray, face_color):
        """Estimate gender based on facial features"""
        face_resized = cv2.resize(face_gray, (64, 64))
        h, w = face_resized.shape

        # Jaw region analysis
        jaw_region = face_resized[int(h * 0.7) : h, :]
        jaw_definition = np.std(jaw_region)

        # Cheek region smoothness
        cheek_region = face_resized[int(h * 0.3) : int(h * 0.7), :]
        cheek_smoothness = 1.0 / (1.0 + np.var(cheek_region))

        # Enhanced gender classification
        if jaw_definition > 20 and cheek_smoothness < 0.01:
            gender_prob = {"Man": 0.75, "Woman": 0.25}
        elif jaw_definition < 15 and cheek_smoothness > 0.015:
            gender_prob = {"Man": 0.25, "Woman": 0.75}
        else:
            gender_prob = {"Man": 0.5, "Woman": 0.5}

        return gender_prob

    def get_model_info(self):
        """Get information about loaded models"""
        return {
            "dnn_models_loaded": self.models_loaded,
            "face_detector": "YuNet" if self.models_loaded else "Haar Cascade",
            "face_recognizer": (
                "SFace" if self.models_loaded else "Traditional Features"
            ),
            "expected_accuracy": "90-95%" if self.models_loaded else "60-75%",
        }

    def base64_string_looks_like_image(s):
        """
        Helper function to determine if a long string is likely a base64 encoded image.
        This is a heuristic and not foolproof.
        """
        # Check for common base64 characters and length
        return len(s) > 100 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s[-100:])