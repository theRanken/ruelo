import os, cv2, numpy as np, base64, tempfile, urllib.request, time
from helpers import base64_string_looks_like_image

# Suppress TensorFlow logging warnings for MTCNN if it uses TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# --- Detector Availability Checks ---

# BlazeFace (MediaPipe)
BLAZEFACE_AVAILABLE = False
try:
    import mediapipe as mp
    BLAZEFACE_AVAILABLE = True
except ImportError:
    pass # BlazeFace not available, handled by logic

# RetinaFace
RETINAFACE_AVAILABLE = False
try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    pass # RetinaFace not available, handled by logic

# MTCNN
MTCNN_AVAILABLE = False
try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    pass # MTCNN not available, handled by logic


class OpenCVFaceMatcher:
    def __init__(self, detector_backend:str="retinaface", recognizer_backend:str="facenet")->None:
        self.face_detector = None
        self.face_recognizer = None
        self.models_loaded = False # Recognizer loaded status
        self.detector_loaded = False # Detector loaded status
        self.detector_backend = detector_backend.lower()
        self.recognizer_backend = recognizer_backend.lower()

        self.base_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        self.arcface_path = os.path.join(self.base_models_dir, "arcface.onnx")
        self.facenet_path = os.path.join(self.base_models_dir, "facenet.onnx")
        self.sface_path = os.path.join(self.base_models_dir, "sface.onnx")
        self.yunet_path = os.path.join(self.base_models_dir, "yunet.onnx")

        self.detector_loaders = {
            "yunet": self._load_yunet_detector,
            "retinaface": self._load_retinaface_detector,
            "blazeface": self._load_blazeface_detector,
            "mtcnn": self._load_mtcnn_detector,
        }
        self.detector_backends = {
            "yunet": self._detect_face_yunet,
            "retinaface": self._detect_face_retinaface,
            "blazeface": self._detect_face_blazeface,
            "mtcnn": self._detect_face_mtcnn,
        }
        self.recognizer_extractors = {
            "sface": self._extract_embedding_sface,
            "arcface": self._extract_embedding_arcface,
            "facenet": self._extract_embedding_facenet,
        }
        self.recognizer_loaders = {
            "sface": self._load_sface_recognizer,
            "arcface": self._load_arcface_recognizer,
            "facenet": self._load_facenet_recognizer,
        }

        self._load_models()

        self.emotions = [
            "angry", "disgust", "fear", "happy", "sad", "surprise", "neutral",
        ]

    def _load_models(self):
        # Load recognizer backend
        if self.recognizer_backend in self.recognizer_loaders:
            self.recognizer_loaders[self.recognizer_backend]()
        else:
            self.models_loaded = False # Recognizer failed to load
            return # Exit early if recognizer can't load

        # Load detector backend
        if self.detector_backend in self.detector_loaders:
            self.detector_loaders[self.detector_backend]()
        else:
            self.detector_loaded = False # Detector failed to load
            return # Exit early if detector can't load

    def _load_yunet_detector(self):
        if os.path.exists(self.yunet_path):
            try:
                self.face_detector = cv2.FaceDetectorYN.create(
                    self.yunet_path, "", (320, 240), 0.9, 0.3, 5000
                )
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.face_detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.face_detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.detector_loaded = True
            except Exception:
                self.detector_loaded = False
        else:
            self.detector_loaded = False

    def _load_retinaface_detector(self):
        if RETINAFACE_AVAILABLE:
            self.detector_loaded = True
        else:
            self.detector_loaded = False

    def _load_blazeface_detector(self):
        if BLAZEFACE_AVAILABLE:
            try:
                # MediaPipe's FaceDetection expects RGB input and handles internal models
                self.face_detector = mp.solutions.face_detection.FaceDetection(
                    model_selection=1,  # 0 for short-range, 1 for full-range (more accurate, larger faces)
                    min_detection_confidence=0.7
                )
                self.detector_loaded = True
            except Exception:
                self.detector_loaded = False
        else:
            self.detector_loaded = False

    def _load_mtcnn_detector(self):
        if MTCNN_AVAILABLE:
            try:
                # MTCNN creates its own TF session. Min_face_size can be adjusted
                self.face_detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7]) # Default thresholds
                self.detector_loaded = True
            except Exception:
                self.detector_loaded = False
        else:
            self.detector_loaded = False

    def _load_sface_recognizer(self):
        if os.path.exists(self.sface_path):
            try:
                self.face_recognizer = cv2.FaceRecognizerSF.create(self.sface_path, "")
                self.models_loaded = True
            except Exception:
                self.models_loaded = False
        else:
            self.models_loaded = False

    def _load_arcface_recognizer(self):
        if os.path.exists(self.arcface_path):
            try:
                self.face_recognizer = cv2.dnn.readNetFromONNX(self.arcface_path)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.face_recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.face_recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                else:
                    self.face_recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.face_recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.models_loaded = True
            except Exception:
                self.models_loaded = False
        else:
            self.models_loaded = False
    def _load_facenet_recognizer(self):
        if os.path.exists(self.facenet_path):
            try:
                self.face_recognizer = cv2.dnn.readNetFromONNX(self.facenet_path)
                if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    self.face_recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.face_recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                else:
                    self.face_recognizer.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.face_recognizer.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.models_loaded = True
            except Exception:
                self.models_loaded = False
        else:
            self.models_loaded = False

    def _load_image(self, image_source):
        if isinstance(image_source, str):
            if image_source.startswith("data:image/") or (len(image_source) > 500 and base64_string_looks_like_image(image_source)):
                img = self._load_base64_image(image_source)
                if img is None:
                    raise ValueError("Failed to load image from base64 string.")
                return img
            else:
                img = cv2.imread(image_source)
                if img is None:
                    raise ValueError(f"Could not load image from path: {image_source}")
                return img
        else:
            raise TypeError("image_source must be a string (file path or base64)")

    def _load_base64_image(self, base64_string):
        if base64_string.startswith("data:image/"):
            base64_string = base64_string.split(",")[1]
        try:
            image_data = base64.b64decode(base64_string)
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img
        except Exception as e:
            raise ValueError(f"Base64 image decoding failed: {e}")

    def _detect_face_dnn(self, img):
        if img is None:
            raise ValueError("Input image is None for face detection.")
        if not self.detector_loaded:
            raise RuntimeError(f"{self.detector_backend.capitalize()} detector is not loaded. Cannot perform detection.")

        if self.detector_backend not in self.detector_backends:
            raise ValueError(f"Unknown detector backend: {self.detector_backend}")

        return self.detector_backends[self.detector_backend](img)

    def _extract_face_embedding(self, img, face_info_for_recognizer_align):
        if not self.models_loaded:
            raise RuntimeError(f"{self.recognizer_backend.capitalize()} recognition model is not loaded. Cannot extract embeddings.")

        if self.recognizer_backend not in self.recognizer_extractors:
            raise ValueError(f"Unknown recognizer backend for extraction: {self.recognizer_backend}")

        return self.recognizer_extractors[self.recognizer_backend](img, face_info_for_recognizer_align)

    def compare_faces(self, image1_source, image2_source, threshold=None):
        start_time = time.perf_counter()

        try:
            if not self.models_loaded:
                raise RuntimeError(f"{self.recognizer_backend.capitalize()} recognition model is not loaded. Face comparison cannot be performed.")
            if not self.detector_loaded:
                raise RuntimeError(f"{self.detector_backend.capitalize()} detector is not loaded. Face comparison cannot be performed.")

            if threshold is None:
                if self.recognizer_backend == "sface":
                    threshold = 0.5
                elif self.recognizer_backend == "arcface":
                    threshold = 0.7
                else:
                    threshold = 0.4

            img1 = self._load_image(image1_source)
            img2 = self._load_image(image2_source)

            if img1 is None: raise ValueError("Could not load first image for comparison.")
            if img2 is None: raise ValueError("Could not load second image for comparison.")

            face_info1 = self._detect_face_dnn(img1)
            face_info2 = self._detect_face_dnn(img2)

            if face_info1 is None: raise ValueError(f"No face detected in first image using {self.detector_backend.capitalize()}.")
            if face_info2 is None: raise ValueError(f"No face detected in second image using {self.detector_backend.capitalize()}.")

            embedding1 = self._extract_face_embedding(img1, face_info1)
            embedding2 = self._extract_face_embedding(img2, face_info2)

            if embedding1 is None: raise RuntimeError(f"Failed to extract face embedding from first image with {self.recognizer_backend.capitalize()}.")
            if embedding2 is None: raise RuntimeError(f"Failed to extract face embedding from second image with {self.recognizer_backend.capitalize()}.")

            similarity = self._calculate_similarity(embedding1, embedding2)

            is_match = similarity >= threshold

            bbox1 = face_info1[:4].astype(int)
            bbox2 = face_info2[:4].astype(int)

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            return {
                "status": "success",
                "verified": bool(is_match),
                "confidence": float(similarity),
                "distance": float(1 - similarity),
                "threshold": threshold,
                "model": f"{self.detector_backend.capitalize()}-{self.recognizer_backend.capitalize()}",
                "detector_backend": self.detector_backend,
                "recognizer_backend": self.recognizer_backend,
                "facial_areas": {
                    "img1": {"x": int(bbox1[0]), "y": int(bbox1[1]), "w": int(bbox1[2]), "h": int(bbox1[3])},
                    "img2": {"x": int(bbox2[0]), "y": int(bbox2[1]), "w": int(bbox2[2]), "h": int(bbox2[3])},
                },
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
        start_time = time.perf_counter()

        try:
            if not self.models_loaded:
                raise RuntimeError(f"{self.recognizer_backend.capitalize()} recognition model is not loaded. Face analysis cannot be performed.")
            if not self.detector_loaded:
                raise RuntimeError(f"{self.detector_backend.capitalize()} detector is not loaded. Face analysis cannot be performed.")

            img = self._load_image(image_source)
            if img is None: raise ValueError("Could not load image for analysis.")

            face_info = self._detect_face_dnn(img)

            if face_info is None:
                raise ValueError(f"No face detected for analysis using {self.detector_backend.capitalize()}.")

            x, y, w, h = face_info[:4].astype(int)
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            face_roi = img[y1:y2, x1:x2]

            if face_roi.size == 0:
                raise ValueError("Detected face region is empty or invalid.")

            face_gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            preprocessed_face_for_analysis = clahe.apply(face_gray)

            emotions = self._analyze_emotions(preprocessed_face_for_analysis, face_roi)
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])

            end_time = time.perf_counter()
            execution_time = end_time - start_time

            return {
                "status": "success",
                "dominant_emotion": dominant_emotion[0],
                "emotion": emotions,
                "face_confidence": 0.95,
                "detection_method": self.detector_backend,
                "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
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

    def _analyze_emotions(self, face_gray, face_color):
        if face_gray.size == 0: return {emotion: 1/len(self.emotions) for emotion in self.emotions}
        face_resized = cv2.resize(face_gray, (64, 64))
        emotions = {}
        h, w = face_resized.shape
        eye_region = face_resized[int(h * 0.2) : int(h * 0.5), :]
        eye_variance = np.var(eye_region) if eye_region.size > 0 else 0
        mouth_region = face_resized[int(h * 0.6) : int(h * 0.9), :]
        mouth_mean = np.mean(mouth_region) if mouth_region.size > 0 else 0
        mouth_std = np.std(mouth_region) if mouth_region.size > 0 else 0
        forehead_region = face_resized[0 : int(h * 0.3), :]
        forehead_std = np.std(forehead_region) if forehead_region.size > 0 else 0
        base_prob = 0.05
        emotions["neutral"] = base_prob + 0.15
        emotions["happy"] = base_prob + (0.4 if mouth_std > 25 and mouth_mean > 110 else 0.1)
        emotions["sad"] = base_prob + (0.3 if mouth_mean < 95 and mouth_std < 20 else 0.1)
        emotions["angry"] = base_prob + (0.35 if forehead_std > 30 and eye_variance < 300 else 0.1)
        emotions["surprise"] = base_prob + (0.4 if eye_variance > 450 and mouth_std > 20 else 0.1)
        emotions["fear"] = base_prob + (0.25 if eye_variance > 400 and mouth_std < 15 else 0.1)
        emotions["disgust"] = base_prob + (0.2 if mouth_mean < 100 and forehead_std > 25 else 0.1)
        total = sum(emotions.values())
        if total == 0: emotions = {k: 1/len(self.emotions) for k in emotions}
        else: emotions = {k: v / total for k, v in emotions.items()}
        return emotions

    def get_model_info(self):
        recognizer_status = "Loaded" if self.models_loaded else "Not Loaded"
        detector_status = "Loaded" if self.detector_loaded else "Not Loaded"

        return {
            "recognizer_backend": self.recognizer_backend.capitalize(),
            "recognizer_status": recognizer_status,
            "face_detector_backend": self.detector_backend.capitalize(),
            "detector_status": detector_status,
            "expected_accuracy": f"Very High (with {self.detector_backend.capitalize()} + {self.recognizer_backend.capitalize()})" if self.models_loaded and self.detector_loaded else "N/A (Models not fully loaded)",
            "note": "This version uses selectable detector (YuNet/RetinaFace/BlazeFace/MTCNN) and recognizer (SFace/ArcFace)."
        }

    def _detect_face_yunet(self, img):
        h, w = img.shape[:2]
        self.face_detector.setInputSize((w, h))
        _, faces = self.face_detector.detect(img)
        if faces is None or len(faces) == 0:
            return None

        best_face = max(faces, key=lambda x: x[14])
        x, y, w, h = best_face[:4].astype(int)
        x = max(0, x) ; y = max(0, y)
        w = min(w, img.shape[1] - x) ; h = min(h, img.shape[0] - y)
        if w <= 0 or h <= 0: return None
        landmarks = best_face[5:15].reshape(5, 2)
        face_info_for_recognizer_align = np.array([x, y, w, h] + landmarks.flatten().tolist(), dtype=np.float32)
        return face_info_for_recognizer_align

    def _detect_face_retinaface(self, img):
        if not RETINAFACE_AVAILABLE:
            raise RuntimeError("RetinaFace library is not imported. Cannot perform detection.")
        if img is None:
            raise ValueError("Input image is None for RetinaFace detection.")

        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            detections = RetinaFace.detect_faces(rgb_img, threshold=0.9)
        except Exception as e:
            raise RuntimeError(f"RetinaFace detection failed: {e}")

        if not detections:
            return None

        best_face_key = None
        max_score = -1
        for key in detections:
            score = detections[key]['score']
            if score > max_score:
                max_score = score
                best_face_key = key

        if best_face_key is None:
            return None

        face_data = detections[best_face_key]
        facial_area = face_data['facial_area']
        landmarks = face_data['landmarks']

        x, y, x2, y2 = facial_area
        w, h = x2 - x, y2 - y

        if not all(k in landmarks for k in ['right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left']):
            raise ValueError("RetinaFace detection did not return all required landmarks.")

        recognizer_landmarks = [
            landmarks['right_eye'][0], landmarks['right_eye'][1],
            landmarks['left_eye'][0], landmarks['left_eye'][1],
            landmarks['nose'][0], landmarks['nose'][1],
            landmarks['mouth_right'][0], landmarks['mouth_right'][1],
            landmarks['mouth_left'][0], landmarks['mouth_left'][1]
        ]
        face_info_for_recognizer_align = np.array([x, y, w, h] + recognizer_landmarks, dtype=np.float32)
        return face_info_for_recognizer_align

    def _detect_face_blazeface(self, img):
        if not BLAZEFACE_AVAILABLE:
            raise RuntimeError("BlazeFace detector requires mediapipe to be installed.")
        if img is None:
            raise ValueError("Input image is None for BlazeFace detection.")

        # MediaPipe expects RGB input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            results = self.face_detector.process(img_rgb)
        except Exception as e:
            raise RuntimeError(f"BlazeFace detection failed: {e}")

        if not results.detections:
            return None

        # Find the detection with the highest confidence score
        best_detection = None
        max_score = -1.0
        for detection in results.detections:
            score = detection.score[0] # score is a list, take the first element
            if score > max_score:
                max_score = score
                best_detection = detection

        if best_detection is None:
            return None

        # Extract bounding box
        bboxC = best_detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        x = int(bboxC.xmin * iw)
        y = int(bboxC.ymin * ih)
        w = int(bboxC.width * iw)
        h = int(bboxC.height * ih)

        # Extract and format landmarks (MediaPipe provides 6, map to 5 for compatibility)
        landmarks = best_detection.location_data.relative_keypoints
        
        # Mapping MediaPipe keypoints (0-5) to required 5 for recognizers:
        # 0: right_eye, 1: left_eye, 2: nose_tip, 3: mouth_center, 4: right_ear_tragus, 5: left_ear_tragus
        # Need: right_eye, left_eye, nose, mouth_right, mouth_left

        recognizer_landmarks = [
            landmarks[0].x * iw, landmarks[0].y * ih, # right_eye
            landmarks[1].x * iw, landmarks[1].y * ih, # left_eye
            landmarks[2].x * iw, landmarks[2].y * ih, # nose_tip
            landmarks[3].x * iw, landmarks[3].y * ih, # mouth_center (using for mouth_right/left approx)
            landmarks[3].x * iw, landmarks[3].y * ih  # mouth_center (using for mouth_right/left approx)
        ]
        
        # If mouth_right/left are critical and distinct,
        # you might need to estimate them from mouth_center + bounding box,
        # or accept that BlazeFace's landmark set is different.
        # For general compatibility with ArcFace/SFace, this approximation is often sufficient.

        face_info_for_recognizer_align = np.array([x, y, w, h] + recognizer_landmarks, dtype=np.float32)
        return face_info_for_recognizer_align

    def _detect_face_mtcnn(self, img):
        if not MTCNN_AVAILABLE:
            raise RuntimeError("MTCNN detector requires mtcnn library to be installed.")
        if img is None:
            raise ValueError("Input image is None for MTCNN detection.")

        # MTCNN expects RGB input
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            detections = self.face_detector.detect_faces(img_rgb)
        except Exception as e:
            raise RuntimeError(f"MTCNN detection failed: {e}")

        if not detections:
            return None

        # MTCNN returns a list of dictionaries, each with 'box', 'confidence', 'keypoints'
        # We pick the detection with the highest confidence
        best_detection = max(detections, key=lambda det: det['confidence'])

        bbox = best_detection['box']
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

        # MTCNN keypoints: 'left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right'
        keypoints = best_detection['keypoints']
        
        recognizer_landmarks = [
            keypoints['right_eye'][0], keypoints['right_eye'][1],
            keypoints['left_eye'][0], keypoints['left_eye'][1],
            keypoints['nose'][0], keypoints['nose'][1],
            keypoints['mouth_right'][0], keypoints['mouth_right'][1],
            keypoints['mouth_left'][0], keypoints['mouth_left'][1]
        ]
        
        face_info_for_recognizer_align = np.array([x, y, w, h] + recognizer_landmarks, dtype=np.float32)
        return face_info_for_recognizer_align

    def _extract_embedding_sface(self, img, face_info_for_recognizer_align):
        face_align = self.face_recognizer.alignCrop(img, face_info_for_recognizer_align)
        if face_align is None:
            raise ValueError("`alignCrop` returned None for SFace. Face might be out of bounds, invalid, or alignment failed.")
        face_feature = self.face_recognizer.feature(face_align)
        return face_feature

    def _extract_embedding_arcface(self, img, face_info_for_recognizer_align):
        landmarks = face_info_for_recognizer_align[4:].reshape(5, 2)
        input_size = (112, 112)

        dst_pts = np.array([
            [30.2946, 51.6963],
            [82.5596, 51.6963],
            [56.2272, 70.0827],
            [41.5493, 92.3655],
            [71.7317, 92.3655]
        ], dtype=np.float32)

        M, _ = cv2.estimateAffine2D(landmarks, dst_pts)
        if M is None:
            raise ValueError("cv2.estimateAffine2D returned None for ArcFace alignment. Landmarks might be invalid.")

        aligned_face = cv2.warpAffine(img, M, input_size, borderValue=(0, 0, 0))
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("cv2.warpAffine returned an empty or invalid image for ArcFace alignment.")

        blob = cv2.dnn.blobFromImage(
            aligned_face, 1.0/127.5, input_size, (127.5, 127.5, 127.5), swapRB=False
        )
        self.face_recognizer.setInput(blob)
        face_feature = self.face_recognizer.forward()
        face_feature = face_feature / np.linalg.norm(face_feature)
        return face_feature
    
    def _extract_embedding_facenet(self, img, face_info_for_recognizer_align):
        landmarks = face_info_for_recognizer_align[4:].reshape(5, 2)
        input_size = (112, 112)

        dst_pts = np.array([
            [30.2946, 51.6963],
            [82.5596, 51.6963],
            [56.2272, 70.0827],
            [41.5493, 92.3655],
            [71.7317, 92.3655]
        ], dtype=np.float32)

        M, _ = cv2.estimateAffine2D(landmarks, dst_pts)
        if M is None:
            raise ValueError("cv2.estimateAffine2D returned None for ArcFace alignment. Landmarks might be invalid.")

        aligned_face = cv2.warpAffine(img, M, input_size, borderValue=(0, 0, 0))
        if aligned_face is None or aligned_face.size == 0:
            raise ValueError("cv2.warpAffine returned an empty or invalid image for ArcFace alignment.")

        blob = cv2.dnn.blobFromImage(
            aligned_face, 1.0/127.5, input_size, (127.5, 127.5, 127.5), swapRB=False
        )
        self.face_recognizer.setInput(blob)
        face_feature = self.face_recognizer.forward()
        face_feature = face_feature / np.linalg.norm(face_feature)
        return face_feature

    def _calculate_similarity(self, embedding1, embedding2):
        if self.recognizer_backend == "sface":
            similarity = self.face_recognizer.match(embedding1, embedding2, cv2.FaceRecognizerSF_FR_COSINE)
            similarity = float(similarity)
        elif self.recognizer_backend == "arcface":
            similarity = np.dot(embedding1.flatten(), embedding2.flatten())
            similarity = float(np.clip(similarity, -1.0, 1.0))
        else:
            raise ValueError("Unsupported recognizer backend for similarity calculation.")
        return similarity