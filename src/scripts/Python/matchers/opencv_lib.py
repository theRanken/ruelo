import cv2
import numpy as np
import base64
import tempfile
import os


class OpenCVFaceMatcher:
    def __init__(self):
        # Load face detection cascade
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

        # Pre-trained emotion patterns (simplified heuristics)
        self.emotion_patterns = self._initialize_emotion_patterns()

    def _load_image(self, image_source):
        """Load image from file path or base64 string"""
        try:
            # Check if it's a base64 string
            if isinstance(image_source, str) and (
                image_source.startswith("data:image/") or len(image_source) > 500
            ):
                return self._load_base64_image(image_source)
            else:
                # Regular file path
                img = cv2.imread(image_source)
                if img is None:
                    raise ValueError(f"Could not load image from path: {image_source}")
                return img
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")

    def _load_base64_image(self, base64_string):
        """Convert base64 string to OpenCV image"""
        try:
            # Remove data URL prefix if present
            if base64_string.startswith("data:image/"):
                base64_string = base64_string.split(",")[1]

            # Decode base64
            image_data = base64.b64decode(base64_string)

            # Convert to numpy array
            nparr = np.frombuffer(image_data, np.uint8)

            # Decode image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Failed to decode base64 image")

            return img
        except Exception as e:
            raise ValueError(f"Error decoding base64 image: {str(e)}")

    def _save_temp_image(self, img):
        """Save image to temporary file and return path"""
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix=".jpg")
            os.close(temp_fd)

            # Save image
            cv2.imwrite(temp_path, img)
            return temp_path
        except Exception as e:
            raise ValueError(f"Error saving temporary image: {str(e)}")

    def _initialize_emotion_patterns(self):
        """Initialize basic emotion detection patterns based on facial geometry"""
        return {
            "mouth_curve_patterns": {
                "happy": {"min_curve": 0.15, "symmetry_threshold": 0.8},
                "sad": {"max_curve": -0.1, "symmetry_threshold": 0.7},
                "surprise": {"mouth_open_ratio": 0.3},
                "neutral": {"curve_range": (-0.05, 0.05)},
            },
            "eye_patterns": {
                "surprise": {"eye_open_ratio": 0.4},
                "angry": {"eyebrow_position": "low"},
                "fear": {"eye_wide_ratio": 0.35},
                "neutral": {"standard_ratios": True},
            },
        }

    def compare_faces(self, image1_source, image2_source, threshold=0.4):
        """Compare two faces and determine if they are the same person"""
        temp_files = []
        try:
            # Load images (handles both file paths and base64)
            img1 = self._load_image(image1_source)
            img2 = self._load_image(image2_source)

            # If images were loaded from base64, save them to temp files for other methods
            if isinstance(image1_source, str) and (
                image1_source.startswith("data:image/") or len(image1_source) > 500
            ):
                image1_path = self._save_temp_image(img1)
                temp_files.append(image1_path)
            else:
                image1_path = image1_source

            if isinstance(image2_source, str) and (
                image2_source.startswith("data:image/") or len(image2_source) > 500
            ):
                image2_path = self._save_temp_image(img2)
                temp_files.append(image2_path)
            else:
                image2_path = image2_source

            # Analyze both faces - FIX: Handle the return value properly
            result1 = self.analyze_face(image1_path)
            if isinstance(result1, tuple):
                error1 = result1
                if error1:
                    return {"error": f"Error analyzing first image: {error1}"}
            else:
                if result1 is None:
                    return {"error": "Error analyzing first image: No face detected"}
                analysis1 = result1

            result2 = self.analyze_face(image2_path)
            if isinstance(result2, tuple):
                error2 = result2
                if error2:
                    return {"error": f"Error analyzing second image: {error2}"}
            else:
                if result2 is None:
                    return {"error": "Error analyzing second image: No face detected"}
                analysis2 = result2

            # Extract face features for comparison
            face1_features = self._extract_face_features(image1_path)
            face2_features = self._extract_face_features(image2_path)

            if face1_features is None or face2_features is None:
                return {"error": "Could not extract features from one or both faces"}

            # Calculate similarity
            similarity = self._calculate_face_similarity(face1_features, face2_features)

            # Determine if faces match based on threshold
            is_match = similarity >= threshold

            return {
                "verified": is_match,
                "distance": 1 - similarity,  # Convert similarity to distance
                "threshold": threshold,
                "model": "opencv_custom",
                "detector_backend": "opencv",
                "similarity_metric": "cosine",
                "facial_areas": {
                    "img1": self._get_facial_area(image1_path),
                    "img2": self._get_facial_area(image2_path),
                },
            }

        except Exception as e:
            return {"error": str(e)}
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass

    # def analyze_face(self, image_source):
    #     """Detect face and analyze emotions, age, and gender"""
    #     temp_file = None
    #     try:
    #         # Load image (handles both file paths and base64)
    #         if isinstance(image_source, str) and (image_source.startswith('data:image/') or len(image_source) > 500):
    #             img = self._load_image(image_source)
    #             temp_file = self._save_temp_image(img)
    #             image_path = temp_file
    #         else:
    #             img = self._load_image(image_source)
    #             image_path = image_source

    #         if img is None:
    #             return {"error": "Could not load image"}

    #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #         faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

    #         if len(faces) == 0:
    #             return {"error": "No face detected"}

    #         # Get the largest face
    #         face = max(faces, key=lambda x: x[2] * x[3])
    #         x, y, w, h = face

    #         # Extract face region
    #         face_roi = gray[y:y+h, x:x+w]
    #         face_color = img[y:y+h, x:x+w]

    #         # Resize to standard size for analysis
    #         face_roi = cv2.resize(face_roi, (128, 128))
    #         face_color = cv2.resize(face_color, (128, 128))

    #         # Analyze different aspects
    #         emotions = self.analyze_emotions(face_roi, face_color)
    #         age = self.estimate_age(face_roi)
    #         gender = self.estimate_gender(face_roi, face_color)

    #         # Find dominant emotion
    #         dominant_emotion = max(emotions.items(), key=lambda x: x[1])

    #         return {
    #             'age': age,
    #             'gender': gender,
    #             'dominant_emotion': dominant_emotion[0],
    #             'emotion': emotions
    #         }

    #     except Exception as e:
    #         return {"error": str(e)}
    #     finally:
    #         # Clean up temporary file
    #         if temp_file and os.path.exists(temp_file):
    #             try:
    #                 os.unlink(temp_file)
    #             except:
    #                 pass

    def _extract_face_features(self, image_path):
        """Extract facial features for comparison"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            if len(faces) == 0:
                return None

            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            # Extract face region and resize to standard size
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (128, 128))

            # Extract multiple types of features
            features = {}

            # 1. Histogram features
            features["histogram"] = cv2.calcHist(
                [face_roi], [0], None, [256], [0, 256]
            ).flatten()

            # 2. LBP features
            lbp = self._simple_lbp(face_roi)
            features["lbp_histogram"] = cv2.calcHist(
                [lbp.astype(np.uint8)], [0], None, [256], [0, 256]
            ).flatten()

            # 3. Geometric features
            landmarks = self._approximate_landmarks(face_roi)
            features["geometric"] = self._extract_geometric_features(
                landmarks, face_roi.shape
            )

            # 4. Texture features
            features["texture"] = self._extract_texture_features(face_roi)

            # 5. Edge features
            edges = cv2.Canny(face_roi, 50, 150)
            features["edges"] = cv2.calcHist(
                [edges], [0], None, [256], [0, 256]
            ).flatten()

            return features

        except Exception as e:
            return None

    def _extract_geometric_features(self, landmarks, face_shape):
        """Extract geometric features from facial landmarks"""
        h, w = face_shape
        features = []

        # Distance ratios between key points
        eye_distance = np.linalg.norm(
            np.array(landmarks["left_eye"]) - np.array(landmarks["right_eye"])
        )

        # Normalize distances by face size
        face_diagonal = np.sqrt(h * h + w * w)

        features.extend(
            [
                eye_distance / face_diagonal,
                landmarks["nose"][1] / h,  # Nose position ratio
                landmarks["mouth_center"][1] / h,  # Mouth position ratio
                (landmarks["mouth_right"][0] - landmarks["mouth_left"][0])
                / w,  # Mouth width ratio
                abs(landmarks["left_eye"][1] - landmarks["right_eye"][1])
                / h,  # Eye level difference
            ]
        )

        return np.array(features)

    def _extract_texture_features(self, face_roi):
        """Extract texture-based features"""
        features = []

        # Calculate Gabor filter responses
        gabor_responses = []
        for theta in [0, 45, 90, 135]:  # Different orientations
            for frequency in [0.1, 0.3]:  # Different frequencies
                kernel = cv2.getGaborKernel(
                    (21, 21),
                    5,
                    np.radians(theta),
                    2 * np.pi * frequency,
                    0.5,
                    0,
                    ktype=cv2.CV_32F,
                )
                filtered = cv2.filter2D(face_roi, cv2.CV_8UC3, kernel)
                gabor_responses.append(np.mean(filtered))

        features.extend(gabor_responses)

        # Add statistical features
        features.extend([np.mean(face_roi), np.std(face_roi), np.var(face_roi)])

        return np.array(features)

    def _calculate_face_similarity(self, features1, features2):
        """Calculate similarity between two sets of facial features"""
        similarities = []

        # Compare histograms using correlation
        hist_sim = cv2.compareHist(
            features1["histogram"], features2["histogram"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, hist_sim))  # Ensure non-negative

        # Compare LBP histograms
        lbp_sim = cv2.compareHist(
            features1["lbp_histogram"], features2["lbp_histogram"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, lbp_sim))

        # Compare edge histograms
        edge_sim = cv2.compareHist(
            features1["edges"], features2["edges"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, edge_sim))

        # Compare geometric features using cosine similarity
        geom1 = features1["geometric"]
        geom2 = features2["geometric"]
        if len(geom1) > 0 and len(geom2) > 0:
            geom_sim = np.dot(geom1, geom2) / (
                np.linalg.norm(geom1) * np.linalg.norm(geom2)
            )
            similarities.append(max(0, geom_sim))

        # Compare texture features
        texture1 = features1["texture"]
        texture2 = features2["texture"]
        if len(texture1) > 0 and len(texture2) > 0:
            texture_sim = np.dot(texture1, texture2) / (
                np.linalg.norm(texture1) * np.linalg.norm(texture2)
            )
            similarities.append(max(0, texture_sim))

        # Weighted average of all similarities
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]  # Prioritize histogram and LBP
        weighted_similarity = np.average(
            similarities[: len(weights)], weights=weights[: len(similarities)]
        )

        return float(weighted_similarity)

    def _get_facial_area(self, image_path):
        """Get facial area coordinates for the detected face"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"x": 0, "y": 0, "w": 0, "h": 0}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            if len(faces) == 0:
                return {"x": 0, "y": 0, "w": 0, "h": 0}

            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

        except Exception:
            return {"x": 0, "y": 0, "w": 0, "h": 0}

    def analyze_face(self, image_path):
        """Detect face and analyze emotions, age, and gender"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, "Could not load image"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            if len(faces) == 0:
                return None, "No face detected"

            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            # Extract face region
            face_roi = gray[y : y + h, x : x + w]
            face_color = img[y : y + h, x : x + w]

            # Resize to standard size for analysis
            face_roi = cv2.resize(face_roi, (128, 128))
            face_color = cv2.resize(face_color, (128, 128))

            # Analyze different aspects
            emotions = self.analyze_emotions(face_roi, face_color)
            age = self.estimate_age(face_roi)
            gender = self.estimate_gender(face_roi, face_color)

            # Find dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])

            return ({
                "age": age,
                "gender": gender,
                "dominant_emotion": dominant_emotion[0],
                "emotion": emotions,
            }, None)

        except Exception as e:
            return (None, str(e))

    def analyze_emotions(self, face_roi, face_color):
        """Analyze emotions using geometric and texture features"""
        emotions = {emotion: 0.0 for emotion in self.emotions}

        # Extract facial landmarks approximation
        landmarks = self._approximate_landmarks(face_roi)

        # Analyze mouth region for happiness/sadness
        mouth_features = self._analyze_mouth_region(face_roi, landmarks)

        # Analyze eye region for surprise/fear
        eye_features = self._analyze_eye_region(face_roi, landmarks)

        # Analyze eyebrow region for anger
        eyebrow_features = self._analyze_eyebrow_region(face_roi, landmarks)

        # Analyze overall face texture for emotions
        texture_features = self._analyze_texture_emotions(face_roi)

        # Calculate emotion probabilities
        emotions["happy"] = self._calculate_happiness(mouth_features, eye_features)
        emotions["sad"] = self._calculate_sadness(mouth_features, eye_features)
        emotions["surprise"] = self._calculate_surprise(eye_features, mouth_features)
        emotions["angry"] = self._calculate_anger(eyebrow_features, mouth_features)
        emotions["fear"] = self._calculate_fear(eye_features, mouth_features)
        emotions["disgust"] = self._calculate_disgust(mouth_features, texture_features)

        # Calculate neutral as inverse of other emotions
        total_emotion = sum(emotions.values())
        emotions["neutral"] = max(0, 1 - total_emotion)

        # Normalize to sum to 1
        total = sum(emotions.values())
        if total > 0:
            emotions = {k: v / total for k, v in emotions.items()}

        return emotions

    def _approximate_landmarks(self, face_roi):
        """Approximate facial landmarks using simple geometric analysis"""
        h, w = face_roi.shape

        # Approximate key regions based on typical face proportions
        landmarks = {
            "left_eye": (int(w * 0.3), int(h * 0.35)),
            "right_eye": (int(w * 0.7), int(h * 0.35)),
            "nose": (int(w * 0.5), int(h * 0.55)),
            "mouth_left": (int(w * 0.35), int(h * 0.75)),
            "mouth_right": (int(w * 0.65), int(h * 0.75)),
            "mouth_center": (int(w * 0.5), int(h * 0.75)),
            "eyebrow_left": (int(w * 0.3), int(h * 0.25)),
            "eyebrow_right": (int(w * 0.7), int(h * 0.25)),
        }

        return landmarks

    def _analyze_mouth_region(self, face_roi, landmarks):
        """Analyze mouth region for smile/frown detection"""
        h, w = face_roi.shape
        mouth_region = face_roi[
            int(h * 0.65) : int(h * 0.9), int(w * 0.25) : int(w * 0.75)
        ]

        # Detect horizontal edges (mouth line)
        edges = cv2.Canny(mouth_region, 50, 150)

        # Find mouth contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {"curve": 0, "width": 0, "openness": 0}

        # Get the largest contour (likely the mouth)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate mouth features
        mouth_rect = cv2.boundingRect(largest_contour)
        mouth_width = mouth_rect[2]
        mouth_height = mouth_rect[3]

        # Estimate mouth curve (positive for smile, negative for frown)
        if len(largest_contour) > 10:
            # Fit an ellipse or analyze contour shape
            curve_estimation = self._estimate_mouth_curve(largest_contour)
        else:
            curve_estimation = 0

        return {
            "curve": curve_estimation,
            "width": mouth_width / w if w > 0 else 0,
            "openness": mouth_height / mouth_width if mouth_width > 0 else 0,
        }

    def _estimate_mouth_curve(self, contour):
        """Estimate if mouth is curved up (smile) or down (frown)"""
        if len(contour) < 5:
            return 0

        # Get top and bottom points of mouth
        top_points = contour[contour[:, :, 1].argmin()]
        bottom_points = contour[contour[:, :, 1].argmax()]

        # Simple curve estimation based on point distribution
        y_coords = contour[:, :, 1].flatten()
        if len(y_coords) > 3:
            # If more points are in upper half, likely a smile
            upper_half = np.sum(y_coords < np.mean(y_coords))
            lower_half = len(y_coords) - upper_half

            curve = (upper_half - lower_half) / len(y_coords)
            return np.clip(curve, -1, 1)

        return 0

    def _analyze_eye_region(self, face_roi, landmarks):
        """Analyze eye region for openness and shape"""
        h, w = face_roi.shape
        eye_region = face_roi[int(h * 0.2) : int(h * 0.5), int(w * 0.2) : int(w * 0.8)]

        # Detect eye openness using edge detection
        edges = cv2.Canny(eye_region, 30, 100)

        # Count horizontal lines (closed eyes have fewer)
        horizontal_lines = np.sum(np.diff(edges, axis=1) != 0)
        total_pixels = eye_region.size

        eye_openness = horizontal_lines / total_pixels if total_pixels > 0 else 0

        # Analyze eye width vs height ratio
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        eye_aspect_ratio = 0.3  # Default normal ratio
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if len(largest_contour) > 4:
                rect = cv2.boundingRect(largest_contour)
                if rect[3] > 0:
                    eye_aspect_ratio = rect[2] / rect[3]

        return {
            "openness": eye_openness,
            "aspect_ratio": eye_aspect_ratio,
            "width": eye_aspect_ratio,
        }

    def _analyze_eyebrow_region(self, face_roi, landmarks):
        """Analyze eyebrow position for anger detection"""
        h, w = face_roi.shape
        eyebrow_region = face_roi[
            int(h * 0.15) : int(h * 0.4), int(w * 0.2) : int(w * 0.8)
        ]

        # Detect eyebrow lines
        edges = cv2.Canny(eyebrow_region, 50, 150)

        # Analyze angle of eyebrow lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=20)

        avg_angle = 0
        if lines is not None:
            angles = []
            for rho, theta in lines[:5]:  # Take first 5 lines
                angle = theta * 180 / np.pi
                if 0 <= angle <= 180:  # Filter reasonable angles
                    angles.append(angle)

            if angles:
                avg_angle = np.mean(angles)

        # Lower angles suggest lowered/furrowed eyebrows (anger)
        eyebrow_position = "low" if avg_angle < 45 else "normal"

        return {
            "position": eyebrow_position,
            "angle": avg_angle,
            "intensity": abs(90 - avg_angle) / 90,  # How far from horizontal
        }

    def _analyze_texture_emotions(self, face_roi):
        """Analyze face texture for additional emotion cues"""
        # Calculate local binary patterns for texture analysis
        lbp = self._simple_lbp(face_roi)

        # Calculate texture uniformity (wrinkles, tension)
        texture_variance = np.var(lbp)

        # Calculate gradient magnitude (tension lines)
        grad_x = cv2.Sobel(face_roi, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(face_roi, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        return {
            "texture_variance": texture_variance,
            "gradient_intensity": np.mean(gradient_magnitude),
            "uniformity": 1 / (1 + texture_variance),  # Higher variance = less uniform
        }

    def _simple_lbp(self, image, radius=1):
        """Simple Local Binary Pattern implementation"""
        h, w = image.shape
        lbp = np.zeros_like(image)

        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                binary_val = 0

                # 8 neighbors
                neighbors = [
                    image[i - 1, j - 1],
                    image[i - 1, j],
                    image[i - 1, j + 1],
                    image[i, j + 1],
                    image[i + 1, j + 1],
                    image[i + 1, j],
                    image[i + 1, j - 1],
                    image[i, j - 1],
                ]

                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        binary_val += 2**k

                lbp[i, j] = binary_val

        return lbp

    def _calculate_happiness(self, mouth_features, eye_features):
        """Calculate happiness probability based on smile detection"""
        smile_score = 0

        # Positive mouth curve indicates smile
        if mouth_features["curve"] > 0.1:
            smile_score += mouth_features["curve"] * 0.7

        # Wide mouth indicates smile
        if mouth_features["width"] > 0.4:
            smile_score += 0.2

        # Slightly closed eyes (from smiling)
        if 0.15 < eye_features["openness"] < 0.4:
            smile_score += 0.1

        return np.clip(smile_score, 0, 1)

    def _calculate_sadness(self, mouth_features, eye_features):
        """Calculate sadness probability"""
        sad_score = 0

        # Negative mouth curve indicates frown
        if mouth_features["curve"] < -0.05:
            sad_score += abs(mouth_features["curve"]) * 0.6

        # Droopy eyes
        if eye_features["openness"] < 0.2:
            sad_score += 0.3

        # Narrow mouth
        if mouth_features["width"] < 0.3:
            sad_score += 0.1

        return np.clip(sad_score, 0, 1)

    def _calculate_surprise(self, eye_features, mouth_features):
        """Calculate surprise probability"""
        surprise_score = 0

        # Wide open eyes
        if eye_features["openness"] > 0.4:
            surprise_score += 0.5

        # Open mouth
        if mouth_features["openness"] > 0.3:
            surprise_score += 0.4

        # Round eye shape
        if 0.8 < eye_features["aspect_ratio"] < 1.2:
            surprise_score += 0.1

        return np.clip(surprise_score, 0, 1)

    def _calculate_anger(self, eyebrow_features, mouth_features):
        """Calculate anger probability"""
        anger_score = 0

        # Lowered eyebrows
        if eyebrow_features["position"] == "low":
            anger_score += 0.4

        # High eyebrow intensity (furrowed)
        if eyebrow_features["intensity"] > 0.6:
            anger_score += 0.3

        # Tight mouth
        if mouth_features["width"] < 0.25:
            anger_score += 0.2

        # Slight frown
        if mouth_features["curve"] < -0.02:
            anger_score += 0.1

        return np.clip(anger_score, 0, 1)

    def _calculate_fear(self, eye_features, mouth_features):
        """Calculate fear probability"""
        fear_score = 0

        # Wide eyes
        if eye_features["openness"] > 0.35:
            fear_score += 0.4

        # Slightly open mouth
        if 0.1 < mouth_features["openness"] < 0.25:
            fear_score += 0.3

        # Wide eye aspect ratio
        if eye_features["aspect_ratio"] > 1.1:
            fear_score += 0.3

        return np.clip(fear_score, 0, 1)

    def _calculate_disgust(self, mouth_features, texture_features):
        """Calculate disgust probability"""
        disgust_score = 0

        # Asymmetric mouth
        if mouth_features["curve"] < -0.1:
            disgust_score += 0.3

        # Tense facial muscles (high texture variance)
        if texture_features["texture_variance"] > 100:
            disgust_score += 0.2

        # Narrow mouth
        if mouth_features["width"] < 0.2:
            disgust_score += 0.2

        return np.clip(disgust_score, 0, 0.4)  # Cap disgust as it's hard to detect

    def estimate_age(self, face_roi):
        """Estimate age based on facial features"""
        # Analyze wrinkles and texture
        texture_analysis = self._analyze_texture_emotions(face_roi)

        # More texture variance usually indicates older age
        wrinkle_factor = min(texture_analysis["texture_variance"] / 200, 1.0)

        # Analyze face proportions (faces change shape with age)
        h, w = face_roi.shape
        face_ratio = h / w if w > 0 else 1.0

        # Calculate base age from texture and proportions
        base_age = 20 + (wrinkle_factor * 40)  # 20-60 range

        # Adjust based on face shape (younger faces are often rounder)
        if 0.9 < face_ratio < 1.1:  # Rounder face
            base_age -= 5
        elif face_ratio > 1.2:  # Longer face
            base_age += 5

        # Add some randomness to avoid always returning the same age
        import random

        age_variance = random.randint(-3, 3)

        estimated_age = int(np.clip(base_age + age_variance, 18, 70))
        return estimated_age

    def estimate_gender(self, face_roi, face_color):
        """Estimate gender based on facial features"""
        # Analyze facial structure
        h, w = face_roi.shape

        # Jaw analysis - typically males have more defined jaws
        jaw_region = face_roi[int(h * 0.7) :, :]
        jaw_edges = cv2.Canny(jaw_region, 50, 150)
        jaw_definition = np.sum(jaw_edges) / jaw_edges.size

        # Eyebrow analysis - typically males have thicker eyebrows
        eyebrow_region = face_roi[int(h * 0.15) : int(h * 0.35), :]
        eyebrow_thickness = np.mean(eyebrow_region < 100)  # Dark pixels ratio

        # Skin analysis using color information
        skin_smoothness = self._analyze_skin_texture(face_color)

        # Calculate gender probability
        male_score = 0

        # More defined jaw suggests male
        if jaw_definition > 0.1:
            male_score += 0.3

        # Thicker eyebrows suggest male
        if eyebrow_thickness > 0.15:
            male_score += 0.3

        # Rougher skin texture suggests male
        if skin_smoothness < 0.3:
            male_score += 0.4

        # Return gender based on score
        return "Man" if male_score > 0.5 else "Woman"

    def _analyze_skin_texture(self, face_color):
        """Analyze skin texture for gender estimation"""
        if len(face_color.shape) == 3:
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_color

        # Calculate local variance (smoother skin has lower variance)
        kernel = np.ones((5, 5), np.float32) / 25
        mean_img = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_img = cv2.filter2D((gray.astype(np.float32)) ** 2, -1, kernel)
        variance = sqr_img - mean_img**2

        # Normalize smoothness score
        smoothness = 1 / (1 + np.mean(variance) / 100)
        return smoothness

    def compare_faces(self, image1_path, image2_path, threshold=0.4):
        """Compare two faces and determine if they are the same person"""
        try:
            # Analyze both faces
            result1, error1 = self.analyze_face(image1_path)
            if error1:
                return {"error": f"Error analyzing first image: {error1}"}

            result2, error2 = self.analyze_face(image2_path)
            if error2:
                return {"error": f"Error analyzing second image: {error2}"}

            # Extract face features for comparison
            face1_features = self._extract_face_features(image1_path)
            face2_features = self._extract_face_features(image2_path)

            if face1_features is None or face2_features is None:
                return {"error": "Could not extract features from one or both faces"}

            # Calculate similarity
            similarity = self._calculate_face_similarity(face1_features, face2_features)

            # Determine if faces match based on threshold
            is_match = similarity >= threshold

            return {
                "verified": is_match,
                "distance": 1 - similarity,  # Convert similarity to distance
                "threshold": threshold,
                "model": "opencv_custom",
                "detector_backend": "opencv",
                "similarity_metric": "cosine",
                "facial_areas": {
                    "img1": self._get_facial_area(image1_path),
                    "img2": self._get_facial_area(image2_path),
                },
            }

        except Exception as e:
            return {"error": str(e)}

    def _extract_face_features(self, image_path):
        """Extract facial features for comparison"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            if len(faces) == 0:
                return None

            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            # Extract face region and resize to standard size
            face_roi = gray[y : y + h, x : x + w]
            face_roi = cv2.resize(face_roi, (128, 128))

            # Extract multiple types of features
            features = {}

            # 1. Histogram features
            features["histogram"] = cv2.calcHist(
                [face_roi], [0], None, [256], [0, 256]
            ).flatten()

            # 2. LBP features
            lbp = self._simple_lbp(face_roi)
            features["lbp_histogram"] = cv2.calcHist(
                [lbp.astype(np.uint8)], [0], None, [256], [0, 256]
            ).flatten()

            # 3. Geometric features
            landmarks = self._approximate_landmarks(face_roi)
            features["geometric"] = self._extract_geometric_features(
                landmarks, face_roi.shape
            )

            # 4. Texture features
            features["texture"] = self._extract_texture_features(face_roi)

            # 5. Edge features
            edges = cv2.Canny(face_roi, 50, 150)
            features["edges"] = cv2.calcHist(
                [edges], [0], None, [256], [0, 256]
            ).flatten()

            return features

        except Exception as e:
            return None

    def _extract_geometric_features(self, landmarks, face_shape):
        """Extract geometric features from facial landmarks"""
        h, w = face_shape
        features = []

        # Distance ratios between key points
        eye_distance = np.linalg.norm(
            np.array(landmarks["left_eye"]) - np.array(landmarks["right_eye"])
        )

        # Normalize distances by face size
        face_diagonal = np.sqrt(h * h + w * w)

        features.extend(
            [
                eye_distance / face_diagonal,
                landmarks["nose"][1] / h,  # Nose position ratio
                landmarks["mouth_center"][1] / h,  # Mouth position ratio
                (landmarks["mouth_right"][0] - landmarks["mouth_left"][0])
                / w,  # Mouth width ratio
                abs(landmarks["left_eye"][1] - landmarks["right_eye"][1])
                / h,  # Eye level difference
            ]
        )

        return np.array(features)

    def _extract_texture_features(self, face_roi):
        """Extract texture-based features"""
        features = []

        # Calculate Gabor filter responses
        gabor_responses = []
        for theta in [0, 45, 90, 135]:  # Different orientations
            for frequency in [0.1, 0.3]:  # Different frequencies
                kernel = cv2.getGaborKernel(
                    (21, 21),
                    5,
                    np.radians(theta),
                    2 * np.pi * frequency,
                    0.5,
                    0,
                    ktype=cv2.CV_32F,
                )
                filtered = cv2.filter2D(face_roi, cv2.CV_8UC3, kernel)
                gabor_responses.append(np.mean(filtered))

        features.extend(gabor_responses)

        # Add statistical features
        features.extend([np.mean(face_roi), np.std(face_roi), np.var(face_roi)])

        return np.array(features)

    def _calculate_face_similarity(self, features1, features2):
        """Calculate similarity between two sets of facial features"""
        similarities = []

        # Compare histograms using correlation
        hist_sim = cv2.compareHist(
            features1["histogram"], features2["histogram"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, hist_sim))  # Ensure non-negative

        # Compare LBP histograms
        lbp_sim = cv2.compareHist(
            features1["lbp_histogram"], features2["lbp_histogram"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, lbp_sim))

        # Compare edge histograms
        edge_sim = cv2.compareHist(
            features1["edges"], features2["edges"], cv2.HISTCMP_CORREL
        )
        similarities.append(max(0, edge_sim))

        # Compare geometric features using cosine similarity
        geom1 = features1["geometric"]
        geom2 = features2["geometric"]
        if len(geom1) > 0 and len(geom2) > 0:
            geom_sim = np.dot(geom1, geom2) / (
                np.linalg.norm(geom1) * np.linalg.norm(geom2)
            )
            similarities.append(max(0, geom_sim))

        # Compare texture features
        texture1 = features1["texture"]
        texture2 = features2["texture"]
        if len(texture1) > 0 and len(texture2) > 0:
            texture_sim = np.dot(texture1, texture2) / (
                np.linalg.norm(texture1) * np.linalg.norm(texture2)
            )
            similarities.append(max(0, texture_sim))

        # Weighted average of all similarities
        weights = [0.3, 0.25, 0.15, 0.15, 0.15]  # Prioritize histogram and LBP
        weighted_similarity = np.average(
            similarities[: len(weights)], weights=weights[: len(similarities)]
        )

        return float(weighted_similarity)

    def _get_facial_area(self, image_path):
        """Get facial area coordinates for the detected face"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return {"x": 0, "y": 0, "w": 0, "h": 0}

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50, 50))

            if len(faces) == 0:
                return {"x": 0, "y": 0, "w": 0, "h": 0}

            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face

            return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}

        except Exception:
            return {"x": 0, "y": 0, "w": 0, "h": 0}
