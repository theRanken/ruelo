import os, cv2, base64
import mediapipe as mp
import numpy as np
from scipy.spatial.distance import cosine

class MediaPipeFaceMatcher:    
    def __init__(self):
        # Face detection and mesh
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)
        
        # Holistic for more detailed face analysis
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True
        )
          # Initialize gender estimation parameters
        self.gender_markers = {
            'jaw_width': [(127, 356), (227, 132)],  # Left and right jaw points
            'nose_to_chin': [(168, 8)],             # Nose tip to chin
            'cheekbone_width': [(116, 345), (212, 432)]  # Cheekbone points
        }
        self.gender_threshold = 0.5  # Threshold for gender determination
            
        self.gender_labels = ['Male', 'Female']
    def predict_gender(self, landmarks):
        """Predict gender using facial landmarks"""
        try:
            # Calculate facial ratios
            ratios = []
            
            # Calculate jaw width ratio
            for points in self.gender_markers['jaw_width']:
                width = abs(landmarks[points[0]].x - landmarks[points[1]].x)
                height = abs(landmarks[points[0]].y - landmarks[points[1]].y)
                ratios.append(width / height if height != 0 else 0)
            
            # Calculate nose to chin ratio
            for points in self.gender_markers['nose_to_chin']:
                ratio = abs(landmarks[points[0]].y - landmarks[points[1]].y)
                ratios.append(ratio)
            
            # Calculate cheekbone width ratio
            for points in self.gender_markers['cheekbone_width']:
                width = abs(landmarks[points[0]].x - landmarks[points[1]].x)
                height = abs(landmarks[points[0]].y - landmarks[points[1]].y)
                ratios.append(width / height if height != 0 else 0)
            
            # Average the ratios
            avg_ratio = sum(ratios) / len(ratios) if ratios else 0
            
            # Make prediction based on average ratio
            # Generally, male faces have larger ratios
            return "Male" if avg_ratio > self.gender_threshold else "Female"
            
        except Exception as e:
            print(f"Error predicting gender: {str(e)}")
            return "Unknown"

    def load_image_from_source(self, image_source):
        """Load image from either file path or base64 string"""
        try:
            # Check if it's a base64 string
            if image_source.startswith('data:image/'):
                # Handle data URL format: data:image/jpeg;base64,/9j/4AAQ...
                header, encoded = image_source.split(',', 1)
                image_data = base64.b64decode(encoded)
            elif self.is_base64(image_source):
                # Handle raw base64 string
                image_data = base64.b64decode(image_source)
            else:
                # Assume it's a file path
                return cv2.imread(image_source), None
            
            # Convert bytes to numpy array
            nparr = np.frombuffer(image_data, np.uint8)
            
            # Decode image
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Could not decode base64 image"
                
            return image, None
            
        except Exception as e:
            return None, f"Error loading image: {str(e)}"
    
    def is_base64(self, s):
        """Check if string is valid base64"""
        try:
            if isinstance(s, str):
                # Check if it looks like base64
                if len(s) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s):
                    base64.b64decode(s, validate=True)
                    return True
            return False
        except Exception:
            return False
    
    def extract_face_landmarks(self, image_source):
        """Extract face landmarks using MediaPipe from file path or base64"""
        try:
            image, error = self.load_image_from_source(image_source)
            if error:
                return None, error
            
            if image is None:
                return None, "Could not load image"
            
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return None, "No face detected"
            
            # Get the first face's landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks), None
            
        except Exception as e:
            return None, str(e)
    
    def compare_faces(self, image1_source, image2_source, threshold=0.4):
        """Compare faces using landmark similarity - accepts file paths or base64 strings"""
        landmarks1, error1 = self.extract_face_landmarks(image1_source)
        if error1:
            return {"error": f"Image 1: {error1}"}
            
        landmarks2, error2 = self.extract_face_landmarks(image2_source)
        if error2:
            return {"error": f"Image 2: {error2}"}
        
        # Calculate cosine similarity
        similarity = 1 - cosine(landmarks1, landmarks2)
        
        # Calculate Euclidean distance (normalized)
        euclidean_dist = np.linalg.norm(landmarks1 - landmarks2)
        max_possible_dist = np.sqrt(len(landmarks1))  # Rough normalization
        normalized_dist = euclidean_dist / max_possible_dist
        
        match = similarity > threshold
        confidence = max(0, min(1, similarity))  # Clamp to 0-1
        return {
            "match": bool(match),
            "confidence": float(confidence),
            "similarity": float(similarity),
            "distance": float(normalized_dist)
        }
        
    def analyze_face(self, image_source):
        """Analyze face for age, gender and emotions"""
        try:
            # Get face landmarks
            image, error = self.load_image_from_source(image_source)
            if error:
                return {"error": error}
            
            if image is None:
                return {"error": "Could not load image"}
                
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {"error": "No face detected"}
            
            face_landmarks = results.multi_face_landmarks[0].landmark
            
            # Calculate facial metrics
            face_height = face_landmarks[152].y - face_landmarks[10].y
            mouth_width = face_landmarks[291].x - face_landmarks[61].x
            mouth_height = face_landmarks[14].y - face_landmarks[13].y
            brow_height = face_landmarks[107].y - face_landmarks[55].y
            eye_open = face_landmarks[159].y - face_landmarks[145].y
            
            # Emotion analysis
            emotions = {
                "happy": 0.2,
                "sad": 0.1,
                "angry": 0.1,
                "surprised": 0.1,
                "neutral": 0.5
            }
            
            # Simple emotion detection
            if mouth_width > 0.45 and mouth_height < 0.1:
                emotions["happy"] = 0.8
                emotions["neutral"] = 0.2
            elif brow_height > 0.15:
                emotions["surprised"] = 0.7
                emotions["neutral"] = 0.3
            elif brow_height < 0.1 and eye_open < 0.1:
                emotions["angry"] = 0.6
                emotions["neutral"] = 0.4
            elif mouth_height < 0.05 and mouth_width < 0.4:
                emotions["sad"] = 0.7
                emotions["neutral"] = 0.3
                
            # Simple age estimation
            base_age = 25
            if face_height > 0.4:
                base_age -= 5
            if brow_height > 0.15:
                base_age -= 5
            
            age = base_age + np.random.randint(-3, 4)
            age = max(18, min(65, age))
              # Use facial landmarks for gender detection
            gender = self.predict_gender(face_landmarks)
            gender_confidence = 0.7  # Fixed confidence since we're using geometric measurements
            
            return {
                "age": int(age),
                "gender": gender,
                "gender_confidence": float(gender_confidence),
                "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0],
                "emotion": {k: float(v) for k, v in emotions.items()}
            }
            
        except Exception as e:
            return {"error": str(e)}
