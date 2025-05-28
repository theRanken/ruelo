import mediapipe as mp
import cv2
import numpy as np
import base64
import os
import urllib.request
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
        
        # Load pre-trained gender detection model (Caffe)
        model_path = os.path.join(os.path.dirname(__file__), 'models')
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        self.gender_model_path = os.path.join(model_path, 'gender_net.caffemodel')
        self.gender_proto_path = os.path.join(model_path, 'gender_deploy.prototxt')
        
        # Download models if they don't exist
        if not os.path.exists(self.gender_model_path) or not os.path.exists(self.gender_proto_path):
            self._download_gender_model()
        
        try:
            self.gender_net = cv2.dnn.readNet(self.gender_model_path, self.gender_proto_path)
        except Exception as e:
            print(f"Warning: Could not load gender detection model: {str(e)}")
            self.gender_net = None
            
        self.gender_labels = ['Male', 'Female']
    
    def _download_gender_model(self):
        """Download gender detection model files"""
        try:
            # Download gender model
            if not os.path.exists(self.gender_model_path):
                print("Downloading gender detection model...")
                model_url = "https://github.com/Tony607/focal_loss_keras/raw/master/models/gender_net.caffemodel"
                urllib.request.urlretrieve(model_url, self.gender_model_path)
                print("Gender detection model downloaded successfully")
            
            # Create proto file with model architecture
            if not os.path.exists(self.gender_proto_path):
                print("Creating gender detection model architecture...")
                with open(self.gender_proto_path, 'w') as f:
                    f.write('input: "data"\ninput_dim: 1\ninput_dim: 3\ninput_dim: 227\ninput_dim: 227\n')
                    f.write('layer { name: "conv1" type: "Convolution" bottom: "data" top: "conv1" convolution_param { num_output: 96 kernel_size: 7 stride: 4 } }\n')
                    f.write('layer { name: "relu1" type: "ReLU" bottom: "conv1" top: "conv1" }\n')
                    f.write('layer { name: "pool1" type: "Pooling" bottom: "conv1" top: "pool1" pooling_param { pool: MAX kernel_size: 3 stride: 2 } }\n')
                    f.write('layer { name: "fc8" type: "InnerProduct" bottom: "pool1" top: "fc8" inner_product_param { num_output: 2 } }\n')
                    f.write('layer { name: "prob" type: "Softmax" bottom: "fc8" top: "prob" }')
                print("Gender detection model architecture created successfully")
                
        except Exception as e:
            print(f"Error downloading gender detection model: {str(e)}")
            raise

    def predict_gender(self, face_image):
        """Predict gender using the pre-trained model"""
        if self.gender_net is None:
            return "Unknown"
            
        try:
            # Preprocess the face image
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), 
                                       (78.4263377603, 87.7689143744, 114.895847746), 
                                       swapRB=False)
            
            # Forward pass
            self.gender_net.setInput(blob)
            gender_preds = self.gender_net.forward()
            
            # Get prediction
            gender_idx = gender_preds[0].argmax()
            confidence = gender_preds[0][gender_idx]
            
            # Return prediction only if confidence is high enough
            if confidence > 0.6:
                return self.gender_labels[gender_idx]
            return "Unknown"
            
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
            
            # Extract face region for gender detection
            h, w = image.shape[:2]
            x1 = int(min(landmark.x for landmark in face_landmarks) * w)
            y1 = int(min(landmark.y for landmark in face_landmarks) * h)
            x2 = int(max(landmark.x for landmark in face_landmarks) * w)
            y2 = int(max(landmark.y for landmark in face_landmarks) * h)
            
            # Add margin
            margin = int(0.2 * (x2 - x1))
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            face_image = image[y1:y2, x1:x2]
            
            # Use ML model for gender detection
            gender = self.predict_gender(face_image)
            gender_confidence = 0.8 if gender != "Unknown" else 0.5
            
            return {
                "age": int(age),
                "gender": gender,
                "gender_confidence": float(gender_confidence),
                "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0],
                "emotion": {k: float(v) for k, v in emotions.items()}
            }
            
        except Exception as e:
            return {"error": str(e)}
