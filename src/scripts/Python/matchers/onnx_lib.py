import os
import cv2
import base64
import numpy as np
from scipy.spatial.distance import cosine
import onnxruntime as ort

class OnnxFaceMatcher:
    def __init__(self):        # Initialize ONNX Runtime session for face detection
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        det_model = os.path.join(model_dir, 'face_detection.onnx')
        rec_model = os.path.join(model_dir, 'face_recognition_optimized.onnx')
        
        # Create ONNX runtime sessions
        self.det_session = ort.InferenceSession(det_model, providers=['CPUExecutionProvider'])
        self.rec_session = ort.InferenceSession(rec_model, providers=['CPUExecutionProvider'])
        
        # Model configurations
        self.det_input_name = self.det_session.get_inputs()[0].name
        self.det_input_shape = (640, 640)  # YOLOv8 input size
        self.det_conf_threshold = 0.5
        self.det_iou_threshold = 0.5
        self.rec_input_shape = (112, 112)  # MobileFaceNet input size

    def load_image_from_source(self, image_source):
        """Load image from either file path or base64 string"""
        try:
            # Check if it's a base64 string
            if image_source.startswith('data:image/'):
                header, encoded = image_source.split(',', 1)
                image_data = base64.b64decode(encoded)
            elif self.is_base64(image_source):
                image_data = base64.b64decode(image_source)
            else:
                return cv2.imread(image_source), None
            
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None, "Could not decode image"
            
            return image, None
            
        except Exception as e:
            return None, f"Error loading image: {str(e)}"

    def is_base64(self, s):
        """Check if string is valid base64"""
        try:
            if isinstance(s, str):
                if len(s) % 4 == 0 and all(c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=' for c in s):
                    base64.b64decode(s, validate=True)
                    return True
            return False
        except Exception:
            return False

    def preprocess_image(self, image, target_size, for_detection=True):
        """Preprocess image for model input"""
        if for_detection:
            # YOLOv8 preprocessing
            img = cv2.resize(image, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.transpose((2, 0, 1))
            img = np.ascontiguousarray(img)
            img = img.astype(np.float32)
            img /= 255.0
            return img[np.newaxis, ...]
        else:
            # MobileFaceNet preprocessing
            img = cv2.resize(image, target_size)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = (img - 127.5) / 128.0
            img = img.transpose((2, 0, 1))
            return img[np.newaxis, ...].astype(np.float32)

    def detect_face(self, image):
        """Detect face in image using YOLOv8-face ONNX model"""
        try:
            orig_height, orig_width = image.shape[:2]
            
            # Preprocess image
            preprocessed = self.preprocess_image(image, self.det_input_shape)
            
            # Run inference
            outputs = self.det_session.run(None, {self.det_input_name: preprocessed})[0]
            
            # Post-process outputs (YOLOv8 format)
            boxes = []
            scores = []
            
            for row in outputs[0]:
                conf = row[4]
                if conf < self.det_conf_threshold:
                    continue
                
                x1, y1, x2, y2 = row[0:4]
                boxes.append([x1, y1, x2, y2])
                scores.append(conf)
            
            if not boxes:
                return None, "No face detected"
            
            # Get highest confidence detection
            best_idx = np.argmax(scores)
            box = boxes[best_idx]
            
            # Scale box to original image size
            box = [
                int(box[0] * orig_width / self.det_input_shape[0]),
                int(box[1] * orig_height / self.det_input_shape[1]),
                int(box[2] * orig_width / self.det_input_shape[0]),
                int(box[3] * orig_height / self.det_input_shape[1])
            ]
            
            return box, None
            
        except Exception as e:
            return None, str(e)

    def get_face_embeddings(self, image):
        """Extract face embeddings using MobileFaceNet ONNX model"""
        try:
            # Detect face
            face_box, error = self.detect_face(image)
            if error:
                return None, error
                
            # Extract and align face
            x1, y1, x2, y2 = face_box
            face = image[y1:y2, x1:x2]
            
            # Preprocess for recognition model
            face = self.preprocess_image(face, self.rec_input_shape, for_detection=False)
            
            # Get embeddings
            embedding = self.rec_session.run(None, {self.rec_session.get_inputs()[0].name: face})[0][0]
            return embedding, None
            
        except Exception as e:
            return None, str(e)

    def compare_faces(self, image1_source, image2_source, threshold=0.4):
        """Compare faces using face embeddings"""
        # Load images
        image1, error1 = self.load_image_from_source(image1_source)
        if error1:
            return {"error": f"Image 1: {error1}"}
            
        image2, error2 = self.load_image_from_source(image2_source)
        if error2:
            return {"error": f"Image 2: {error2}"}

        # Get embeddings
        embedding1, error1 = self.get_face_embeddings(image1)
        if error1:
            return {"error": f"Image 1: {error1}"}
            
        embedding2, error2 = self.get_face_embeddings(image2)
        if error2:
            return {"error": f"Image 2: {error2}"}
        
        # Calculate cosine similarity
        similarity = 1 - cosine(embedding1, embedding2)
        
        # Calculate confidence (normalize similarity to 0-1 range)
        confidence = max(0, min(1, similarity))
        
        # Determine if it's a match
        match = similarity > threshold
        
        return {
            "match": bool(match),
            "confidence": float(confidence),
            "similarity": float(similarity)
        }
        
    def analyze_face(self, image_source):
        """Analyze face attributes"""
        try:
            # Load image
            image, error = self.load_image_from_source(image_source)
            if error:
                return {"error": error}
            
            # Detect face
            face_box, error = self.detect_face(image)
            if error:
                return {"error": error}
                
            # For now, return simplified analysis
            return {
                "age": 25,  # Placeholder
                "gender": "Unknown",
                "gender_confidence": 0.0,
                "dominant_emotion": "neutral",
                "emotion": {
                    "neutral": 1.0,
                    "happy": 0.0,
                    "sad": 0.0,
                    "surprised": 0.0,
                    "angry": 0.0
                }
            }
            
        except Exception as e:
            return {"error": str(e)}
