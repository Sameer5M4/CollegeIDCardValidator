# In face_validator.py

import cv2
import numpy as np
import os

# --- Configuration ---
PROTOTXT_PATH = "deploy.prototxt.txt"
WEIGHTS_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
MIN_FACE_CONFIDENCE = 0.7
MIN_BLUR_SCORE = 100.0
MIN_FACE_SIZE_PIXELS = 25

# --- Load Model (once, when the server starts) ---
def load_face_detection_model(prototxt_path, weights_path):
    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        print(f"FATAL ERROR: Face model files not found. Ensure '{prototxt_path}' and '{weights_path}' exist.")
        return None
    return cv2.dnn.readNetFromCaffe(prototxt_path, weights_path)

print("Loading face detection model...")
face_detection_net = load_face_detection_model(PROTOTXT_PATH, WEIGHTS_PATH)
if face_detection_net:
    print("Face detection model loaded successfully.")

# --- Main Function to be called by FastAPI ---
# V V V THIS IS THE IMPORTANT LINE V V V
def run_face_check(image_np: np.ndarray) -> dict:
    """
    Validates the face on an ID card from a NumPy image array.
    Returns a dictionary with a validation score and details.
    """
    if face_detection_net is None:
        return {"score": 0.0, "details": "Face model not loaded."}

    try:
        (h, w) = image_np.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image_np, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_detection_net.setInput(blob)
        detections = face_detection_net.forward()

        detected_faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > MIN_FACE_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                area = (endX - startX) * (endY - startY)
                if area > 0:
                    detected_faces.append({
                        "box": (startX, startY, endX, endY),
                        "confidence": float(confidence), "area": area
                    })

        if not detected_faces:
            return {"score": 0.0, "details": "No human face detected."}

        # Pick the largest face, assuming it's the main ID photo
        best_face = max(detected_faces, key=lambda x: x['area'])
        (startX, startY, endX, endY) = best_face["box"]
        face_roi = image_np[startY:endY, startX:endX]
        
        if face_roi.size == 0:
             return {"score": 0.1, "details": "Detected face region is empty."}

        # Check face size
        face_h, face_w = face_roi.shape[:2]
        if face_w < MIN_FACE_SIZE_PIXELS or face_h < MIN_FACE_SIZE_PIXELS:
            return {"score": 0.2, "details": f"Face too small ({face_w}x{face_h}px)."}

        # Check face clarity (blur)
        gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray_face_roi, cv2.CV_64F).var()
        if blur_score < MIN_BLUR_SCORE:
            return {"score": 0.4, "details": f"Face is too blurry (Score: {blur_score:.2f})."}

        # If all checks pass, the score is the detection confidence
        final_score = best_face["confidence"]
        return {
            "score": final_score,
            "details": f"Face validated with confidence {final_score:.2f}"
        }

    except Exception as e:
        print(f"ERROR in face validation: {e}")
        return {"score": 0.0, "details": str(e)}