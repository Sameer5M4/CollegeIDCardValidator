import tensorflow as tf
import numpy as np
import cv2 # We use OpenCV for color conversion, which is standard

# --- Configuration ---
MODEL_PATH = 'id_card_validator_final.keras'  # Path to your saved .keras model
IMG_WIDTH, IMG_HEIGHT = 224, 224

# --- Load the Model (once, when the server starts) ---
try:
    print("Loading template validation model...")
    # This model is loaded into memory only one time
    template_model = tf.keras.models.load_model(MODEL_PATH)
    print("Template model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load template model from '{MODEL_PATH}'.")
    # In a real server, you might want the app to exit if the model can't load.
    template_model = None

# --- Main Function to be called by FastAPI ---

def run_template_check(image_np: np.ndarray) -> dict:
    """
    Takes an image as a NumPy array and returns a template validation score.

    Args:
        image_np: The image read by OpenCV (in BGR format).

    Returns:
        A dictionary with the confidence score, e.g., {"score": 0.95}
    """
    if template_model is None:
        # This is a fallback in case the model failed to load
        return {"score": 0.0, "details": "Model not loaded"}

    try:
        # 1. Preprocess the image
        # Convert from BGR (OpenCV's default) to RGB (TensorFlow's default)
        rgb_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        
        # Resize image to the model's expected input size
        resized_image = tf.image.resize(rgb_image, [IMG_HEIGHT, IMG_WIDTH])
        
        # Expand dimensions to create a batch of 1
        img_array_expanded = tf.expand_dims(resized_image, 0)
        
        # Use the same preprocessing as during training
        processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)

        # 2. Make prediction
        prediction_scores = template_model.predict(processed_img)
        
        # The raw sigmoid output is the confidence score for the "genuine" class
        # It's a value between 0.0 and 1.0
        genuine_score = float(prediction_scores[0][0])

        # 3. Return the score in the required format
        return {
            "score": genuine_score,
            "details": "Template validation complete"
        }

    except Exception as e:
        print(f"ERROR in template validation: {e}")
        return {"score": 0.0, "details": str(e)}