import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
MODEL_PATH = 'id_card_validator_final.keras'  # Path to your saved .keras model
IMAGE_TO_TEST_PATH = 'test_samples/18.jpg' # <<< REPLACE with the actual path to an image

# Model input image dimensions (must match what the model was trained on)
IMG_WIDTH, IMG_HEIGHT = 224, 224

# **IMPORTANT ASSUMPTIONS - Adjust these based on your training setup**
# 1. Define your class labels
CLASS_LABEL_POSITIVE = "genuine"  # The class name if sigmoid output > threshold
CLASS_LABEL_NEGATIVE = "fake" # The class name if sigmoid output <= threshold

# 2. Decision threshold (use 0.5 or your optimal one)
DECISION_THRESHOLD = 0.9459 # From your previous results, or use 0.5 as a default

# --- Load the Trained Model ---
print(f"Loading model from: {MODEL_PATH}")
if not os.path.exists(MODEL_PATH):
    print(f"ERROR: Model file not found at '{MODEL_PATH}'")
    exit()
try:
    trained_model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# --- Predict on a Single Image ---
if not os.path.exists(IMAGE_TO_TEST_PATH):
    print(f"ERROR: Image file not found at '{IMAGE_TO_TEST_PATH}'")
    exit()

try:
    # 1. Load the image
    img_display = tf.keras.preprocessing.image.load_img( # For display
        IMAGE_TO_TEST_PATH, target_size=(IMG_HEIGHT, IMG_WIDTH)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img_display)
    # 2. Expand dimensions to create a batch of 1
    img_array_expanded = tf.expand_dims(img_array, 0)
    # 3. Preprocess the image (must match training preprocessing)
    processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)

    # 4. Make prediction
    prediction_scores = trained_model.predict(processed_img)
    raw_confidence_for_positive_class = prediction_scores[0][0] # Sigmoid output

    # 5. Determine predicted label and confidence
    predicted_label_str = ""
    confidence_in_label = 0.0

    if raw_confidence_for_positive_class > DECISION_THRESHOLD:
        predicted_label_str = CLASS_LABEL_POSITIVE
        confidence_in_label = raw_confidence_for_positive_class
    else:
        predicted_label_str = CLASS_LABEL_NEGATIVE
        confidence_in_label = 1 - raw_confidence_for_positive_class

    print(f"\n--- Prediction for: {os.path.basename(IMAGE_TO_TEST_PATH)} ---")
    print(f"  Raw Sigmoid Output (for '{CLASS_LABEL_POSITIVE}'): {raw_confidence_for_positive_class:.4f}")
    print(f"  Decision Threshold: {DECISION_THRESHOLD:.2f}")
    print(f"  Predicted Label: {predicted_label_str}")
    print(f"  Confidence in Predicted Label: {confidence_in_label*100:.2f}%")

    # 6. Display the image with the prediction
    plt.imshow(img_display) # Show the original loaded image
    plt.title(f"Predicted: {predicted_label_str} ({confidence_in_label*100:.2f}%)")
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"An error occurred during prediction: {e}")
    import traceback
    traceback.print_exc()

print("\nSingle image prediction script finished.")