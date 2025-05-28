import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# # Load trained model
model = load_model("id_card_layout_detector.h5")

# Path to test image
img_path = "test_samples/11.jpg"  # replace with your image

# Preprocess the image
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
prediction = model.predict(img_array)[0][0]

# Interpret result
if prediction < 0.5:
    print("Prediction: FAKE layout")
else:
    print("Prediction: GENUINE  layout")

# Also show confidence
print("Confidence:", round(float(prediction), 4))


