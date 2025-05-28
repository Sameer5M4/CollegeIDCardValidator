import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt

# Set image size (same as training)
IMAGE_SIZE = (224, 224)

# Load the trained model
model = tf.keras.models.load_model('resnet_template_model2.h5')

class_labels = ['genuine', 'fake']  
def predict_single_image(image_path):
    # Load and preprocess image
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Prediction
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    # Print results
    print(f"Predicted Class: {class_labels[class_index]}")
    print(f"Confidence: {prediction[0][class_index]*100:.2f}%")

    # Show image with title
    plt.imshow(img)
    plt.title(f"Predicted: {class_labels[class_index]}")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    # Hardcode your test image path here
    image_path = 'test_samples/7.jpg'
    predict_single_image(image_path)

