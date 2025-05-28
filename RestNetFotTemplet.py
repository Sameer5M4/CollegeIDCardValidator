import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import os

# Load and compile the model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
NUM_CLASSES = 2  # genuine and fake

# Load base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # Freeze layers

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data = train_datagen.flow_from_directory(
    'dataset2/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
val_data = train_datagen.flow_from_directory(
    'dataset/',
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Train the model
history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS)

# Save the model
model.save("resnet_template_model2.h5")

# -------------------------
# 📊 Print precision, recall, f1
# -------------------------

# Get predictions
y_pred = model.predict(val_data)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_data.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes, target_names=val_data.class_indices.keys()))

# -------------------------
# 📷 Predict on Single Image
# -------------------------

def predict_single_image(image_path, model):
    img = image.load_img(image_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    class_labels = ['genuine', 'fake']  # manually defined order


    print(f"\nPredicted Class: {class_labels[class_index]}")
    print(f"Confidence: {prediction[0][class_index]*100:.2f}%")

    # Display image
    plt.imshow(img)
    plt.title(f"Predicted: {class_labels[class_index]}")
    plt.axis('off')
    plt.show()

# 🔍 Test with a sample image
predict_single_image("test_samples/1.jpg", model)
