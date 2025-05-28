import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import seaborn as sns
import os
import shutil
from sklearn.utils.class_weight import compute_class_weight

# --- Configuration ---
DATASET_PATH = 'finalDataset'
MODEL_SAVE_PATH = 'id_card_validator_final.keras' # New model filename with .keras extension
LOG_DIR = 'logs_final_validator'

IMG_WIDTH, IMG_HEIGHT = 224, 224
BASE_MODEL_NAME = 'EfficientNetB0' # Options: 'EfficientNetB0', 'EfficientNetB1'

BATCH_SIZE = 16
EPOCHS_INITIAL_TRAINING = 20
EPOCHS_FINE_TUNING = 15
LEARNING_RATE_INITIAL = 1e-3
LEARNING_RATE_FINE_TUNE = 5e-6
VALIDATION_SPLIT = 0.2

# --- 1. Prepare Data ---

if os.path.exists(MODEL_SAVE_PATH):
    os.remove(MODEL_SAVE_PATH)
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Dataset path '{DATASET_PATH}' not found.")
    exit()
if not os.path.exists(os.path.join(DATASET_PATH, 'genuine')) or \
   not os.path.exists(os.path.join(DATASET_PATH, 'fake')):
    print(f"ERROR: Subfolders 'genuine' and 'fake' not found inside '{DATASET_PATH}'.")
    exit()

train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest',
    brightness_range=[0.9, 1.1],
    validation_split=VALIDATION_SPLIT
)

validation_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    validation_split=VALIDATION_SPLIT
)

train_generator = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print("Class Indices:", train_generator.class_indices)
if len(train_generator.class_indices) != 2:
    print(f"Error: Expected 2 classes, but found {len(train_generator.class_indices)}.")
    exit()

class_weights_calculated = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights_calculated))
print(f"Calculated Class Weights: {class_weight_dict}")

# --- 2. Build the Model ---

def build_model_for_id_validation(num_classes_output=1, learning_rate=LEARNING_RATE_INITIAL):
    if BASE_MODEL_NAME == 'EfficientNetB0':
        base_model_fn = tf.keras.applications.EfficientNetB0
        img_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    elif BASE_MODEL_NAME == 'EfficientNetB1': # Example if you wanted to switch
        base_model_fn = tf.keras.applications.EfficientNetB1
        img_shape = (240, 240, 3) # B1 default input
        if IMG_HEIGHT != 240 or IMG_WIDTH != 240:
            print("Warning: IMG_HEIGHT/WIDTH not 240x240 for EfficientNetB1. Adjust or expect issues.")
    else:
        raise ValueError(f"Unsupported BASE_MODEL_NAME: {BASE_MODEL_NAME}")

    base_model_instance = base_model_fn(
        weights='imagenet',
        include_top=False,
        input_shape=img_shape
    )
    base_model_instance.trainable = False

    constructed_model = Sequential([
        base_model_instance,
        GlobalAveragePooling2D(name="gap"),
        BatchNormalization(),
        Dropout(0.4, name="dropout_1"),
        Dense(256, activation='relu', name="fc1"),
        BatchNormalization(),
        Dropout(0.4, name="dropout_2"),
        Dense(num_classes_output, activation='sigmoid', name="output_sigmoid")
    ])

    optimizer_instance = Adam(learning_rate=learning_rate)
    constructed_model.compile(
        optimizer=optimizer_instance,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    return constructed_model, base_model_instance

model, base_model_ref = build_model_for_id_validation() # Changed variable name for base_model
model.summary()

# --- 3. Initial Training Phase ---
print("\n--- Initial Training (Training new head) ---")

initial_callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_auc',
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,
        verbose=1,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

history_initial = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL_TRAINING,
    validation_data=validation_generator,
    callbacks=initial_callbacks,
    class_weight=class_weight_dict
)

# --- 4. Fine-Tuning Phase ---
print("\n--- Fine-Tuning ---")

loaded_model_for_finetune = None
base_model_for_finetune = None
history_fine_tune = None # Initialize

print(f"Loading best model from initial phase: {MODEL_SAVE_PATH}")
try:
    loaded_model_for_finetune = tf.keras.models.load_model(MODEL_SAVE_PATH)
    if isinstance(loaded_model_for_finetune.layers[0], tf.keras.Model):
        base_model_for_finetune = loaded_model_for_finetune.layers[0]
    else:
        print("Warning: First layer of loaded model is not a Keras Model. Cannot access base_model for fine-tuning.")
except Exception as e:
    print(f"Error loading model for fine-tuning: {e}")

if loaded_model_for_finetune and base_model_for_finetune:
    base_model_for_finetune.trainable = True
    
    # Determine fine_tune_at_layer_name based on BASE_MODEL_NAME
    if BASE_MODEL_NAME.startswith('EfficientNetB0'):
        fine_tune_from_layer = 'block5a_expand_conv'
    elif BASE_MODEL_NAME.startswith('EfficientNetB1'):
        fine_tune_from_layer = 'block5a_expand_conv' # Adjust if different for B1
    else: # Default or other models
        fine_tune_from_layer = 'block6a_expand_conv'

    set_trainable_flag = False
    layer_found_for_finetune = False
    for layer in base_model_for_finetune.layers:
        if layer.name == fine_tune_from_layer:
            set_trainable_flag = True
            layer_found_for_finetune = True
        if set_trainable_flag:
            layer.trainable = not isinstance(layer, BatchNormalization)
        else:
            layer.trainable = False
    
    if not layer_found_for_finetune:
        print(f"Warning: Fine-tune layer '{fine_tune_from_layer}' not found. Unfreezing all non-BN layers in base model.")
        for layer in base_model_for_finetune.layers:
            layer.trainable = not isinstance(layer, BatchNormalization)

    optimizer_finetune_instance = Adam(learning_rate=LEARNING_RATE_FINE_TUNE)
    loaded_model_for_finetune.compile(
        optimizer=optimizer_finetune_instance,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')]
    )
    print("Model summary after setting up for fine-tuning:")
    loaded_model_for_finetune.summary()

    fine_tuning_callbacks = [
        ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=4,
            min_lr=1e-8,
            verbose=1
        )
    ]

    history_fine_tune = loaded_model_for_finetune.fit(
        train_generator,
        epochs=EPOCHS_INITIAL_TRAINING + EPOCHS_FINE_TUNING,
        initial_epoch=history_initial.epoch[-1] + 1 if history_initial and history_initial.epoch else 0,
        validation_data=validation_generator,
        callbacks=fine_tuning_callbacks,
        class_weight=class_weight_dict
    )
else:
    print("Skipping fine-tuning phase due to issues with loading or accessing the base model.")

# --- 5. Evaluate the Final Model ---
print("\n--- Evaluating Final Model ---")

final_best_model = None
if os.path.exists(MODEL_SAVE_PATH):
    print(f"Loading best performing model from: {MODEL_SAVE_PATH}")
    final_best_model = tf.keras.models.load_model(MODEL_SAVE_PATH)
else:
    print(f"Error: Model file {MODEL_SAVE_PATH} not found. Using model in memory (if available from last training step).")
    # Fallback to the model in memory from fine-tuning if file not found (less ideal)
    final_best_model = loaded_model_for_finetune if loaded_model_for_finetune else model

if final_best_model:
    loss, acc, prec, rec, auc_val = final_best_model.evaluate(validation_generator, verbose=1)
    print(f"\nValidation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {acc:.4f}")
    print(f"Validation Precision: {prec:.4f}")
    print(f"Validation Recall: {rec:.4f}")
    print(f"Validation AUC: {auc_val:.4f}")

    Y_pred_probs = final_best_model.predict(validation_generator)
    y_true_labels = validation_generator.classes

    precisions, recalls, thresholds = precision_recall_curve(y_true_labels, Y_pred_probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    f1_scores = np.nan_to_num(f1_scores)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5 # ensure index is valid
    print(f"Optimal Threshold (max F1): {optimal_threshold:.4f} with F1-score: {f1_scores[optimal_idx]:.4f}")

    Y_pred_optimal = (Y_pred_probs > optimal_threshold).astype(int).reshape(-1)
    class_names_for_report = list(train_generator.class_indices.keys())

    print("\nClassification Report (with Optimal Threshold):")
    print(classification_report(y_true_labels, Y_pred_optimal, target_names=class_names_for_report))

    print("\nConfusion Matrix (with Optimal Threshold):")
    cm = confusion_matrix(y_true_labels, Y_pred_optimal)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_for_report, yticklabels=class_names_for_report)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (Threshold: {optimal_threshold:.2f})')
    plt.show()
else:
    print("No model available for evaluation.")


# --- Plot Training History ---
def plot_combined_training_history(initial_hist, finetune_hist=None):
    if not initial_hist:
        print("No initial history to plot.")
        return

    metrics_to_plot = ['accuracy', 'loss', 'auc', 'precision', 'recall']
    num_plots = len([m for m in metrics_to_plot if m in initial_hist.history or (finetune_hist and m in finetune_hist.history)])
    
    if num_plots == 0:
        print("No standard metrics found in history objects.")
        return

    plt.figure(figsize=(6 * num_plots, 5)) # Adjust figure size based on number of plots

    plot_idx = 1
    for metric_name in metrics_to_plot:
        train_metric = initial_hist.history.get(metric_name, [])
        val_metric = initial_hist.history.get(f'val_{metric_name}', [])

        if finetune_hist:
            train_metric.extend(finetune_hist.history.get(metric_name, []))
            val_metric.extend(finetune_hist.history.get(f'val_{metric_name}', []))
        
        if not train_metric and not val_metric: # Skip if metric not found
            continue

        epochs_range = range(len(train_metric))
        plt.subplot(1, num_plots, plot_idx)
        if train_metric: plt.plot(epochs_range, train_metric, label=f'Training {metric_name.capitalize()}')
        if val_metric: plt.plot(epochs_range, val_metric, label=f'Validation {metric_name.capitalize()}')
        
        if finetune_hist and initial_hist.epoch:
            plt.axvline(len(initial_hist.epoch)-1, color='gray', linestyle='--', label='Fine-tune Start')
        
        plt.legend(loc='best')
        plt.title(metric_name.capitalize())
        plt.xlabel('Epoch')
        plt.ylabel(metric_name.capitalize())
        plot_idx += 1
    
    plt.suptitle('Model Training History', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if history_initial: # Only plot if initial training happened
    plot_combined_training_history(history_initial, history_fine_tune)
else:
    print("Initial training history not available to plot.")

print(f"\nModel training complete. Best model saved to {MODEL_SAVE_PATH if os.path.exists(MODEL_SAVE_PATH) else 'Not saved (or error occurred)'}")

# --- 6. Example Prediction Function ---
def predict_id_image(image_path, model_instance, class_idx_map, decision_threshold=0.5, target_img_height=IMG_HEIGHT, target_img_width=IMG_WIDTH):
    try:
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(target_img_height, target_img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array_expanded = tf.expand_dims(img_array, 0)

        processed_img = tf.keras.applications.efficientnet.preprocess_input(img_array_expanded)
        prediction_scores = model_instance.predict(processed_img)
        raw_confidence = prediction_scores[0][0]

        # Determine label based on which class index is '1' (positive class for sigmoid)
        predicted_label_str = "Unknown"
        confidence_in_label = 0.0

        # Assuming class_idx_map is like {'genuine': 0, 'fake': 1} or vice-versa
        if class_idx_map.get('fake') == 1: # 'fake' is the positive class
            predicted_label_str = "fake" if raw_confidence > decision_threshold else "genuine"
            confidence_in_label = raw_confidence if predicted_label_str == "fake" else 1 - raw_confidence
        elif class_idx_map.get('genuine') == 1: # 'genuine' is the positive class
            predicted_label_str = "genuine" if raw_confidence > decision_threshold else "fake"
            confidence_in_label = raw_confidence if predicted_label_str == "genuine" else 1 - raw_confidence
        else:
            # Fallback if class names are different or not standard 0/1 mapping for 'fake'/'genuine'
            print(f"Warning: Standard 'fake'/'genuine' mapping to 0/1 not found in class_idx_map: {class_idx_map}. Using raw decision.")
            # This part assumes the positive class (index 1) is what the sigmoid predicts.
            # You might need to adjust if your class indices are arbitrary.
            positive_class_name = [name for name, idx in class_idx_map.items() if idx == 1]
            negative_class_name = [name for name, idx in class_idx_map.items() if idx == 0]
            if positive_class_name and negative_class_name:
                predicted_label_str = positive_class_name[0] if raw_confidence > decision_threshold else negative_class_name[0]
                confidence_in_label = raw_confidence if raw_confidence > decision_threshold else 1 - raw_confidence
            else: # Failsafe
                predicted_label_str = "Positive Class" if raw_confidence > decision_threshold else "Negative Class"


        print(f"\nPrediction for: {os.path.basename(image_path)}")
        print(f"  Raw Sigmoid Output: {raw_confidence:.4f} (Threshold: {decision_threshold:.2f})")
        print(f"  Predicted Label: {predicted_label_str}")
        print(f"  Confidence in Predicted Label: {confidence_in_label*100:.2f}%")

        plt.imshow(tf.keras.preprocessing.image.load_img(image_path))
        plt.title(f"Predicted: {predicted_label_str} ({confidence_in_label*100:.2f}%)")
        plt.axis('off')
        plt.show()
        return predicted_label_str, confidence_in_label

    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
    except Exception as e:
        print(f"Error predicting image {image_path}: {e}")
    return "Error", 0.0

# Example Usage (after training and evaluation, assuming `final_best_model` and `optimal_threshold` exist)
# if final_best_model and 'optimal_threshold' in locals():
#     test_image_example_path = "test_samples/your_test_image.jpg" # REPLACE with an actual image path
#     if os.path.exists(test_image_example_path):
#         predict_id_image(test_image_example_path, final_best_model, train_generator.class_indices, decision_threshold=optimal_threshold)
#     else:
#         print(f"Test image not found: {test_image_example_path}. Skipping single image prediction example.")