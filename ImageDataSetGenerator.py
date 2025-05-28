import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

# Paths
input_dir = 'dataset/fake'         # Folder where your raw images are
output_dir = 'dataset/GenuineAugmented_fake'       # Folder to save augmented images
os.makedirs(output_dir, exist_ok=True)

# Augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Number of augmented images per original
num_augmented = 10

# Loop over each image
for filename in os.listdir(input_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        img_path = os.path.join(input_dir, filename)
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)

        # Generate 'num_augmented' images
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir=output_dir,
                                  save_prefix=filename.split('.')[0],
                                  save_format='jpg'):
            i += 1
            if i >= num_augmented:
                break
