#classifier finale code (highest recored training accuracy: 0.9011 --> 300 epochs)

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, ImageFile
import os

# Enable PIL to handle truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Function to check for corrupted images
def check_images(directory):
    corrupted_images = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
                file_path = os.path.join(root, file)
                try:
                    img = Image.open(file_path)
                    img.verify()  # Verify that it is, in fact, an image
                except (IOError, SyntaxError, OSError) as e:
                    print(f'Corrupt image: {file_path} - {e}')
                    corrupted_images.append(file_path)
    return corrupted_images

# Remove corrupted images
def remove_corrupted_images(corrupted_images):
    for file_path in corrupted_images:
        os.remove(file_path)

# Check and remove corrupted images
corrupted_train_images = check_images('/content/dataset/training')
corrupted_validation_images = check_images('/content/dataset/validation')
remove_corrupted_images(corrupted_train_images)
remove_corrupted_images(corrupted_validation_images)


num_classes = 4

# Define your dataset directories
train_dir = '/content/dataset/training'
validation_dir = '/content/dataset/validation'

# Create ImageDataGenerators for data augmentation and normalization
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(32, 32),  # Adjust the target size as needed
    batch_size=32,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(32, 32),  # Adjust the target size as needed
    batch_size=32,
    class_mode='sparse'
)

# Build your CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_classes))  # Output layer with 4 neurons


model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
history = model.fit(train_generator,
                    epochs=300,
                    validation_data=validation_generator)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(validation_generator)
print("Test accuracy:", test_acc)


# Save the model
model.save("aircraft_classifier_finale.h5")