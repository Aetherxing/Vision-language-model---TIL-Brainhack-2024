#nvm i lied


import matplotlib.pyplot as plt
import sklearn
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Define RGB ranges for each color category
color_ranges = {
    'black': ([0, 0, 0], [50, 50, 50]),
    'brown': ([101, 67, 33], [165, 105, 60]),
    'blue': ([0, 0, 150], [50, 50, 255]),
    'gray': ([100, 100, 100], [180, 180, 180]),
    'green': ([0, 100, 0], [50, 180, 50]),
    'orange': ([200, 100, 0], [255, 165, 0]),
    'pink': ([200, 100, 150], [255, 192, 203]),
    'purple': ([100, 0, 100], [180, 50, 180]),
    'red': ([150, 0, 0], [255, 50, 50]),
    'white': ([200, 200, 200], [255, 255, 255]),
    'yellow': ([200, 200, 0], [255, 255, 100])
}



# Generate synthetic RGB values and corresponding labels
num_samples_per_color = 1000
rgb_values = []
labels = []

for color, (low_rgb, high_rgb) in color_ranges.items():
    for _ in range(num_samples_per_color):
        rgb = [np.random.randint(low_rgb[i], high_rgb[i] + 1) for i in range(3)]
        rgb_values.append(rgb)
        labels.append(list(color_ranges.keys()).index(color))

# Convert lists to numpy arrays
rgb_values = np.array(rgb_values)
labels = np.array(labels)

# Split the dataset into training and validation sets
rgb_train, rgb_val, labels_train, labels_val = train_test_split(rgb_values, labels, test_size=0.2, random_state=42)

# Define and compile the model with regularization and dropout
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),  # Input shape for RGB values
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(11, activation='softmax')  # 11 output neurons for 11 color categories
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Implement early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Train the model with early stopping
history = model.fit(rgb_train, labels_train, epochs=200, batch_size=32, validation_data=(rgb_val, labels_val), callbacks=[early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(rgb_val, labels_val)
print("Validation accuracy:", test_acc)

train_test_loss, train_test_acc = model.evaluate(rgb_train, labels_train)
print("Testing accuract:", train_test_acc)

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model.save('color_classifier.h5')
