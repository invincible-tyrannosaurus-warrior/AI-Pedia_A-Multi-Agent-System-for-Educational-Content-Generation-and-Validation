import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# Create the directory if it doesn't exist
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Convert labels to categorical one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the CNN architecture
def create_cnn_model():
    """
    Creates a Convolutional Neural Network model for image classification
    """
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
    ])
    
    return model

# Create the model
model = create_cnn_model()

# Display model architecture
model.summary()

# Compile the model with optimizer, loss function, and metrics
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    x_train, y_train,
    batch_size=32,
    epochs=10,
    validation_data=(x_test, y_test),
    verbose=1
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cnn_training_history.png'))
plt.close()

# Make predictions on a few test samples
predictions = model.predict(x_test[:5])

# Display some test images with their predictions
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(5):
    # Display image
    axes[i].imshow(x_test[i])
    axes[i].set_title(f'True: {class_names[np.argmax(y_test[i])]}')
    axes[i].axis('off')
    
    # Add prediction text
    pred_class = class_names[np.argmax(predictions[i])]
    pred_confidence = np.max(predictions[i])
    axes[i].text(0.5, -0.1, f'Pred: {pred_class} ({pred_confidence:.2f})',
                transform=axes[i].transAxes, ha='center', va='top')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'cnn_predictions.png'))
plt.close()

# Save the trained model
model.save(os.path.join(output_dir, 'cnn_model.h5'))

print("CNN model training completed successfully!")
print(f"Final test accuracy: {test_accuracy:.4f}")