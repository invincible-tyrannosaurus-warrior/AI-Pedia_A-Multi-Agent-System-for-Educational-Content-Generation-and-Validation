import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Create a synthetic dataset for demonstration
# This simulates a real-world classification problem with 10 features
X, y = make_classification(
    n_samples=1000,      # Number of samples
    n_features=10,       # Number of features
    n_informative=8,     # Number of informative features
    n_redundant=2,       # Number of redundant features
    n_classes=3,         # Number of classes
    random_state=42
)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale the features to have zero mean and unit variance
# This is important for neural networks to converge properly
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert labels to categorical one-hot encoding
# Required for multi-class classification in Keras
y_train_categorical = to_categorical(y_train)
y_test_categorical = to_categorical(y_test)

# Build the deep learning model
# Sequential model - layers are added one after another
model = Sequential([
    # First hidden layer with 64 neurons and ReLU activation
    # ReLU helps with vanishing gradient problem
    Dense(64, activation='relu', input_shape=(10,)),
    
    # Second hidden layer with 32 neurons
    Dense(32, activation='relu'),
    
    # Third hidden layer with 16 neurons
    Dense(16, activation='relu'),
    
    # Output layer with 3 neurons (one for each class)
    # Softmax activation for multi-class classification
    Dense(3, activation='softmax')
])

# Compile the model
# Specify optimizer, loss function, and metrics
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Adam optimizer with learning rate
    loss='categorical_crossentropy',      # Loss function for multi-class classification
    metrics=['accuracy']                   # Track accuracy during training
)

# Display model architecture
print("Model Architecture:")
model.summary()

# Train the model
# Fit the model on training data
history = model.fit(
    X_train_scaled, y_train_categorical,
    epochs=50,                    # Number of training iterations
    batch_size=32,                # Number of samples per gradient update
    validation_split=0.2,         # Use 20% of training data for validation
    verbose=1                     # Show training progress
)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test_categorical, verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Make predictions on test set
y_pred_prob = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_prob, axis=1)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Save the plot
plt.tight_layout()
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/training_history.png')
plt.close()

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Save confusion matrix plot
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/confusion_matrix.png')
plt.close()

# Save model weights
model.save_weights('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/deep_learning_model_weights.h5')

# Save training results to CSV
results_df = pd.DataFrame({
    'epoch': range(1, len(history.history['accuracy']) + 1),
    'train_accuracy': history.history['accuracy'],
    'val_accuracy': history.history['val_accuracy'],
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

results_df.to_csv('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/training_results.csv', index=False)

print("Deep learning example completed successfully!")
print("Generated files saved to D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/")