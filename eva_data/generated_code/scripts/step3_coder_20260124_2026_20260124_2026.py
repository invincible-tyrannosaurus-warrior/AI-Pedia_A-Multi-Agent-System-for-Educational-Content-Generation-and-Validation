import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import os

# Set the output directory for generated assets
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic time series data
def generate_time_series(n_samples=1000):
    """
    Generate a synthetic time series with trend and seasonality
    """
    # Create time steps
    t = np.arange(n_samples)
    
    # Create trend component
    trend = 0.02 * t
    
    # Create seasonal component (sine wave)
    seasonal = 5 * np.sin(2 * np.pi * t / 50)
    
    # Create noise
    noise = np.random.normal(0, 1, n_samples)
    
    # Combine components
    series = trend + seasonal + noise
    
    return series

# Prepare data for RNN training
def prepare_data(series, sequence_length=50):
    """
    Prepare time series data for RNN training
    """
    X, y = [], []
    
    # For each sequence of length sequence_length, predict the next value
    for i in range(len(series) - sequence_length):
        X.append(series[i:i+sequence_length])
        y.append(series[i+sequence_length])
    
    return np.array(X), np.array(y)

# Generate time series data
print("Generating time series data...")
time_series = generate_time_series(1000)

# Prepare data for training
sequence_length = 50
X, y = prepare_data(time_series, sequence_length)

# Reshape data for RNN input (samples, time_steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data into training and testing sets
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Build the RNN model
print("Building RNN model...")
model = Sequential([
    # Simple RNN layer with 50 units
    SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
    # Dense layer to map RNN output to prediction
    Dense(1)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Display model architecture
model.summary()

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions on test data
print("Making predictions...")
predictions = model.predict(X_test)

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot MAE
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rnn_training_history.png'))
plt.close()

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))

# Plot a subset of test data for visualization
n_plot_points = 200
plt.plot(range(n_plot_points), y_test[:n_plot_points], label='Actual Values', alpha=0.7)
plt.plot(range(n_plot_points), predictions[:n_plot_points], label='Predictions', alpha=0.7)
plt.title('RNN Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rnn_predictions.png'))
plt.close()

# Plot the original time series with a portion of predictions overlaid
plt.figure(figsize=(15, 6))

# Plot the entire time series
plt.plot(range(len(time_series)), time_series, label='Original Time Series', alpha=0.7)

# Overlay a section of predictions
start_idx = len(y_test) - 100
end_idx = len(y_test)
plt.plot(range(start_idx, end_idx), predictions[start_idx:end_idx], 
         label='RNN Predictions', marker='o', markersize=2, linewidth=1)

plt.title('Original Time Series with RNN Predictions')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'rnn_time_series_with_predictions.png'))
plt.close()

print("RNN demonstration completed successfully!")
print(f"Generated plots saved to: {output_dir}")