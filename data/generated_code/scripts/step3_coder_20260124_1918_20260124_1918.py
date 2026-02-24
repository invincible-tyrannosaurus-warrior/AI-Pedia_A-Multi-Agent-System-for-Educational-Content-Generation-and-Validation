import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import os

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic time series data for demonstration
def generate_time_series(n_samples=1000):
    """
    Generate a synthetic time series with trend and seasonality
    """
    # Create time steps
    t = np.arange(n_samples)
    
    # Create a combination of trend, seasonality, and noise
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50) + 5 * np.cos(2 * np.pi * t / 25)
    noise = np.random.normal(0, 2, n_samples)
    
    # Combine components
    series = trend + seasonal + noise
    
    return series

# Prepare data for RNN training
def prepare_data(series, sequence_length=50):
    """
    Prepare time series data for RNN training
    """
    X, y = [], []
    
    # Create sequences of length sequence_length
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

# Reshape data for RNN input (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train and test sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Build the RNN model
print("Building RNN model...")
model = Sequential([
    # Simple RNN layer with 50 units
    SimpleRNN(50, activation='relu', input_shape=(sequence_length, 1)),
    # Dense layer for output
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Display model architecture
print("Model Architecture:")
model.summary()

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
print("Making predictions...")
predictions = model.predict(X_test)

# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rnn_training_history.png'))
plt.close()

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_test[:100], label='Actual Values', alpha=0.7)
plt.plot(predictions[:100], label='Predictions', alpha=0.7)
plt.title('RNN Predictions vs Actual Values')
plt.xlabel('Time Steps')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rnn_predictions.png'))
plt.close()

# Plot the original time series with a portion of predictions overlaid
plt.figure(figsize=(12, 6))
# Plot the entire time series
plt.plot(range(len(time_series)), time_series, label='Original Time Series', alpha=0.7)
# Highlight the test period
test_start_idx = len(time_series) - len(y_test)
plt.plot(range(test_start_idx, len(time_series)), y_test, label='Actual Test Values', alpha=0.7)
plt.plot(range(test_start_idx, test_start_idx + len(predictions)), predictions, 
         label='RNN Predictions', alpha=0.7)
plt.title('RNN Performance on Time Series Data')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'rnn_performance.png'))
plt.close()

print("RNN demonstration completed successfully!")
print(f"Results saved to: {output_dir}")