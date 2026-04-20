import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Set the output directory
output_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_f4651d6f9ca74c4dbac3e919101d21e9/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic time series data
def generate_time_series(n_samples=1000):
    """Generate synthetic time series data with trend and seasonality"""
    # Create time index
    t = np.arange(n_samples)
    
    # Generate trend component
    trend = 0.02 * t
    
    # Generate seasonal component (monthly pattern)
    seasonal = 10 * np.sin(2 * np.pi * t / 30)
    
    # Generate noise
    noise = np.random.normal(0, 2, n_samples)
    
    # Combine components
    series = trend + seasonal + noise
    
    return series

# Generate data
np.random.seed(42)
data = generate_time_series(1000)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    """Create sequences of data for LSTM training"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length)])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length
sequence_length = 50

# Create sequences
X, y = create_sequences(scaled_data, sequence_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Reshape data for LSTM input (samples, time steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    # First LSTM layer with return sequences=True to stack LSTM layers
    LSTM(units=50, return_sequences=True, input_shape=(sequence_length, 1)),
    Dropout(0.2),  # Dropout for regularization
    
    # Second LSTM layer
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    
    # Dense layer to produce final output
    Dense(units=25),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Display model architecture
print("Model Architecture:")
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# Make predictions
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# Inverse transform predictions to original scale
train_predictions = scaler.inverse_transform(train_predictions)
test_predictions = scaler.inverse_transform(test_predictions)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'lstm_training_history.png'))
plt.close()

# Plot predictions vs actual values
plt.figure(figsize=(15, 6))

# Plot training predictions
plt.subplot(1, 2, 1)
plt.plot(y_train_actual, label='Actual Training Data', alpha=0.7)
plt.plot(train_predictions, label='Predicted Training Data', alpha=0.7)
plt.title('Training Data Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output_dir, 'lstm_training_predictions.png'))
plt.close()

# Plot test predictions
plt.subplot(1, 2, 2)
plt.plot(y_test_actual, label='Actual Test Data', alpha=0.7)
plt.plot(test_predictions, label='Predicted Test Data', alpha=0.7)
plt.title('Test Data Predictions')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.savefig(os.path.join(output_dir, 'lstm_test_predictions.png'))
plt.close()

# Save model
model.save(os.path.join(output_dir, 'lstm_model.h5'))

# Print model performance metrics
train_rmse = np.sqrt(np.mean((train_predictions.flatten() - y_train_actual.flatten())**2))
test_rmse = np.sqrt(np.mean((test_predictions.flatten() - y_test_actual.flatten())**2))

print(f"Training RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# Create a summary report
summary_text = f"""
LSTM Model Summary
==================
- Sequence Length: {sequence_length}
- Training Samples: {len(X_train)}
- Test Samples: {len(X_test)}
- Training RMSE: {train_rmse:.4f}
- Test RMSE: {test_rmse:.4f}

Model Architecture:
- LSTM Layer 1: 50 units, return_sequences=True
- Dropout: 0.2
- LSTM Layer 2: 50 units, return_sequences=False
- Dropout: 0.2
- Dense Layer: 25 units
- Output Layer: 1 unit

The model was trained for 20 epochs with Adam optimizer.
"""

with open(os.path.join(output_dir, 'lstm_summary.txt'), 'w') as f:
    f.write(summary_text)

print("Analysis complete. Results saved to:", output_dir)