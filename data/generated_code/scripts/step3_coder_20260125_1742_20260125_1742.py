import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam
import os

# Create the directory if it doesn't exist
os.makedirs('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets', exist_ok=True)

# Generate synthetic time series data for demonstration
def generate_time_series(n_samples=1000):
    """
    Generate synthetic time series data with seasonal patterns
    """
    # Create time steps
    t = np.arange(n_samples)
    
    # Create a combination of sine waves with different frequencies and noise
    signal = (0.5 * np.sin(0.1 * t) + 
              0.3 * np.sin(0.3 * t) + 
              0.2 * np.sin(0.7 * t))
    
    # Add some noise
    noise = np.random.normal(0, 0.1, n_samples)
    
    # Combine signal and noise
    data = signal + noise
    
    return data

# Prepare data for RNN training
def prepare_data(data, sequence_length=50):
    """
    Prepare time series data for RNN training by creating sequences
    """
    X, y = [], []
    
    # For each sequence in the data
    for i in range(len(data) - sequence_length):
        # Take a sequence of 'sequence_length' points
        sequence = data[i:i+sequence_length]
        # The target is the next point after the sequence
        target = data[i+sequence_length]
        
        X.append(sequence)
        y.append(target)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Reshape X for RNN input (samples, time_steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    
    return X, y

# Generate the time series data
print("Generating time series data...")
data = generate_time_series(1000)

# Prepare data for training
sequence_length = 50
X, y = prepare_data(data, sequence_length)

# Split into training and validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")

# Build the RNN model
print("Building RNN model...")
model = Sequential([
    # First RNN layer with return sequences=True to pass outputs to next layer
    SimpleRNN(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
    
    # Second RNN layer
    SimpleRNN(50, activation='relu'),
    
    # Dense layer for final prediction
    Dense(1)
])

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Display model architecture
print("Model Architecture:")
model.summary()

# Train the model
print("Training the model...")
history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=20,
    validation_data=(X_val, y_val),
    verbose=1
)

# Make predictions on validation set
predictions = model.predict(X_val)

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
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/rnn_training_history.png')
plt.close()

# Plot predictions vs actual values
plt.figure(figsize=(12, 6))
plt.plot(y_val[:100], label='Actual Values', alpha=0.7)
plt.plot(predictions[:100], label='Predictions', alpha=0.7)
plt.title('RNN Predictions vs Actual Values')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/rnn_predictions.png')
plt.close()

# Plot the original time series with a few predictions overlaid
plt.figure(figsize=(12, 6))
# Plot the last 100 points of original data
plt.plot(range(len(data)-100, len(data)), data[len(data)-100:], label='Original Data', linewidth=2)
# Plot a few predictions from the model
test_sequence = X_val[0].reshape(1, sequence_length, 1)
pred_values = []
for _ in range(20):
    pred = model.predict(test_sequence, verbose=0)[0][0]
    pred_values.append(pred)
    # Update sequence by removing first element and adding prediction
    test_sequence = np.roll(test_sequence, -1, axis=1)
    test_sequence[0, -1, 0] = pred

plt.plot(range(len(data)-100, len(data)-100+len(pred_values)), pred_values, 
         label='RNN Predictions', marker='o', markersize=4)
plt.title('RNN Time Series Forecasting')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.legend()
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/rnn_forecasting.png')
plt.close()

print("RNN demonstration completed successfully!")
print("Generated plots saved to D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/")