import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic time series data
def generate_time_series(n_samples=1000):
    """Generate a synthetic time series with trend and seasonality"""
    t = np.linspace(0, 100, n_samples)
    # Create a combination of sine waves with different frequencies and noise
    series = (
        0.5 * np.sin(0.1 * t) +           # Low frequency component
        0.3 * np.sin(0.5 * t) +           # Medium frequency component
        0.2 * np.sin(2.0 * t) +           # High frequency component
        0.1 * np.sin(5.0 * t) +           # Very high frequency component
        0.05 * np.random.randn(n_samples) # Noise
    )
    return series

# Prepare data for training
def prepare_data(series, sequence_length=50):
    """Prepare time series data for RNN training"""
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_series) - sequence_length):
        X.append(scaled_series[i:i+sequence_length])
        y.append(scaled_series[i+sequence_length])
    
    return np.array(X), np.array(y), scaler

# Define the RNN model
class SimpleRNN(nn.Module):
    """Simple RNN model for time series prediction"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Forward pass through RNN
        rnn_out, _ = self.rnn(x, h0)
        
        # Take the last time step output
        out = self.fc(rnn_out[:, -1, :])
        return out

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    """Train the RNN model"""
    model.train()
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_x, batch_y in train_loader:
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return losses

# Main execution
if __name__ == "__main__":
    # Generate time series data
    print("Generating time series data...")
    time_series = generate_time_series(1000)
    
    # Prepare data
    print("Preparing data...")
    sequence_length = 50
    X, y, scaler = prepare_data(time_series, sequence_length)
    
    # Split into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).unsqueeze(2)  # Add feature dimension
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(2)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    print("Initializing RNN model...")
    model = SimpleRNN(
        input_size=1,          # Single feature
        hidden_size=64,        # Number of hidden units
        num_layers=2,          # Number of RNN layers
        output_size=1          # Single output
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training model...")
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs.squeeze(), y_test_tensor)
        print(f'Test Loss: {test_loss:.4f}')
    
    # Make predictions for visualization
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor)
    
    # Inverse transform predictions and actual values
    predictions_np = predictions.numpy().flatten()
    actual_np = y_test
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot original time series
    plt.subplot(2, 2, 2)
    plt.plot(time_series, label='Original Series', alpha=0.7)
    plt.title('Original Time Series')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot test predictions vs actual
    plt.subplot(2, 2, 3)
    plt.plot(actual_np, label='Actual', alpha=0.7)
    plt.plot(predictions_np, label='Predicted', alpha=0.7)
    plt.title('Test Predictions vs Actual')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    
    # Plot zoomed view of predictions
    plt.subplot(2, 2, 4)
    plt.plot(actual_np[:50], label='Actual', alpha=0.7)
    plt.plot(predictions_np[:50], label='Predicted', alpha=0.7)
    plt.title('Zoomed View (First 50 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/run_b3cd94c6a4564a23af97ca20b80d1ab8/assets/rnn_time_series_results.png')
    plt.close()
    
    print("RNN time series prediction completed successfully!")
    print(f"Final test loss: {test_loss:.4f}")