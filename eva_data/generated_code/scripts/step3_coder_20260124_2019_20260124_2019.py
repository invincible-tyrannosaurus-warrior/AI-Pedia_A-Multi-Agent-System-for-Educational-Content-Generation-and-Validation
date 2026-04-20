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

# Create sample time series data (sinusoidal pattern with noise)
def generate_time_series(n_samples=1000):
    """Generate synthetic time series data"""
    # Generate time steps
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # Create a complex time series with multiple frequencies
    series = (
        0.5 * np.sin(t) + 
        0.3 * np.sin(3*t) + 
        0.2 * np.sin(5*t) + 
        0.1 * np.random.randn(n_samples)
    )
    
    return series.reshape(-1, 1)

# Prepare data for RNN training
def prepare_data(series, sequence_length=50):
    """Prepare time series data for RNN training"""
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_series = scaler.fit_transform(series)
    
    # Create sequences
    X, y = [], []
    for i in range(len(scaled_series) - sequence_length):
        X.append(scaled_series[i:(i + sequence_length)])
        y.append(scaled_series[i + sequence_length])
    
    return np.array(X), np.array(y), scaler

# Define the RNN model
class TimeSeriesRNN(nn.Module):
    """Simple RNN model for time series prediction"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(TimeSeriesRNN, self).__init__()
        
        # Initialize RNN layers
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer for output
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
            
            # Calculate loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Update weights
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
    data = generate_time_series(1000)
    
    # Prepare data for training
    print("Preparing data for training...")
    sequence_length = 50
    X, y, scaler = prepare_data(data, sequence_length)
    
    # Split into train and test sets
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Initialize model, loss function, and optimizer
    print("Initializing model...")
    model = TimeSeriesRNN(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model
    print("Training model...")
    losses = train_model(model, train_loader, criterion, optimizer, num_epochs=100)
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    # Make predictions for visualization
    model.eval()
    with torch.no_grad():
        # Predict on test data
        predictions = model(X_test_tensor)
        # Inverse transform predictions and actual values
        predictions_np = scaler.inverse_transform(predictions.numpy())
        actual_np = scaler.inverse_transform(y_test)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Plot original data
    plt.subplot(2, 2, 2)
    plt.plot(data, label='Original Data', alpha=0.7)
    plt.title('Original Time Series Data')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot predictions vs actual
    plt.subplot(2, 2, 3)
    plt.plot(actual_np, label='Actual', linewidth=2)
    plt.plot(predictions_np, label='Predicted', linewidth=2)
    plt.title('RNN Predictions vs Actual')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    # Plot residuals
    plt.subplot(2, 2, 4)
    residuals = actual_np.flatten() - predictions_np.flatten()
    plt.plot(residuals, 'o', markersize=2)
    plt.title('Residuals (Actual - Predicted)')
    plt.xlabel('Time')
    plt.ylabel('Residual')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot to specified directory
    plt.savefig('D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/rnn_time_series_results.png')
    plt.close()
    
    print("RNN time series prediction completed successfully!")
    print("Results saved to D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets/rnn_time_series_results.png")