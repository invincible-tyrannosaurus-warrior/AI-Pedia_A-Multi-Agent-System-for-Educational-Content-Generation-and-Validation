import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

# Set the seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the path for saving assets
ASSETS_PATH = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'

# Create the directory if it doesn't exist
os.makedirs(ASSETS_PATH, exist_ok=True)

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # After two pooling operations, 28x28 -> 7x7
        self.fc2 = nn.Linear(128, 10)  # 10 classes for MNIST digits
        
    def forward(self, x):
        # Apply first conv layer + activation + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second conv layer + activation + pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 7 * 7)
        # Apply dropout and first fully connected layer
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        # Apply second fully connected layer
        x = self.fc2(x)
        return x

# Data preprocessing and loading
def load_data():
    # Define transformations for the training and test sets
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST dataset statistics
    ])
    
    # Load MNIST training dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    # Load MNIST test dataset
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# Training function
def train_model(model, train_loader, criterion, optimizer, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device (CPU or GPU)
        data, target = data.to(device), target.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# Testing function
def test_model(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():  # Disable gradient computation for efficiency
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            test_loss += criterion(output, target).item()
            
            # Statistics
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Calculate average loss and accuracy
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

# Main execution function
def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    train_loader, test_loader = load_data()
    
    # Initialize the model
    model = SimpleCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Lists to store metrics for plotting
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    # Training loop
    epochs = 5
    print("Starting training...")
    for epoch in range(epochs):
        # Train the model
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        
        # Test the model
        test_loss, test_acc = test_model(model, test_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        print(f'Epoch [{epoch+1}/{epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        print('-' * 50)
    
    # Plot training and testing metrics
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Testing Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_PATH, 'cnn_training_metrics.png'))
    plt.close()
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Testing Accuracy: {test_accuracies[-1]:.2f}%")
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(ASSETS_PATH, 'cnn_model.pth'))
    print("Model saved to:", os.path.join(ASSETS_PATH, 'cnn_model.pth'))

if __name__ == "__main__":
    main()