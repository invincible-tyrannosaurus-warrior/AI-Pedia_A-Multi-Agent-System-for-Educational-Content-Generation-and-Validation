import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# Set the data directory for generated assets
ASSETS_DIR = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'

# Create the assets directory if it doesn't exist
os.makedirs(ASSETS_DIR, exist_ok=True)

# Define a simple CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)  # 10 classes for CIFAR-10
        
    def forward(self, x):
        # Apply first conv layer + ReLU + pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second conv layer + ReLU + pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the tensor for fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        # Apply dropout and fully connected layers
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Data preprocessing and loading
def load_data():
    # Define transformations for training and testing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# Training function
def train_model(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if i % 100 == 99:  # Print every 100 mini-batches
            print(f'Batch {i+1}, Loss: {running_loss/100:.3f}')
            running_loss = 0.0
    
    accuracy = 100. * correct / total
    return accuracy

# Testing function
def test_model(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy

# Visualization of CNN layers
def visualize_filters(model, save_path):
    # Get the weights from the first convolutional layer
    conv_weights = model.conv1.weight.data
    
    # Normalize weights for visualization
    conv_weights = (conv_weights - conv_weights.min()) / (conv_weights.max() - conv_weights.min())
    
    # Create a grid to display filters
    n_filters = min(32, conv_weights.shape[0])  # Show at most 32 filters
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    fig.suptitle('First Convolutional Layer Filters')
    
    for i in range(n_filters):
        row = i // 8
        col = i % 8
        ax = axes[row, col]
        # Plot filter as grayscale image (average over channels)
        filter_img = conv_weights[i].mean(dim=0)
        ax.imshow(filter_img.cpu(), cmap='gray')
        ax.set_title(f'Filter {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Main execution
def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    trainloader, testloader = load_data()
    
    # Initialize model
    model = SimpleCNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 5
    train_accuracies = []
    test_accuracies = []
    
    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        train_acc = train_model(model, trainloader, criterion, optimizer, device)
        test_acc = test_model(model, testloader, device)
        
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f'Train Accuracy: {train_acc:.2f}%, Test Accuracy: {test_acc:.2f}%')
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(ASSETS_DIR, 'cnn_model.pth'))
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('CNN Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(ASSETS_DIR, 'training_history.png'))
    plt.close()
    
    # Visualize filters
    visualize_filters(model, os.path.join(ASSETS_DIR, 'filters_visualization.png'))
    
    print("Training completed successfully!")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")

if __name__ == "__main__":
    main()