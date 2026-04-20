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

# Define a simple CNN model for demonstration
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

# Function to visualize CNN architecture
def visualize_cnn_architecture():
    """Create a visualization of the CNN architecture"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Define layers positions
    layers = [
        ('Input\n(3x32x32)', (0.1, 0.5)),
        ('Conv1\n(32x32x32)', (0.25, 0.5)),
        ('ReLU', (0.35, 0.5)),
        ('Pool', (0.45, 0.5)),
        ('Conv2\n(64x16x16)', (0.55, 0.5)),
        ('ReLU', (0.65, 0.5)),
        ('Pool', (0.75, 0.5)),
        ('Flatten\n(64x8x8)', (0.85, 0.5)),
        ('FC1\n(512)', (0.95, 0.5))
    ]
    
    # Draw connections between layers
    for i in range(len(layers)-1):
        ax.annotate('', xy=layers[i+1][1], xytext=layers[i][1],
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
    
    # Draw each layer as a rectangle
    for layer_name, pos in layers:
        rect = plt.Rectangle((pos[0]-0.05, pos[1]-0.1), 0.1, 0.2, 
                           facecolor='lightblue', edgecolor='black')
        ax.add_patch(rect)
        ax.text(pos[0], pos[1], layer_name, ha='center', va='center', fontsize=10)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Simple CNN Architecture', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, 'cnn_architecture.png'))
    plt.close()

# Function to train the CNN model
def train_model():
    """Train the CNN on CIFAR-10 dataset"""
    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
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
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)
    
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    epochs = 5
    train_losses = []
    train_accuracies = []
    
    for epoch in range(epochs):
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
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(trainloader)}], Loss: {running_loss/100:.4f}')
                running_loss = 0.0
        
        # Calculate training accuracy for this epoch
        train_accuracy = 100. * correct / total
        train_accuracies.append(train_accuracy)
        train_losses.append(running_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] - Training Accuracy: {train_accuracy:.2f}%')
    
    # Evaluate on test set
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
    
    test_accuracy = 100. * correct / total
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    
    return model, train_accuracies, test_accuracy

# Function to plot training progress
def plot_training_progress(train_accuracies):
    """Plot training accuracy over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, marker='o', color='blue')
    plt.title('Training Accuracy Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.savefig(os.path.join(ASSETS_DIR, 'training_progress.png'))
    plt.close()

# Main execution function
def main():
    """Main function to demonstrate CNN basics"""
    print("Starting CNN Basics Demonstration...")
    
    # Visualize CNN architecture
    print("Creating CNN architecture visualization...")
    visualize_cnn_architecture()
    
    # Train the model
    print("Training CNN model...")
    model, train_accuracies, test_accuracy = train_model()
    
    # Plot training progress
    print("Plotting training progress...")
    plot_training_progress(train_accuracies)
    
    # Print model summary
    print("\nModel Summary:")
    print(model)
    
    # Print some information about CNN components
    print("\nCNN Components Explanation:")
    print("- Convolutional Layers: Extract features from input images using filters")
    print("- ReLU Activation: Introduces non-linearity to the network")
    print("- Pooling Layers: Reduce spatial dimensions while preserving important features")
    print("- Fully Connected Layers: Make final predictions based on extracted features")
    print("- Dropout: Prevents overfitting by randomly setting some neurons to zero")
    
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

if __name__ == "__main__":
    main()