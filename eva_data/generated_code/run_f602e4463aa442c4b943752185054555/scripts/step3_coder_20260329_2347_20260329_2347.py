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

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create the assets directory if it doesn't exist
assets_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_f602e4463aa442c4b943752185054555/assets'
os.makedirs(assets_dir, exist_ok=True)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Forward pass through first conv block
        x = self.pool1(self.relu1(self.conv1(x)))
        
        # Forward pass through second conv block
        x = self.pool2(self.relu2(self.conv2(x)))
        
        # Forward pass through third conv block
        x = self.pool3(self.relu3(self.conv3(x)))
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Data preprocessing and loading
def load_data():
    # Define transformations for training and testing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Load CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='/tmp', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='/tmp', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    
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
        
        # Backward pass and optimize
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
def test_model(model, testloader, criterion, device):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy, test_loss / len(testloader)

# Main execution
def main():
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    trainloader, testloader = load_data()
    
    # Initialize model
    model = SimpleCNN(num_classes=10)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 10
    train_accuracies = []
    test_accuracies = []
    train_losses = []
    test_losses = []
    
    print("Starting training...")
    
    # Training loop
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Train
        train_acc = train_model(model, trainloader, criterion, optimizer, device)
        train_accuracies.append(train_acc)
        
        # Test
        test_acc, test_loss = test_model(model, testloader, criterion, device)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        
        print(f'Train Accuracy: {train_acc:.2f}%')
        print(f'Test Accuracy: {test_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), test_accuracies, label='Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(assets_dir, 'training_history.png'))
    plt.close()
    
    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(assets_dir, 'cnn_model.pth'))
    
    print("\nTraining completed!")
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Testing Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Model saved to {os.path.join(assets_dir, 'cnn_model.pth')}")

if __name__ == "__main__":
    main()