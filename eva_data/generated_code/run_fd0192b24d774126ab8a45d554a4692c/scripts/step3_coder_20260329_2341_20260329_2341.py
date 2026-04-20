import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Create a synthetic dataset for demonstration
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                          n_redundant=10, n_clusters_per_class=1, 
                          random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.LongTensor(y_test)

# Define the MLP model architecture
class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(MLP, self).__init__()
        
        # Create a list of layers
        layers = []
        
        # First hidden layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.2))  # Dropout for regularization
        
        # Additional hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        
        # Create the sequential model
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Initialize the model
input_size = X_train.shape[1]  # Number of features
hidden_sizes = [64, 32, 16]    # Hidden layer sizes
num_classes = len(np.unique(y)) # Number of classes

model = MLP(input_size, hidden_sizes, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Training parameters
num_epochs = 100
train_losses = []
train_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_train, predicted.numpy())
    
    # Store metrics
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    
    # Print progress every 20 epochs
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# Evaluation on test set
model.eval()  # Set model to evaluation mode
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    _, predicted = torch.max(test_outputs.data, 1)
    test_accuracy = accuracy_score(y_test, predicted.numpy())

print(f'\nTest Accuracy: {test_accuracy:.4f}')

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_test, predicted.numpy()))

# Plot training loss and accuracy
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot training loss
ax1.plot(train_losses)
ax1.set_title('Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.grid(True)

# Plot training accuracy
ax2.plot(train_accuracies)
ax2.set_title('Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.grid(True)

plt.tight_layout()
plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_fd0192b24d774126ab8a45d554a4692c/assets/mlp_training_metrics.png')
plt.close()

# Plot decision boundary for first two features (for visualization purposes)
def plot_decision_boundary():
    # Create a mesh
    h = 0.02
    x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
    y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    mesh_points = np.c_[xx.ravel(), yy.ravel()] 
    # Pad with zeros to match input size
    mesh_points = np.pad(mesh_points, ((0, 0), (0, X_train_scaled.shape[1]-2)), 'constant')
    
    # Convert to tensor and predict
    mesh_tensor = torch.FloatTensor(mesh_points)
    with torch.no_grad():
        Z = model(mesh_tensor)
        Z = torch.argmax(Z, dim=1).numpy()
    
    # Reshape and plot
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title('MLP Decision Boundary (First Two Features)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_fd0192b24d774126ab8a45d554a4692c/assets/mlp_decision_boundary.png')
    plt.close()

# Create decision boundary plot
plot_decision_boundary()

# Print model architecture summary
print("\nModel Architecture:")
print(model)
print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters())}")