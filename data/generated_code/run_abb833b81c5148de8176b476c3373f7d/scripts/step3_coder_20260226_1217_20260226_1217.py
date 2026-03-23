# Import required libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/run_abb833b81c5148de8176b476c3373f7d/assets'

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species of iris (setosa, versicolor, virginica)

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget distribution:")
print(df['species'].value_counts())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier with RBF kernel
svm_model = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale')

# Train the model on the scaled training data
svm_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_scaled)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix for SVM Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_svm.png'))
plt.close()

# Visualize decision boundaries for first two features
# Create a mesh to plot the decision boundaries
def plot_decision_boundaries():
    # Use only first two features for visualization
    X_train_2d = X_train_scaled[:, :2]
    X_test_2d = X_test_scaled[:, :2]
    
    # Train SVM with only 2 features
    svm_2d = SVC(kernel='rbf', random_state=42, C=1.0, gamma='scale')
    svm_2d.fit(X_train_2d, y_train)
    
    # Create a mesh
    h = 0.02  # step size in the mesh
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot the training points
    scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    plt.xlabel('Sepal Length (scaled)')
    plt.ylabel('Sepal Width (scaled)')
    plt.title('SVM Decision Boundaries (First Two Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'decision_boundaries_svm.png'))
    plt.close()

# Generate the decision boundary plot
plot_decision_boundaries()

# Show feature importance (coefficients for linear SVM)
# Note: This is only applicable for linear kernel
linear_svm = SVC(kernel='linear', random_state=42)
linear_svm.fit(X_train_scaled, y_train)

# Get support vectors
support_vectors = linear_svm.support_vectors_
print(f"\nNumber of support vectors: {len(support_vectors)}")

# Save model parameters
model_info = {
    'kernel': 'RBF',
    'C': 1.0,
    'gamma': 'scale',
    'accuracy': accuracy,
    'features_used': iris.feature_names[:2]  # First two features for visualization
}

# Create a summary file
summary_text = f"""
SVM Model Summary
=================
Kernel: {model_info['kernel']}
C parameter: {model_info['C']}
Gamma parameter: {model_info['gamma']}
Accuracy: {model_info['accuracy']:.4f}
Features used for visualization: {', '.join(model_info['features_used'])}

Support vectors count: {len(support_vectors)}
"""

with open(os.path.join(output_dir, 'svm_model_summary.txt'), 'w') as f:
    f.write(summary_text)

print("\nAnalysis complete. Files saved to:")
print(f"- Confusion matrix: {os.path.join(output_dir, 'confusion_matrix_svm.png')}")
print(f"- Decision boundaries: {os.path.join(output_dir, 'decision_boundaries_svm.png')}")
print(f"- Model summary: {os.path.join(output_dir, 'svm_model_summary.txt')}")