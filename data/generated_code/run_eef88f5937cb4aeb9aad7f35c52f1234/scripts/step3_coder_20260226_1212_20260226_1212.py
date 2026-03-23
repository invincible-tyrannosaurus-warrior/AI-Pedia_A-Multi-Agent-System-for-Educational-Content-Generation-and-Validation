# Import required libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set the random seed for reproducibility
np.random.seed(42)

# Define the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/run_eef88f5937cb4aeb9aad7f35c52f1234/assets'

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nTarget distribution:")
print(df['species'].value_counts())

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale the features to have zero mean and unit variance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Train the SVM model
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Calculate accuracy
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
plt.savefig(f'{output_dir}/confusion_matrix_svm.png')
plt.close()

# Visualize the relationship between two features
# Using petal length and petal width for visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X[:, 2], X[:, 3], c=y, cmap=plt.cm.RdYlBu, alpha=0.7)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Iris Dataset - Petal Length vs Petal Width')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/iris_petal_features.png')
plt.close()

# Create decision boundary visualization for two features
# We'll use only two features for easier visualization
X_train_2d = X_train_scaled[:, [2, 3]]  # Petal length and petal width
X_test_2d = X_test_scaled[:, [2, 3]]

# Train SVM on these two features
svm_2d = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_2d.fit(X_train_2d, y_train)

# Create a mesh to plot the decision boundary
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
scatter = plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.xlabel('Petal Length (scaled)')
plt.ylabel('Petal Width (scaled)')
plt.title('SVM Decision Boundary (Petal Features)')
plt.colorbar(scatter)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/svm_decision_boundary.png')
plt.close()

# Print model parameters
print("\nSVM Model Parameters:")
print(f"Kernel: {svm_classifier.kernel}")
print(f"C (regularization parameter): {svm_classifier.C}")
print(f"Gamma (kernel coefficient): {svm_classifier.gamma}")

# Save feature importance (support vectors)
print(f"\nNumber of support vectors: {svm_classifier.n_support_}")
print(f"Total support vectors: {len(svm_classifier.support_)}")

# Create a summary report
summary_data = {
    'Metric': ['Accuracy', 'Support Vectors (per class)'],
    'Value': [f"{accuracy:.4f}", f"{svm_classifier.n_support_}"]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(f'{output_dir}/svm_summary.csv', index=False)
print("\nSummary saved to CSV file.")

print("\nAnalysis complete. All visualizations saved to the specified directory.")