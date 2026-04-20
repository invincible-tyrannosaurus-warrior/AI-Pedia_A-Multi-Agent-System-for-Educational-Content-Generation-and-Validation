import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import os

# Set the output directory
output_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_6582e3da972d4038a9a2939718382054/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate a synthetic dataset for binary classification
X, y = make_classification(
    n_samples=1000,      # Number of samples
    n_features=2,        # Number of features
    n_redundant=0,       # No redundant features
    n_informative=2,     # Number of informative features
    n_clusters_per_class=1,
    random_state=42      # For reproducibility
)

# Create a DataFrame for easier handling
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['target'] = y

# Display first few rows of the dataset
print("Dataset sample:")
print(df.head())

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
logistic_model = LogisticRegression(random_state=42)
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logistic_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Visualize the decision boundary
def plot_decision_boundary():
    # Create a mesh to plot the decision boundary
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    Z = logistic_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.colorbar(scatter)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.savefig(os.path.join(output_dir, 'decision_boundary.png'))
    plt.close()

# Call the function to plot decision boundary
plot_decision_boundary()

# Show model coefficients
print("\nModel Coefficients:")
print(f"Intercept: {logistic_model.intercept_[0]:.4f}")
print(f"Coefficients: {logistic_model.coef_[0]}")

# Save the dataset to CSV
df.to_csv(os.path.join(output_dir, 'logistic_regression_dataset.csv'), index=False)

print(f"\nAll outputs saved to {output_dir}")