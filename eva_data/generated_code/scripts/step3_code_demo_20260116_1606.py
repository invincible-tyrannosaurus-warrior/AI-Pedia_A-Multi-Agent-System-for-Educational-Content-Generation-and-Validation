import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# Ensure the output directory exists
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Load the MNIST dataset from OpenML
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)

# Use a subset of data for faster execution
X_subset = X[:10000]
y_subset = y[:10000]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# Train a Random Forest classifier
print("Training classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.title('Confusion Matrix - MNIST Dataset')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save the plot
plot_path = os.path.join(output_dir, 'confusion_matrix.png')
plt.savefig(plot_path)
plt.close()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nConfusion matrix saved to: {plot_path}")