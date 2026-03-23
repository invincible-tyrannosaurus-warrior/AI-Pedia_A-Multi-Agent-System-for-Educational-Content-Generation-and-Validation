# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/run_ea11a8edb41d412a95f5fb3ec3c8e1bd/assets'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Load the iris dataset
iris = load_iris()
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

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier with 100 trees
rf_classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    random_state=42,       # For reproducible results
    max_depth=3,           # Maximum depth of the trees
    min_samples_split=2,   # Minimum samples required to split a node
    min_samples_leaf=1     # Minimum samples required at a leaf node
)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

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
plt.title('Confusion Matrix - Random Forest Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_rf.png'))
plt.close()

# Feature importance analysis
feature_importance = rf_classifier.feature_importances_
feature_names = iris.feature_names

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
bars = plt.bar(importance_df['feature'], importance_df['importance'])
plt.title('Feature Importance in Random Forest Model')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance_rf.png'))
plt.close()

# Show top 3 most important features
top_features = importance_df.head(3)
print(f"\nTop 3 Most Important Features:")
for idx, row in top_features.iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# Demonstrate how changing parameters affects performance
print("\nTesting different numbers of trees:")

# Test different numbers of trees
tree_counts = [10, 50, 100, 200]
accuracies = []

for n_trees in tree_counts:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf_temp.fit(X_train, y_train)
    temp_pred = rf_temp.predict(X_test)
    temp_accuracy = accuracy_score(y_test, temp_pred)
    accuracies.append(temp_accuracy)
    print(f"Trees: {n_trees}, Accuracy: {temp_accuracy:.4f}")

# Plot accuracy vs number of trees
plt.figure(figsize=(10, 6))
plt.plot(tree_counts, accuracies, marker='o')
plt.title('Model Accuracy vs Number of Trees')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_vs_trees.png'))
plt.close()

# Save the trained model's parameters
print(f"\nRandom Forest Model Parameters:")
print(f"- Number of estimators: {rf_classifier.n_estimators}")
print(f"- Max depth: {rf_classifier.max_depth}")
print(f"- Min samples split: {rf_classifier.min_samples_split}")
print(f"- Min samples leaf: {rf_classifier.min_samples_leaf}")

# Summary of what we've done
print("\nSummary:")
print("- Loaded the Iris dataset")
print("- Split data into training and testing sets")
print("- Trained a Random Forest classifier with 100 trees")
print("- Evaluated model performance using accuracy and classification report")
print("- Visualized confusion matrix and feature importance")
print("- Tested how varying number of trees affects performance")
print("- Saved all visualizations to the specified directory")