# Import required libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_iris
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/run_eedd396e1ddd4b1bb72920e4dec8dce2/assets'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset shape:", df.shape)
print("\nFirst few rows of the dataset:")
print(df.head())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Create a Random Forest Classifier with 100 trees
rf_classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    random_state=42,       # For reproducible results
    max_depth=3,           # Maximum depth of trees (optional)
    min_samples_split=2,   # Minimum samples required to split a node
    min_samples_leaf=1     # Minimum samples required at a leaf node
)

# Train the model on the training data
print("\nTraining the Random Forest model...")
rf_classifier.fit(X_train, y_train)
print("Model training completed!")

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Confusion Matrix
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

# Demonstrate prediction on new sample
# Using first sample from test set as example
sample_data = X_test[0].reshape(1, -1)
prediction = rf_classifier.predict(sample_data)
prediction_proba = rf_classifier.predict_proba(sample_data)

print(f"\nPrediction for first test sample:")
print(f"Predicted class: {iris.target_names[prediction[0]]}")
print(f"Probabilities: {dict(zip(iris.target_names, prediction_proba[0]))}")

# Test with different number of trees to show impact on performance
tree_counts = [10, 50, 100, 200]
accuracies = []

for n_trees in tree_counts:
    rf_temp = RandomForestClassifier(n_estimators=n_trees, random_state=42)
    rf_temp.fit(X_train, y_train)
    temp_pred = rf_temp.predict(X_test)
    temp_accuracy = accuracy_score(y_test, temp_pred)
    accuracies.append(temp_accuracy)
    print(f"Accuracy with {n_trees} trees: {temp_accuracy:.4f}")

# Plot accuracy vs number of trees
plt.figure(figsize=(10, 6))
plt.plot(tree_counts, accuracies, marker='o')
plt.title('Model Accuracy vs Number of Trees in Random Forest')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'accuracy_vs_trees.png'))
plt.close()

print("\nRandom Forest demonstration completed successfully!")
print(f"All visualizations saved to: {output_dir}")