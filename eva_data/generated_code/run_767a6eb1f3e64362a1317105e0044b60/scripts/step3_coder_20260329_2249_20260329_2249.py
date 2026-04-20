# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_wine
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the output directory for assets
output_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_767a6eb1f3e64362a1317105e0044b60/assets'
os.makedirs(output_dir, exist_ok=True)

# Load the wine dataset
wine_data = load_wine()
X = wine_data.data
y = wine_data.target

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=wine_data.feature_names)
df['target'] = y

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nTarget distribution:")
print(df['target'].value_counts())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create a Random Forest Classifier with 100 trees
rf_classifier = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    random_state=42,       # For reproducible results
    max_depth=5            # Maximum depth of each tree (optional parameter)
)

# Train the model on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate and print the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=wine_data.target_names))

# Create a confusion matrix heatmap
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=wine_data.target_names,
            yticklabels=wine_data.target_names)
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
plt.close()

# Feature importance analysis
feature_importance = rf_classifier.feature_importances_
feature_names = wine_data.feature_names

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

# Plot top 10 most important features
plt.figure(figsize=(10, 6))
top_features = importance_df.head(10)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Most Important Features in Random Forest')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# Show top 5 most important features
print("\nTop 5 Most Important Features:")
print(importance_df.head())

# Demonstrate how changing number of trees affects performance
n_estimators_range = [10, 50, 100, 200, 300]
accuracies = []

for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
    rf_temp.fit(X_train, y_train)
    pred_temp = rf_temp.predict(X_test)
    acc_temp = accuracy_score(y_test, pred_temp)
    accuracies.append(acc_temp)

# Plot accuracy vs number of estimators
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracies, marker='o')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Model Performance vs Number of Trees')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'performance_vs_trees.png'))
plt.close()

print("\nPerformance with different numbers of trees:")
for i, n_est in enumerate(n_estimators_range):
    print(f"Trees: {n_est}, Accuracy: {accuracies[i]:.4f}")

# Save feature importance data to CSV
importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

print(f"\nAll visualizations saved to {output_dir}")