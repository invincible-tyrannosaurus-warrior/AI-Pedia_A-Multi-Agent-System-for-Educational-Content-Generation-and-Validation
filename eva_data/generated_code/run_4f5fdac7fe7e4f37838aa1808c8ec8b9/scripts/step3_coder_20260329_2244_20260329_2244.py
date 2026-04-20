import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

# Create the assets directory if it doesn't exist
assets_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_4f5fdac7fe7e4f37838aa1808c8ec8b9/assets'
os.makedirs(assets_dir, exist_ok=True)

# Load the iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species of iris (setosa, versicolor, virginica)

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset shape:", df.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
# max_depth limits tree depth to prevent overfitting
# min_samples_split sets minimum samples required to split a node
dt_classifier = DecisionTreeClassifier(
    max_depth=3,           # Limit tree depth for better visualization
    min_samples_split=2,   # Minimum samples needed to split a node
    min_samples_leaf=1,    # Minimum samples needed at leaf node
    random_state=42        # For reproducible results
)

# Train the model on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, 
          feature_names=iris.feature_names,
          class_names=iris.target_names,
          filled=True,           # Color nodes by class
          rounded=True,          # Round node corners
          fontsize=10)           # Font size for text

# Save the decision tree plot
plt.savefig(os.path.join(assets_dir, 'decision_tree_visualization.png'))
plt.close()

# Feature importance analysis
feature_importance = dt_classifier.feature_importances_
feature_names = iris.feature_names

# Create a bar chart showing feature importance
plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]  # Sort features by importance

plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance in Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance')

# Save the feature importance plot
plt.savefig(os.path.join(assets_dir, 'feature_importance.png'))
plt.close()

# Show feature importance values
print("\nFeature Importance Rankings:")
for i in range(len(feature_importance)):
    print(f"{feature_names[indices[i]]}: {feature_importance[indices[i]]:.3f}")

# Demonstrate how predictions work with sample data
sample_data = [[5.1, 3.5, 1.4, 0.2],  # Should be setosa
               [6.2, 2.8, 4.8, 1.8],  # Should be versicolor
               [7.2, 3.0, 5.8, 1.6]]  # Should be virginica

sample_predictions = dt_classifier.predict(sample_data)
sample_probabilities = dt_classifier.predict_proba(sample_data)

print("\nSample Predictions:")
for i, (data, pred, prob) in enumerate(zip(sample_data, sample_predictions, sample_probabilities)):
    print(f"Sample {i+1}: {data}")
    print(f"  Predicted class: {iris.target_names[pred]}")
    print(f"  Probabilities: {dict(zip(iris.target_names, prob))}")
    print()

print("Decision tree analysis completed successfully!")
print(f"All visualizations saved to: {assets_dir}")