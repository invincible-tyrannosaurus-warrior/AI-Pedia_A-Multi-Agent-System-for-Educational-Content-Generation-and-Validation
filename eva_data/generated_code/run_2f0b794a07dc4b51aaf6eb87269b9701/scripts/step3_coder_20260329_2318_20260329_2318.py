# Import required libraries
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the output directory
output_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_2f0b794a07dc4b51aaf6eb87269b9701/assets'
os.makedirs(output_dir, exist_ok=True)

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species of iris (setosa, versicolor, virginica)

# Create a DataFrame for better visualization
df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display basic information about the dataset
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nSpecies distribution:")
print(df['species_name'].value_counts())

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create and train the Naive Bayes classifier
# We'll use Gaussian Naive Bayes since features are continuous
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of Naive Bayes classifier: {accuracy:.4f}")

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
plt.title('Confusion Matrix - Naive Bayes Classifier')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'confusion_matrix_naive_bayes.png'))
plt.close()

# Visualize feature distributions by class
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, feature in enumerate(iris.feature_names):
    for species in range(3):
        data = df[df['species'] == species][feature]
        axes[i].hist(data, alpha=0.7, label=iris.target_names[species], bins=15)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('Frequency')
    axes[i].set_title(f'Distribution of {feature} by Species')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
plt.close()

# Show some sample predictions
print("\nSample Predictions vs Actual:")
for i in range(10):
    actual = iris.target_names[y_test[i]]
    predicted = iris.target_names[y_pred[i]]
    print(f"Actual: {actual:<12} Predicted: {predicted}")

# Demonstrate how Naive Bayes works with a simple example
print("\n--- How Naive Bayes Works ---")
print("Naive Bayes applies Bayes' theorem with the assumption of independence between features.")
print("For a given sample, it calculates P(class|features) ∝ P(features|class) × P(class)")
print("Then assigns the class with the highest probability.")

# Show feature means for each class
print("\nFeature means by class:")
feature_means = df.groupby('species')[iris.feature_names].mean()
print(feature_means)

# Save feature means to CSV
feature_means.to_csv(os.path.join(output_dir, 'feature_means_by_class.csv'))

# Print model parameters
print("\nModel Parameters:")
print("Class priors (P(class)):")
for i, class_prior in enumerate(nb_classifier.class_prior_):
    print(f"  {iris.target_names[i]}: {class_prior:.4f}")

print("\nFeature variances for each class:")
for i, class_var in enumerate(nb_classifier.var_):
    print(f"  {iris.target_names[i]}: {class_var}")

print("\n--- End of Naive Bayes Demo ---")