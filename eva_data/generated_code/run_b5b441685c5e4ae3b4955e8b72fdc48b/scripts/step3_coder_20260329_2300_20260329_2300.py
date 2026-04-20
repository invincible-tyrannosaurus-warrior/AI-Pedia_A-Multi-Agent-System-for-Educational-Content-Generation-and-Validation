import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate a synthetic dataset for demonstration
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, 
                          n_classes=3, random_state=42)

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create KNN classifier with k=5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model using the training data
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

# Print model performance metrics
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Create a visualization of the dataset and decision boundaries
def plot_decision_boundaries():
    # Create a mesh to plot the decision boundaries
    h = 0.02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    
    # Plot the training points
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                         cmap=plt.cm.RdYlBu, edgecolors='black')
    
    # Add labels and title
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Decision Boundaries (k=5)')
    plt.colorbar(scatter)
    
    # Save the plot to the specified directory
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_b5b441685c5e4ae3b4955e8b72fdc48b/assets/knn_decision_boundaries.png')
    plt.close()

# Create the decision boundary plot
plot_decision_boundaries()

# Test different values of k to find the optimal number of neighbors
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train, y_train)
    y_pred_temp = knn_temp.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred_temp))

# Plot accuracy vs k values
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='blue')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy vs Number of Neighbors')
plt.grid(True)

# Mark the best k value
best_k = k_values[np.argmax(accuracies)]
best_accuracy = max(accuracies)
plt.axvline(x=best_k, color='red', linestyle='--', alpha=0.7, 
           label=f'Best k = {best_k} (Accuracy: {best_accuracy:.4f})')
plt.legend()

# Save the plot
plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_b5b441685c5e4ae3b4955e8b72fdc48b/assets/knn_accuracy_vs_k.png')
plt.close()

# Create a summary DataFrame for results
results_df = pd.DataFrame({
    'k': k_values,
    'accuracy': accuracies
})

# Save results to CSV
results_df.to_csv('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_b5b441685c5e4ae3b4955e8b72fdc48b/assets/knn_results.csv', index=False)

print("Analysis completed. Results saved to assets directory.")