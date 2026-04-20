import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data - using iris dataset for demonstration
iris = datasets.load_iris()
X = iris.data[:, :2]  # Use only first two features for visualization
y = iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features for better SVM performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SVM classifier with RBF kernel
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = svm_classifier.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create a confusion matrix
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
plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_d24a844f7f9a4f5885b2ee23f3a50867/assets/confusion_matrix.png')
plt.close()

# Visualize the decision boundary
def plot_decision_boundary(X, y, model, scaler):
    """
    Plot the decision boundary of the SVM classifier
    """
    h = 0.02  # Step size in the mesh
    # Create a mesh to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Make predictions on the mesh points
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # Plot the data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.title('SVM Decision Boundary')
    plt.colorbar(scatter)
    plt.tight_layout()

# Plot decision boundary using scaled data
plot_decision_boundary(X_train_scaled, y_train, svm_classifier, scaler)
plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_d24a844f7f9a4f5885b2ee23f3a50867/assets/decision_boundary.png')
plt.close()

# Show some sample predictions
print("\nSample Predictions:")
for i in range(5):
    print(f"True: {iris.target_names[y_test[i]]}, Predicted: {iris.target_names[y_pred[i]]}")

# Save model parameters
print(f"\nModel Parameters:")
print(f"Kernel: {svm_classifier.kernel}")
print(f"C (regularization parameter): {svm_classifier.C}")
print(f"Gamma: {svm_classifier.gamma}")

# Create a simple summary report
summary_data = {
    'Metric': ['Accuracy', 'Support Vector Count'],
    'Value': [f'{accuracy:.2f}', len(svm_classifier.support_vectors_)]
}

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_d24a844f7f9a4f5885b2ee23f3a50867/assets/svm_summary.csv', index=False)

print("\nSummary saved to CSV file.")