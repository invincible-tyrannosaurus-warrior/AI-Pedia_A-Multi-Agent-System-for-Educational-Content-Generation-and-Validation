import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print results
print("Model Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Create confusion matrix plot
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/confusion_matrix.png')
plt.close()

# Feature importance plot
feature_importance = model.feature_importances_
feature_names = iris.feature_names

plt.figure(figsize=(10, 6))
indices = np.argsort(feature_importance)[::-1]
plt.bar(range(len(feature_importance)), feature_importance[indices])
plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=45)
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/feature_importance.png')
plt.close()

# Save results to CSV
results_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Correct': y_test == y_pred
})
results_df.to_csv('D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/predictions.csv', index=False)

print("\nResults saved to:")
print("- Confusion matrix: D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/confusion_matrix.png")
print("- Feature importance: D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/feature_importance.png")
print("- Predictions: D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets/predictions.csv")