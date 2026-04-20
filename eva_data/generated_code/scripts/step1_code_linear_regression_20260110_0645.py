# Linear Regression Demo - Beginner-Friendly Script
# This script demonstrates end-to-end linear regression using synthetic data

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)

# Step 1: Generate synthetic dataset
# Create 100 data points with some noise
n_samples = 100
X = np.random.uniform(0, 10, n_samples)  # Independent variable
# Create dependent variable with linear relationship + noise
y = 2.5 * X + 3 + np.random.normal(0, 2, n_samples)  # y = 2.5x + 3 + noise

# Step 2: Create DataFrame for easier handling
data = pd.DataFrame({'feature': X, 'target': y})

# Step 3: Visualize the data
plt.figure(figsize=(10, 6))
plt.scatter(data['feature'], data['target'], alpha=0.7)
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Scatter Plot of Generated Data')
plt.grid(True, alpha=0.3)
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/scatter_plot.png')
plt.close()

# Step 4: Prepare data for modeling
# Reshape X to be a 2D array (required by sklearn)
X_reshaped = X.reshape(-1, 1)
y_array = y

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_array, test_size=0.2, random_state=42)

# Step 5: Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Extract model parameters
slope = model.coef_[0]  # Slope of the line
intercept = model.intercept_  # Y-intercept

print("Linear Regression Model Parameters:")
print(f"Slope (Coefficient): {slope:.2f}")
print(f"Intercept: {intercept:.2f}")

# Step 7: Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Step 8: Calculate performance metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print("\nModel Performance Metrics:")
print(f"Training MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Testing R²: {test_r2:.2f}")

# Step 9: Visualize the fitted line
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.7, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.7, label='Test Data')

# Plot the regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Fitted Line')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/regression_fit.png')
plt.close()

# Step 10: Make a simple prediction
# Predict the target value for a new feature value
new_feature_value = 7.5
prediction = model.predict([[new_feature_value]])
print(f"\nPrediction for X = {new_feature_value}:")
print(f"Predicted y = {prediction[0]:.2f}")

# Save the dataset to CSV for reference
data.to_csv('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/generated_dataset.csv', index=False)

print("\nAll outputs saved successfully!")