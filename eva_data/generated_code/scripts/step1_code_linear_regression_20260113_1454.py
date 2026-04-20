# Linear Regression Teaching Script
# This script demonstrates linear regression using synthetic data
# Dependencies: numpy, matplotlib, scikit-learn

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic dataset
# We'll create 100 data points with some noise around a linear relationship
np.random.seed(42)  # For reproducible results
n_samples = 100
X = np.random.randn(n_samples, 1) * 10  # Feature values
true_slope = 2.5
true_intercept = 5
y = true_slope * X.ravel() + true_intercept + np.random.randn(n_samples) * 5  # Target values with noise

# Reshape X for sklearn (needs 2D array)
X = X.reshape(-1, 1)

# Split data into training and testing sets
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate key metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model parameters and metrics
print("Linear Regression Model Parameters:")
print(f"Estimated slope: {model.coef_[0]:.2f}")
print(f"Estimated intercept: {model.intercept_:.2f}")
print(f"True slope: {true_slope}")
print(f"True intercept: {true_intercept}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Create visualization
plt.figure(figsize=(10, 6))

# Plot the training data
plt.scatter(X_train, y_train, alpha=0.7, color='blue', label='Training data')

# Plot the test data
plt.scatter(X_test, y_test, alpha=0.7, color='green', label='Test data')

# Plot the regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression line')

# Add labels and title
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Linear Regression: Synthetic Data Fit')
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig(os.path.join(output_dir, 'linear_regression_plot.png'))
plt.close()

# Create a second plot showing residuals
plt.figure(figsize=(10, 6))

# Calculate residuals
residuals = y_test - y_pred

# Plot residuals vs predicted values
plt.scatter(y_pred, residuals, alpha=0.7, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot: Checking Model Assumptions')
plt.grid(True, alpha=0.3)

# Save the residual plot
plt.savefig(os.path.join(output_dir, 'linear_regression_residuals.png'))
plt.close()

# Print summary of what we've learned
print("\n--- Summary ---")
print("This script demonstrates linear regression concepts:")
print("1. Data generation with known underlying relationship")
print("2. Model fitting using scikit-learn")
print("3. Evaluation using MSE and R² metrics")
print("4. Visualization of the fitted line and residuals")
print("5. The model learns the relationship between X and y")

# Save the dataset to CSV for reference
dataset = np.column_stack([X.ravel(), y])
np.savetxt(os.path.join(output_dir, 'synthetic_dataset.csv'), dataset, 
           delimiter=',', header='feature,target', comments='')
print(f"\nDataset saved to {os.path.join(output_dir, 'synthetic_dataset.csv')}")

print("\nPlots saved to:", output_dir)