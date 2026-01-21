import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create directory if it doesn't exist
os.makedirs('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets', exist_ok=True)

# Generate synthetic data similar to Chapter 2 examples
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1)
y = 2 * X.ravel() + 1 + np.random.randn(n_samples) * 0.5

# Save data to CSV
data = pd.DataFrame({'X': X.ravel(), 'y': y})
data.to_csv('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/chapter2_data.csv', index=False)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, label='Data points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Linear regression line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/linear_regression_plot.png')
plt.close()

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model on training set
model_train = LinearRegression()
model_train.fit(X_train, y_train)

# Predict on test set
y_test_pred = model_train.predict(X_test)

# Calculate test metrics
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print(f"\nTest Set Metrics:")
print(f"Test MSE: {test_mse:.4f}")
print(f"Test R² Score: {test_r2:.4f}")

# Plot test results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.7, label='Test data')
plt.scatter(X_test, y_test_pred, alpha=0.7, label='Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Test Results')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/linear_regression_test_plot.png')
plt.close()

# Multiple linear regression example
np.random.seed(42)
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)
y_multi = 2*X1 + 3*X2 + np.random.randn(n_samples)*0.5

# Create DataFrame
multi_data = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y_multi})
multi_data.to_csv('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/multi_linear_data.csv', index=False)

# Prepare features and target
X_multi = multi_data[['X1', 'X2']]
y_multi = multi_data['y']

# Fit multiple linear regression
multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)

# Make predictions
y_multi_pred = multi_model.predict(X_multi)

# Calculate metrics
multi_mse = mean_squared_error(y_multi, y_multi_pred)
multi_r2 = r2_score(y_multi, y_multi_pred)

print(f"\nMultiple Linear Regression:")
print(f"MSE: {multi_mse:.4f}")
print(f"R² Score: {multi_r2:.4f}")
print(f"Coefficients: {multi_model.coef_}")
print(f"Intercept: {multi_model.intercept_:.4f}")

# Visualization of multiple regression (using first feature)
plt.figure(figsize=(10, 6))
plt.scatter(X1, y_multi, alpha=0.7, label='Data points')
plt.xlabel('X1')
plt.ylabel('y')
plt.title('Multiple Linear Regression (X1 vs y)')
plt.grid(True, alpha=0.3)
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/multi_linear_plot.png')
plt.close()

print("\nAll Chapter 2 Linear Regression examples completed successfully!")