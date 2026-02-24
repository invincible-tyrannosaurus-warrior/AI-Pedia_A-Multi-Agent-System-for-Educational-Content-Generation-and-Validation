import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the directory exists
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1) * 10  # Feature: random values
y = 2.5 * X.flatten() + 3 + np.random.randn(n_samples) * 5  # Target: linear relationship with noise

# Create a DataFrame for better data handling
data = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    data['feature'], data['target'], test_size=0.2, random_state=42
)

# Reshape the data for scikit-learn (it expects 2D arrays)
X_train_reshaped = X_train.values.reshape(-1, 1)
X_test_reshaped = X_test.values.reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train_reshaped, y_train)

# Make predictions on both training and test data
y_train_pred = model.predict(X_train_reshaped)
y_test_pred = model.predict(X_test_reshaped)

# Calculate performance metrics
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print model parameters and performance metrics
print("Linear Regression Model Parameters:")
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"Test MSE: {test_mse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Test R²: {test_r2:.2f}")

# Create visualization of the results
plt.figure(figsize=(10, 6))

# Plot the training data points
plt.scatter(X_train, y_train, alpha=0.6, color='blue', label='Training data')

# Plot the test data points
plt.scatter(X_test, y_test, alpha=0.6, color='red', label='Test data')

# Plot the regression line using the trained model
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='green', linewidth=2, label='Regression line')

# Add labels and title
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Feature vs Target')
plt.legend()

# Save the plot to the specified directory
plt.savefig(os.path.join(output_dir, 'linear_regression_plot.png'))
plt.close()

# Create a scatter plot showing actual vs predicted values for test set
plt.figure(figsize=(10, 6))

# Plot actual vs predicted values
plt.scatter(y_test, y_test_pred, alpha=0.7, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Test Set)')

# Save the second plot
plt.savefig(os.path.join(output_dir, 'actual_vs_predicted_plot.png'))
plt.close()

# Print summary of what we've done
print("\nSummary:")
print("- Generated synthetic data with linear relationship plus noise")
print("- Split data into training (80%) and test (20%) sets")
print("- Trained a linear regression model")
print("- Evaluated model performance using MSE and R² metrics")
print("- Created visualizations showing the regression line and predictions")
print(f"- Plots saved to: {output_dir}")