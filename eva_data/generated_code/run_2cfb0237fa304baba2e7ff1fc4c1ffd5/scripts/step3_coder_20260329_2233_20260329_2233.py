import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create the assets directory if it doesn't exist
assets_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_2cfb0237fa304baba2e7ff1fc4c1ffd5/assets'
os.makedirs(assets_dir, exist_ok=True)

# Generate synthetic data for demonstration
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1) * 10  # Feature: random values
y = 2.5 * X.flatten() + 3 + np.random.randn(n_samples) * 5  # Target: linear relationship with noise

# Create a DataFrame for easier handling
data = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    data['feature'], data['target'], test_size=0.2, random_state=42
)

# Reshape the data for sklearn (it expects 2D arrays)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate metrics to evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model parameters and performance metrics
print(f"Linear Regression Model Parameters:")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Create visualization of the results
plt.figure(figsize=(10, 6))

# Plot the original data points
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual Data')

# Plot the regression line
X_line = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')

# Add labels and title
plt.xlabel('Feature Value')
plt.ylabel('Target Value')
plt.title('Linear Regression: Feature vs Target')
plt.legend()

# Save the plot
plt.savefig(os.path.join(assets_dir, 'linear_regression_plot.png'))
plt.close()

# Create a scatter plot showing predicted vs actual values
plt.figure(figsize=(10, 6))

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values in Linear Regression')

# Save the second plot
plt.savefig(os.path.join(assets_dir, 'actual_vs_predicted_plot.png'))
plt.close()

# Print summary information
print("\nSummary:")
print("-" * 40)
print("This example demonstrates linear regression using scikit-learn.")
print("We generated synthetic data with a known linear relationship plus noise.")
print("The model learns the relationship between feature and target variables.")
print("Performance is evaluated using Mean Squared Error and R² Score.")

# Save the dataset to CSV for reference
data.to_csv(os.path.join(assets_dir, 'generated_dataset.csv'), index=False)
print(f"\nDataset saved to: {os.path.join(assets_dir, 'generated_dataset.csv')}")