import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create the assets directory if it doesn't exist
assets_dir = 'D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets'
os.makedirs(assets_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data for linear regression example
n_samples = 100
X = np.random.randn(n_samples, 1)
y = 2 * X.flatten() + 1 + np.random.randn(n_samples) * 0.5

# Create DataFrame
df = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Save the dataset
df.to_csv(f'{assets_dir}/linear_regression_data.csv', index=False)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print results
print("Linear Regression Results:")
print(f"Training MSE: {train_mse:.4f}")
print(f"Test MSE: {test_mse:.4f}")
print(f"Training R²: {train_r2:.4f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.7, label='Training Data')
plt.scatter(X_test, y_test, alpha=0.7, label='Test Data')
plt.plot(X_train, y_pred_train, color='red', linewidth=2, label='Regression Line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Feature vs Target')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{assets_dir}/linear_regression_plot.png')
plt.close()

# Multiple linear regression example
print("\nMultiple Linear Regression Example:")

# Generate more complex synthetic data
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)
y = 2*X1 + 3*X2 + np.random.randn(n_samples)*0.5

# Create DataFrame
df_multi = pd.DataFrame({
    'feature1': X1,
    'feature2': X2,
    'target': y
})

# Save the dataset
df_multi.to_csv(f'{assets_dir}/multiple_linear_regression_data.csv', index=False)

# Prepare features and target
X_multi = df_multi[['feature1', 'feature2']]
y_multi = df_multi['target']

# Split the data
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42
)

# Fit the model
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Make predictions
y_pred_train_multi = model_multi.predict(X_train_multi)
y_pred_test_multi = model_multi.predict(X_test_multi)

# Calculate metrics
train_mse_multi = mean_squared_error(y_train_multi, y_pred_train_multi)
test_mse_multi = mean_squared_error(y_test_multi, y_pred_test_multi)
train_r2_multi = r2_score(y_train_multi, y_pred_train_multi)
test_r2_multi = r2_score(y_test_multi, y_pred_test_multi)

# Print results
print("Multiple Linear Regression Results:")
print(f"Training MSE: {train_mse_multi:.4f}")
print(f"Test MSE: {test_mse_multi:.4f}")
print(f"Training R²: {train_r2_multi:.4f}")
print(f"Test R²: {test_r2_multi:.4f}")
print(f"Coefficient 1: {model_multi.coef_[0]:.4f}")
print(f"Coefficient 2: {model_multi.coef_[1]:.4f}")
print(f"Intercept: {model_multi.intercept_:.4f}")

# Create scatter plot of predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_multi, y_pred_test_multi, alpha=0.7)
plt.plot([y_test_multi.min(), y_test_multi.max()], [y_test_multi.min(), y_test_multi.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.grid(True, alpha=0.3)
plt.savefig(f'{assets_dir}/multiple_linear_regression_plot.png')
plt.close()

# Simple polynomial regression example
print("\nPolynomial Regression Example:")

# Generate data with non-linear relationship
np.random.seed(42)
n_samples = 100
X_poly = np.linspace(0, 10, n_samples)
y_poly = 0.1 * X_poly**2 + 0.5 * X_poly + 2 + np.random.randn(n_samples) * 2

# Create DataFrame
df_poly = pd.DataFrame({'feature': X_poly, 'target': y_poly})

# Save the dataset
df_poly.to_csv(f'{assets_dir}/polynomial_regression_data.csv', index=False)

# Create polynomial features (degree 2)
X_poly_features = np.column_stack([X_poly, X_poly**2])

# Split the data
X_train_poly, X_test_poly, y_train_poly, y_test_poly = train_test_split(
    X_poly_features, y_poly, test_size=0.2, random_state=42
)

# Fit the model
model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train_poly)

# Make predictions
y_pred_train_poly = model_poly.predict(X_train_poly)
y_pred_test_poly = model_poly.predict(X_test_poly)

# Calculate metrics
train_mse_poly = mean_squared_error(y_train_poly, y_pred_train_poly)
test_mse_poly = mean_squared_error(y_test_poly, y_pred_test_poly)
train_r2_poly = r2_score(y_train_poly, y_pred_train_poly)
test_r2_poly = r2_score(y_test_poly, y_pred_test_poly)

# Print results
print("Polynomial Regression Results:")
print(f"Training MSE: {train_mse_poly:.4f}")
print(f"Test MSE: {test_mse_poly:.4f}")
print(f"Training R²: {train_r2_poly:.4f}")
print(f"Test R²: {test_r2_poly:.4f}")
print(f"Coefficient (x): {model_poly.coef_[0]:.4f}")
print(f"Coefficient (x^2): {model_poly.coef_[1]:.4f}")
print(f"Intercept: {model_poly.intercept_:.4f}")

# Plot polynomial regression results
plt.figure(figsize=(10, 6))
X_plot = np.linspace(0, 10, 100)
X_plot_features = np.column_stack([X_plot, X_plot**2])
y_plot = model_poly.predict(X_plot_features)

plt.scatter(X_poly, y_poly, alpha=0.7, label='Data Points')
plt.plot(X_plot, y_plot, color='red', linewidth=2, label='Polynomial Fit')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Polynomial Regression: Quadratic Relationship')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{assets_dir}/polynomial_regression_plot.png')
plt.close()

print("\nAll code examples executed successfully!")
print("Generated files saved to:", assets_dir)