# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets'

# Generate synthetic dataset
np.random.seed(42)
X = np.random.randn(100, 1) * 10  # Generate 100 samples with 1 feature
y = 2.5 * X.flatten() + 10 + np.random.randn(100) * 5  # Linear relationship with noise

# Create a DataFrame for better visualization
data = pd.DataFrame({'feature': X.flatten(), 'target': y})
print("Dataset head:")
print(data.head())
print(f"\nDataset shape: {data.shape}")

# Save the dataset to CSV
data.to_csv(f'{output_dir}/synthetic_data.csv', index=False)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nModel Performance:")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.2f}")

# Print model parameters
print(f"\nModel Parameters:")
print(f"Slope (coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Create visualizations
# Plot 1: Data points and fitted line
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.6, label='Actual data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Fitted line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/linear_regression_plot.png')
plt.close()

# Plot 2: Predicted vs Actual values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/predicted_vs_actual.png')
plt.close()

print(f"\nPlots saved to: {output_dir}")