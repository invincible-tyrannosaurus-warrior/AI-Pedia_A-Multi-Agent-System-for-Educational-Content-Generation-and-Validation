import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the output directory exists
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate a simple dataset
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1) * 10
y = 2.5 * X.flatten() + 3 + np.random.randn(n_samples) * 5

# Create a DataFrame for easier handling
df = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Save the dataset to CSV
df.to_csv(f'{output_dir}/linear_regression_data.csv', index=False)

# Prepare data for modeling
X_train = df[['feature']]
y_train = df['target']

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_train)

# Calculate metrics
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

# Print metrics
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Create visualization of the fitted line vs data
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.7, color='blue', label='Data points')
plt.plot(X_train, y_pred, color='red', linewidth=2, label='Fitted line')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Fitted Line vs Data Points')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/linear_regression_fitted_line.png')
plt.close()

# Create residual plot
residuals = y_train - y_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='green')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/linear_regression_residuals.png')
plt.close()

# Print model parameters
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# Save model metrics to a text file
with open(f'{output_dir}/model_metrics.txt', 'w') as f:
    f.write(f"Linear Regression Model Metrics\n")
    f.write(f"================================\n")
    f.write(f"Mean Squared Error: {mse:.2f}\n")
    f.write(f"Root Mean Squared Error: {rmse:.2f}\n")
    f.write(f"R-squared: {r2:.2f}\n")
    f.write(f"Intercept: {model.intercept_:.2f}\n")
    f.write(f"Coefficient: {model.coef_[0]:.2f}\n")

print("Analysis complete. All outputs saved to:", output_dir)