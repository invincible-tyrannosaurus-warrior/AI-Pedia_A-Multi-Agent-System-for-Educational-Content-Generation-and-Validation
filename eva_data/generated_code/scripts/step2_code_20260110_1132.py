import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set the output directory
output_dir = 'D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets'

# Generate synthetic dataset
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 1) * 10
y = 2.5 * X.flatten() + 3 + np.random.randn(n_samples) * 5

# Create DataFrame for better visualization
df = pd.DataFrame({'feature': X.flatten(), 'target': y})
print("Synthetic Dataset Sample:")
print(df.head())
print(f"\nDataset shape: {df.shape}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate evaluation metrics
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("\nModel Performance:")
print(f"Training RMSE: {train_rmse:.2f}")
print(f"Testing RMSE: {test_rmse:.2f}")
print(f"Training R²: {train_r2:.2f}")
print(f"Testing R²: {test_r2:.2f}")

print(f"\nModel Parameters:")
print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Create visualization
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, alpha=0.7, color='blue', label='Training Data')

# Plot testing data
plt.scatter(X_test, y_test, alpha=0.7, color='red', label='Testing Data')

# Plot the regression line
X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='green', linewidth=2, label='Fitted Line')

# Add labels and title
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Feature vs Target')
plt.legend()

# Save the plot
plt.savefig(f'{output_dir}/linear_regression_plot.png')
plt.close()

print(f"\nPlot saved to: {output_dir}/linear_regression_plot.png")

# Print some predictions
print("\nSample Predictions:")
print("Actual vs Predicted (first 5 test samples):")
for i in range(5):
    print(f"Actual: {y_test[i]:.2f}, Predicted: {y_test_pred[i]:.2f}")