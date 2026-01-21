# Import required libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Set the data directory
data_dir = 'D:/L3/Individual_project/AI_Pedia_Local/archive/data/generated_code/assets'

# Generate synthetic dataset
np.random.seed(42)  # For reproducible results
n_samples = 100
X = np.random.randn(n_samples, 1) * 10  # Feature: random values
y = 2.5 * X.flatten() + 3 + np.random.randn(n_samples) * 5  # Target: linear relationship with noise

# Create DataFrame
df = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Save dataset to CSV
df.to_csv(f'{data_dir}/linear_regression_data.csv', index=False)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df['feature'], df['target'], test_size=0.2, random_state=42
)

# Reshape features for sklearn (needs 2D array)
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on test set
y_pred = model.predict(X_test)

# Calculate model performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print model parameters and performance
print("Linear Regression Model Results:")
print(f"Coefficient (slope): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.2f}")

# Create visualization of the regression line
plt.figure(figsize=(10, 6))

# Plot the original data points
plt.scatter(df['feature'], df['target'], alpha=0.6, color='blue', label='Data Points')

# Plot the regression line
X_line = np.linspace(df['feature'].min(), df['feature'].max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color='red', linewidth=2, label='Regression Line')

# Add labels and title
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Feature vs Target')
plt.legend()

# Save the plot
plt.savefig(f'{data_dir}/linear_regression_plot.png')
plt.close()

# Create a scatter plot of actual vs predicted values
plt.figure(figsize=(10, 6))

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Add labels and title
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values in Linear Regression')

# Save the second plot
plt.savefig(f'{data_dir}/actual_vs_predicted_plot.png')
plt.close()

print("Linear regression analysis completed successfully.")
print(f"Results saved to: {data_dir}")