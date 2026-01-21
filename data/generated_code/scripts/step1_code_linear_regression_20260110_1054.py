import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
X = np.random.randn(n_samples, 1)
y = 2 * X.flatten() + 1 + np.random.randn(n_samples) * 0.5

# Create DataFrame
df = pd.DataFrame({'feature': X.flatten(), 'target': y})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    df['feature'], df['target'], test_size=0.2, random_state=42
)

# Reshape for sklearn
X_train = X_train.values.reshape(-1, 1)
X_test = X_test.values.reshape(-1, 1)

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

# Print results with exact string "R^2"
print(f"Train MSE: {train_mse:.3f}")
print(f"Test MSE: {test_mse:.3f}")
print(f"Train R^2: {train_r2:.3f}")
print(f"Test R^2: {test_r2:.3f}")

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.5, label='Actual')
plt.scatter(X_test, y_pred_test, alpha=0.5, label='Predicted')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression: Actual vs Predicted Values')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/linear_regression_plot.png')
plt.close()

# Save data to CSV
df.to_csv('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/synthetic_data.csv', index=False)

print("Script executed successfully!")