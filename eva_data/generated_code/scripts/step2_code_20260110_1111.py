import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Set the random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000
size = np.random.normal(100, 20, n_samples)
price = 5000 + 100 * size + np.random.normal(0, 500, n_samples)

# Create DataFrame
data = pd.DataFrame({
    'size': size,
    'price': price
})

# Split the data into training and testing sets
X = data[['size']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R^2 scores
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print metrics with literal 'R^2'
print(f"R^2 Score (training): {r2_train:.3f}")
print(f"R^2 Score (testing): {r2_test:.3f}")

# Create line points for plotting using DataFrame with correct column name
x_line = pd.DataFrame({'size': np.linspace(X['size'].min(), X['size'].max(), 100)})
y_line = model.predict(x_line)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(X_train['size'], y_train, alpha=0.5, label='Training data')
plt.scatter(X_test['size'], y_test, alpha=0.5, label='Testing data')
plt.plot(x_line['size'], y_line, color='red', linewidth=2, label='Regression line')
plt.xlabel('Size')
plt.ylabel('Price')
plt.title('Linear Regression: Price vs Size')
plt.legend()
plt.grid(True, alpha=0.3)

# Save plot to specified directory
plt.savefig('D:/L3/Individual_project/AI_Pedia_Local/data/generated_code/assets/linear_regression_plot.png')
plt.close()

# Print model parameters
print(f"Slope: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")