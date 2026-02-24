import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Ensure the assets directory exists
assets_dir = 'D:/L3/Individual_project/AI_Pedia_Local_stream/data/generated_code/assets'
os.makedirs(assets_dir, exist_ok=True)

# Chapter 2: Linear Regression - Complete Lesson

print("=== Chapter 2: Linear Regression ===\n")

# Section 1: Introduction to Linear Regression
print("Section 1: Introduction to Linear Regression")
print("-" * 50)
print("Linear regression is a statistical method used to model the relationship between a dependent variable and one or more independent variables.")
print("It assumes a linear relationship between variables.")

# Create sample data for demonstration
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2.5 * X + 3 + np.random.normal(0, 2, 50)  # y = 2.5x + 3 + noise

# Plot the data
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, color='blue')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Sample Data for Linear Regression')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(assets_dir, 'linear_regression_sample_data.png'))
plt.close()

print("Created scatter plot of sample data with linear relationship\n")

# Section 2: Simple Linear Regression Model
print("Section 2: Simple Linear Regression Model")
print("-" * 50)
print("Simple linear regression uses one independent variable to predict the dependent variable.")
print("The model takes the form: y = β₀ + β₁x + ε")

# Reshape X for sklearn (needs 2D array)
X_reshaped = X.reshape(-1, 1)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_reshaped, y)

# Get model parameters
intercept = model.intercept_
slope = model.coef_[0]

print(f"Intercept (β₀): {intercept:.2f}")
print(f"Slope (β₁): {slope:.2f}")
print(f"Equation: y = {slope:.2f}x + {intercept:.2f}")

# Make predictions
y_pred = model.predict(X_reshaped)

# Plot the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.7, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Simple Linear Regression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(assets_dir, 'simple_linear_regression.png'))
plt.close()

print("Created plot showing simple linear regression line\n")

# Section 3: Model Evaluation Metrics
print("Section 3: Model Evaluation Metrics")
print("-" * 50)
print("We evaluate linear regression models using metrics like MSE, RMSE, and R².")

# Calculate evaluation metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")

# Interpretation of R²
print(f"\nR² interpretation: The model explains {r2*100:.1f}% of the variance in the dependent variable.")

# Section 4: Multiple Linear Regression
print("\nSection 4: Multiple Linear Regression")
print("-" * 50)
print("Multiple linear regression uses multiple independent variables to predict the dependent variable.")
print("The model takes the form: y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε")

# Generate sample data with two features
np.random.seed(123)
n_samples = 100
X1 = np.random.randn(n_samples)
X2 = np.random.randn(n_samples)
y_multi = 2*X1 + 3*X2 + np.random.randn(n_samples)*0.5  # y = 2x1 + 3x2 + noise

# Combine features into a matrix
X_multi = np.column_stack([X1, X2])

# Fit multiple linear regression model
multi_model = LinearRegression()
multi_model.fit(X_multi, y_multi)

# Get coefficients
intercept_multi = multi_model.intercept_
coefficients_multi = multi_model.coef_

print(f"Intercept (β₀): {intercept_multi:.2f}")
print(f"Coefficient for x1 (β₁): {coefficients_multi[0]:.2f}")
print(f"Coefficient for x2 (β₂): {coefficients_multi[1]:.2f}")
print(f"Equation: y = {intercept_multi:.2f} + {coefficients_multi[0]:.2f}x₁ + {coefficients_multi[1]:.2f}x₂")

# Predictions
y_pred_multi = multi_model.predict(X_multi)

# Calculate metrics for multiple regression
mse_multi = mean_squared_error(y_multi, y_pred_multi)
rmse_multi = np.sqrt(mse_multi)
r2_multi = r2_score(y_multi, y_pred_multi)

print(f"\nMultiple Regression Metrics:")
print(f"MSE: {mse_multi:.2f}")
print(f"RMSE: {rmse_multi:.2f}")
print(f"R²: {r2_multi:.4f}")

# Section 5: Train-Test Split
print("\nSection 5: Train-Test Split")
print("-" * 50)
print("To properly evaluate our model, we split data into training and testing sets.")

# Create larger dataset for train-test split
np.random.seed(456)
X_large = np.linspace(0, 20, 100)
y_large = 1.5 * X_large + 2 + np.random.normal(0, 3, 100)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_large.reshape(-1, 1), y_large, test_size=0.2, random_state=42
)

# Fit model on training data
train_model = LinearRegression()
train_model.fit(X_train, y_train)

# Make predictions on test set
y_pred_test = train_model.predict(X_test)

# Evaluate on test set
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print(f"Test Set Metrics:")
print(f"MSE: {mse_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"R²: {r2_test:.4f}")

# Visualize train-test split
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.7, color='blue', label='Training data')
plt.scatter(X_test, y_test, alpha=0.7, color='green', label='Testing data')
plt.plot(X_large, train_model.predict(X_large.reshape(-1, 1)), color='red', linewidth=2, label='Regression line')
plt.xlabel('X (Independent Variable)')
plt.ylabel('y (Dependent Variable)')
plt.title('Train-Test Split Visualization')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(assets_dir, 'train_test_split.png'))
plt.close()

print("Created visualization showing train-test split\n")

# Section 6: Assumptions of Linear Regression
print("\nSection 6: Assumptions of Linear Regression")
print("-" * 50)
print("Linear regression makes several key assumptions:")
print("1. Linearity: Relationship between variables is linear")
print("2. Independence: Observations are independent")
print("3. Homoscedasticity: Constant variance of residuals")
print("4. Normality: Residuals are normally distributed")

# Check residuals
residuals = y - y_pred

# Plot residuals vs fitted values
plt.figure(figsize=(10, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='purple')
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(assets_dir, 'residuals_plot.png'))
plt.close()

print("Created residual plot to check homoscedasticity assumption\n")

# Section 7: Making Predictions
print("\nSection 7: Making Predictions")
print("-" * 50)
print("Once we have a trained model, we can make predictions on new data.")

# Make predictions for new values
new_X = np.array([[5], [7], [10], [15]])
predictions = train_model.predict(new_X)

print("Predictions for new data points:")
for i, (x_val, pred) in enumerate(zip(new_X.flatten(), predictions)):
    print(f"x = {x_val}: predicted y = {pred:.2f}")

# Section 8: Practical Example with Realistic Dataset
print("\nSection 8: Practical Example with Realistic Dataset")
print("-" * 50)
print("Creating a realistic example with house price prediction.")

# Create synthetic housing data
np.random.seed(789)
n_houses = 200
size = np.random.normal(2000, 500, n_houses)  # Square footage
bedrooms = np.random.randint(1, 6, n_houses)  # Number of bedrooms
age = np.random.randint(0, 50, n_houses)  # Age of house

# Create realistic price based on features
price = (
    size * 100 + 
    bedrooms * 10000 + 
    (50 - age) * 1000 + 
    np.random.normal(0, 10000, n_houses)  # Noise
)

# Create DataFrame
housing_df = pd.DataFrame({
    'size': size,
    'bedrooms': bedrooms,
    'age': age,
    'price': price
})

# Save dataset
housing_df.to_csv(os.path.join(assets_dir, 'housing_data.csv'), index=False)
print("Created synthetic housing dataset and saved to CSV")

# Prepare features and target
X_house = housing_df[['size', 'bedrooms', 'age']]
y_house = housing_df['price']

# Split data
X_train_house, X_test_house, y_train_house, y_test_house = train_test_split(
    X_house, y_house, test_size=0.2, random_state=42
)

# Fit model
house_model = LinearRegression()
house_model.fit(X_train_house, y_train_house)

# Predictions
y_pred_house = house_model.predict(X_test_house)

# Evaluate
mse_house = mean_squared_error(y_test_house, y_pred_house)
rmse_house = np.sqrt(mse_house)
r2_house = r2_score(y_test_house, y_pred_house)

print(f"Housing Price Prediction Model Results:")
print(f"MSE: {mse_house:.0f}")
print(f"RMSE: {rmse_house:.0f}")
print(f"R²: {r2_house:.4f}")

# Print feature importance
feature_names = ['Size', 'Bedrooms', 'Age']
coefficients = house_model.coef_
print("\nFeature Coefficients:")
for name, coef in zip(feature_names, coefficients):
    print(f"{name}: {coef:.2f}")

print("\n=== Lesson Summary ===")
print("This lesson covered:")
print("1. Basic concepts of linear regression")
print("2. Simple and multiple linear regression models")
print("3. Model evaluation techniques")
print("4. Train-test splitting for proper evaluation")
print("5. Assumptions of linear regression")
print("6. Making predictions with trained models")
print("7. Practical example with real-world data")
print("\nAll visualizations and datasets were saved to the assets folder.")