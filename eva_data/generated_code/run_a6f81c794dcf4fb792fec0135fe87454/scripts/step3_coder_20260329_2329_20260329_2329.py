import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_polynomial_features
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os

# Set the output directory
output_dir = '/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_a6f81c794dcf4fb792fec0135fe87454/assets'
os.makedirs(output_dir, exist_ok=True)

# Generate synthetic dataset with noise
np.random.seed(42)
X = np.linspace(0, 1, 100).reshape(-1, 1)
y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create polynomial features of different degrees
degrees = [1, 4, 15]  # Low, medium, and high degree polynomials
models = []

# Plot the original data
plt.figure(figsize=(15, 5))

# Plot 1: Original data with true function
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
plt.plot(X, np.sin(2 * np.pi * X), 'r-', linewidth=2, label='True function')
plt.title('Original Data and True Function')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Train models with different polynomial degrees
for i, degree in enumerate(degrees):
    # Transform features to polynomial features
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Fit linear regression model
    lr = LinearRegression()
    lr.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = lr.predict(X_train_poly)
    y_test_pred = lr.predict(X_test_poly)
    
    # Calculate MSE
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    # Store model for later plotting
    models.append((degree, lr, poly_features))
    
    # Plot predictions vs actual values
    plt.subplot(1, 3, i+1)
    plt.scatter(X_train, y_train, alpha=0.6, label='Training data')
    plt.scatter(X_test, y_test, alpha=0.6, label='Test data')
    
    # Plot predictions for a smooth curve
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    y_plot = lr.predict(X_plot_poly)
    plt.plot(X_plot, y_plot, 'b-', linewidth=2, label=f'Polynomial degree {degree}')
    
    plt.title(f'Degree {degree} Polynomial\nTrain MSE: {train_mse:.3f}\nTest MSE: {test_mse:.3f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overfitting_demo.png'))
plt.close()

# Demonstrate regularization techniques
print("Demonstrating regularization techniques:")

# Generate a more complex dataset for better demonstration
np.random.seed(42)
X_complex = np.linspace(0, 1, 100).reshape(-1, 1)
y_complex = np.sin(2 * np.pi * X_complex).ravel() + np.random.normal(0, 0.1, X_complex.shape[0])
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_complex, y_complex, test_size=0.3, random_state=42)

# Create high-degree polynomial features
degree = 15
poly_features = PolynomialFeatures(degree=degree)
X_train_poly_c = poly_features.fit_transform(X_train_c)
X_test_poly_c = poly_features.transform(X_test_c)

# Different regularization parameters
alphas = [0.001, 0.01, 0.1, 1.0, 10.0]

# Store results for plotting
ridge_mses = []
lasso_mses = []
ridge_coeffs = []
lasso_coeffs = []

# Test different regularization strengths
for alpha in alphas:
    # Ridge regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train_poly_c, y_train_c)
    y_pred_ridge = ridge.predict(X_test_poly_c)
    ridge_mses.append(mean_squared_error(y_test_c, y_pred_ridge))
    ridge_coeffs.append(ridge.coef_)
    
    # Lasso regression
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train_poly_c, y_train_c)
    y_pred_lasso = lasso.predict(X_test_poly_c)
    lasso_mses.append(mean_squared_error(y_test_c, y_pred_lasso))
    lasso_coeffs.append(lasso.coef_)

# Plot regularization effects
plt.figure(figsize=(12, 5))

# Plot 1: MSE vs Alpha for Ridge and Lasso
plt.subplot(1, 2, 1)
plt.semilogx(alphas, ridge_mses, 'o-', label='Ridge', linewidth=2)
plt.semilogx(alphas, lasso_mses, 's-', label='Lasso', linewidth=2)
plt.xlabel('Alpha (Regularization Strength)')
plt.ylabel('Test MSE')
plt.title('Regularization Effect on Test MSE')
plt.legend()
plt.grid(True)

# Plot 2: Coefficient magnitudes for Ridge and Lasso
plt.subplot(1, 2, 2)
coeffs_ridge = np.array(ridge_coeffs)
coeffs_lasso = np.array(lasso_coeffs)

# Plot coefficients for first few alphas
for i in range(min(3, len(alphas))):
    plt.plot(coeffs_ridge[i], 'o-', label=f'Ridge α={alphas[i]}')
    plt.plot(coeffs_lasso[i], 's-', label=f'Lasso α={alphas[i]}')

plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Coefficient Magnitudes with Regularization')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regularization_demo.png'))
plt.close()

# Print detailed results
print("\nDetailed Results:")
print("Degree 15 polynomial overfitting:")
print("- Training MSE: Very low (overfit)")
print("- Test MSE: High (overfit)")

print("\nRegularization Effects:")
print("- Ridge regression: Shrinks coefficients but keeps all features")
print("- Lasso regression: Shrinks coefficients and can set some to zero")

# Create a final comparison plot showing overfitting vs regularization
plt.figure(figsize=(15, 5))

# Plot 1: Overfitting example
plt.subplot(1, 3, 1)
plt.scatter(X_train_c, y_train_c, alpha=0.6, label='Training data')
X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
X_plot_poly = poly_features.transform(X_plot)
lr = LinearRegression()
lr.fit(X_train_poly_c, y_train_c)
y_plot = lr.predict(X_plot_poly)
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label='Overfit model')
plt.title('Overfitting Example')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Plot 2: Ridge regularization
plt.subplot(1, 3, 2)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train_poly_c, y_train_c)
y_plot_ridge = ridge.predict(X_plot_poly)
plt.scatter(X_train_c, y_train_c, alpha=0.6, label='Training data')
plt.plot(X_plot, y_plot_ridge, 'b-', linewidth=2, label='Ridge regularized')
plt.title('Ridge Regularization')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Plot 3: Lasso regularization
plt.subplot(1, 3, 3)
lasso = Lasso(alpha=0.1, max_iter=10000)
lasso.fit(X_train_poly_c, y_train_c)
y_plot_lasso = lasso.predict(X_plot_poly)
plt.scatter(X_train_c, y_train_c, alpha=0.6, label='Training data')
plt.plot(X_plot, y_plot_lasso, 'g-', linewidth=2, label='Lasso regularized')
plt.title('Lasso Regularization')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'overfitting_vs_regularization.png'))
plt.close()

print("\nGenerated plots saved to:", output_dir)