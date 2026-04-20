import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
def generate_data(n_samples=100):
    """Generate noisy sine wave data"""
    X = np.linspace(0, 1, n_samples).reshape(-1, 1)
    y = np.sin(2 * np.pi * X).ravel() + np.random.normal(0, 0.1, n_samples)
    return X, y

# Create polynomial models with different degrees
def create_models():
    """Create polynomial models with varying complexity"""
    models = {}
    
    # Very simple model (high bias, low variance)
    models['underfit'] = Pipeline([
        ('poly', PolynomialFeatures(degree=1)),
        ('linear', LinearRegression())
    ])
    
    # Moderate model (balanced)
    models['normal'] = Pipeline([
        ('poly', PolynomialFeatures(degree=4)),
        ('linear', LinearRegression())
    ])
    
    # Complex model (low bias, high variance)
    models['overfit'] = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('linear', LinearRegression())
    ])
    
    return models

# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model on training and test sets"""
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate errors
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    
    return train_mse, test_mse

# Main execution
if __name__ == "__main__":
    # Generate data
    X_full, y_full = generate_data(100)
    
    # Split into train and test sets
    X_train, X_test = X_full[:80], X_full[80:]
    y_train, y_test = y_full[:80], y_full[80:]
    
    # Create models
    models = create_models()
    
    # Evaluate models
    results = {}
    for name, model in models.items():
        train_mse, test_mse = evaluate_model(model, X_train, y_train, X_test, y_test)
        results[name] = {'train_mse': train_mse, 'test_mse': test_mse}
    
    # Print results
    print("Model Performance Comparison:")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"{name:8} | Train MSE: {metrics['train_mse']:.4f} | Test MSE: {metrics['test_mse']:.4f}")
    
    # Visualization of bias-variance tradeoff
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Data and models
    plt.subplot(2, 2, 1)
    X_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    
    # Plot true function
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), 'b-', linewidth=2, label='True Function')
    
    # Plot training data
    plt.scatter(X_train, y_train, c='red', alpha=0.6, label='Training Data')
    
    # Plot predictions for each model
    colors = ['green', 'orange', 'purple']
    names = ['underfit', 'normal', 'overfit']
    
    for i, (name, color) in enumerate(zip(names, colors)):
        model = models[name]
        model.fit(X_train, y_train)
        y_plot = model.predict(X_plot)
        plt.plot(X_plot, y_plot, color=color, linewidth=2, 
                label=f'{name.capitalize()} Model')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Models vs True Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Bias-Variance decomposition
    plt.subplot(2, 2, 2)
    
    # Create multiple datasets to demonstrate variance
    n_datasets = 50
    degrees = [1, 4, 15]
    colors = ['green', 'orange', 'purple']
    
    # Generate multiple datasets and fit models
    for d, color in zip(degrees, colors):
        # Store predictions for all datasets
        all_predictions = []
        
        for _ in range(n_datasets):
            # Generate new dataset
            X_new, y_new = generate_data(80)
            
            # Fit model
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=d)),
                ('linear', LinearRegression())
            ])
            model.fit(X_new, y_new)
            
            # Get predictions on X_plot
            y_pred = model.predict(X_plot)
            all_predictions.append(y_pred)
        
        # Calculate mean prediction and variance
        mean_pred = np.mean(all_predictions, axis=0)
        var_pred = np.var(all_predictions, axis=0)
        
        # Plot mean prediction
        plt.plot(X_plot, mean_pred, color=color, linewidth=2, 
                label=f'Degree {d} - Mean Prediction')
        
        # Plot variance bands
        plt.fill_between(X_plot.ravel(), 
                        mean_pred - 2*np.sqrt(var_pred),
                        mean_pred + 2*np.sqrt(var_pred),
                        color=color, alpha=0.2)
    
    # Plot true function
    plt.plot(X_plot, np.sin(2 * np.pi * X_plot), 'b--', linewidth=2, label='True Function')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Bias-Variance Tradeoff Visualization')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Training vs Test Error vs Complexity
    plt.subplot(2, 2, 3)
    
    # Vary model complexity from 1 to 20
    complexities = range(1, 21)
    train_errors = []
    test_errors = []
    
    for degree in complexities:
        model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        
        # Fit and evaluate
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        train_errors.append(mean_squared_error(y_train, y_train_pred))
        test_errors.append(mean_squared_error(y_test, y_test_pred))
    
    plt.plot(complexities, train_errors, 'o-', label='Training Error', linewidth=2)
    plt.plot(complexities, test_errors, 's-', label='Test Error', linewidth=2)
    plt.xlabel('Model Complexity (Polynomial Degree)')
    plt.ylabel('Mean Squared Error')
    plt.title('Bias-Variance Tradeoff')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Bias-Variance Decomposition
    plt.subplot(2, 2, 4)
    
    # Calculate bias and variance components
    # We'll focus on the overfit model (complexity 15) for demonstration
    model_complex = Pipeline([
        ('poly', PolynomialFeatures(degree=15)),
        ('linear', LinearRegression())
    ])
    
    # Generate multiple datasets
    n_datasets = 100
    predictions = []
    
    for _ in range(n_datasets):
        X_new, y_new = generate_data(80)
        model_complex.fit(X_new, y_new)
        pred = model_complex.predict(X_plot)
        predictions.append(pred)
    
    # Calculate mean prediction
    mean_pred = np.mean(predictions, axis=0)
    
    # Calculate bias^2 and variance
    bias_squared = (mean_pred - np.sin(2 * np.pi * X_plot.ravel())) ** 2
    variance = np.var(predictions, axis=0)
    
    # Plot components
    plt.plot(X_plot, bias_squared, 'r-', linewidth=2, label='Bias²')
    plt.plot(X_plot, variance, 'g-', linewidth=2, label='Variance')
    plt.plot(X_plot, bias_squared + variance, 'b--', linewidth=2, label='Total Error')
    
    plt.xlabel('X')
    plt.ylabel('Error Components')
    plt.title('Bias-Variance Decomposition')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_60838c0318584faeb369e3d8493439da/assets/bias_variance_tradeoff.png')
    plt.close()
    
    # Print detailed analysis
    print("\nDetailed Analysis:")
    print("=" * 50)
    print("Underfit Model (Degree 1):")
    print("  - High bias: Simple model can't capture complex patterns")
    print("  - Low variance: Consistent predictions across datasets")
    print("  - Results in high training error")
    
    print("\nNormal Model (Degree 4):")
    print("  - Balanced bias and variance")
    print("  - Good generalization to unseen data")
    print("  - Optimal tradeoff")
    
    print("\nOverfit Model (Degree 15):")
    print("  - Low bias: Complex model fits training data very well")
    print("  - High variance: Sensitive to small changes in training data")
    print("  - Results in low training error but high test error")
    
    print("\nThe Bias-Variance Tradeoff shows that we must balance:")
    print("1. Model complexity (bias)")
    print("2. Sensitivity to training data (variance)")
    print("3. Generalization ability (test performance)")