import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
def generate_data(n_samples=100):
    """Generate synthetic data for linear regression"""
    X = np.random.randn(n_samples)
    # True relationship: y = 2*x + 1 + noise
    y = 2 * X + 1 + np.random.randn(n_samples) * 0.5
    return X, y

# Define the model: y = wx + b
def linear_model(X, w, b):
    """Linear model function"""
    return w * X + b

# Define the loss function: Mean Squared Error
def mse_loss(y_true, y_pred):
    """Mean Squared Error loss function"""
    return np.mean((y_true - y_pred) ** 2)

# Define the gradient of the loss function
def compute_gradients(X, y, w, b):
    """Compute gradients of the loss function with respect to parameters"""
    n = len(X)
    
    # Predictions
    y_pred = linear_model(X, w, b)
    
    # Compute gradients
    dw = -2/n * np.sum(X * (y - y_pred))
    db = -2/n * np.sum(y - y_pred)
    
    return dw, db

# Gradient descent algorithm
def gradient_descent(X, y, learning_rate=0.01, iterations=1000, tolerance=1e-6):
    """
    Perform gradient descent optimization
    
    Parameters:
    X: input features
    y: target values
    learning_rate: step size for updates
    iterations: number of iterations to run
    tolerance: convergence threshold
    
    Returns:
    w_history: history of weight values
    b_history: history of bias values
    loss_history: history of loss values
    """
    
    # Initialize parameters randomly
    w = np.random.randn()
    b = np.random.randn()
    
    # Store histories for visualization
    w_history = [w]
    b_history = [b]
    loss_history = []
    
    # Gradient descent loop
    for i in range(iterations):
        # Compute current loss
        y_pred = linear_model(X, w, b)
        current_loss = mse_loss(y, y_pred)
        loss_history.append(current_loss)
        
        # Compute gradients
        dw, db = compute_gradients(X, y, w, b)
        
        # Update parameters using gradient descent rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Store updated values
        w_history.append(w)
        b_history.append(b)
        
        # Check for convergence
        if i > 0 and abs(loss_history[-2] - loss_history[-1]) < tolerance:
            print(f"Converged after {i} iterations")
            break
    
    return w, b, w_history, b_history, loss_history

# Visualization functions
def plot_data_and_fit(X, y, w, b, title="Data and Fitted Line"):
    """Plot the data points and the fitted line"""
    plt.figure(figsize=(10, 6))
    
    # Plot data points
    plt.scatter(X, y, alpha=0.6, label='Data points')
    
    # Plot fitted line
    x_line = np.linspace(X.min(), X.max(), 100)
    y_line = linear_model(x_line, w, b)
    plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted line: y = {w:.2f}x + {b:.2f}')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_68dadac37f91488f81658ac1bd385654/assets/gradient_descent_fit.png')
    plt.close()

def plot_loss_history(loss_history, title="Loss Function Over Iterations"):
    """Plot how the loss decreases over iterations"""
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_68dadac37f91488f81658ac1bd385654/assets/loss_history.png')
    plt.close()

def plot_parameter_evolution(w_history, b_history, title="Parameter Evolution"):
    """Plot how parameters change during training"""
    plt.figure(figsize=(10, 6))
    plt.plot(w_history, label='Weight (w)', marker='o', markersize=3)
    plt.plot(b_history, label='Bias (b)', marker='s', markersize=3)
    plt.xlabel('Iteration')
    plt.ylabel('Parameter Value')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_68dadac37f91488f81658ac1bd385654/assets/parameter_evolution.png')
    plt.close()

# Main execution
if __name__ == "__main__":
    # Generate data
    X, y = generate_data(100)
    
    # Run gradient descent
    print("Starting gradient descent...")
    final_w, final_b, w_history, b_history, loss_history = gradient_descent(
        X, y, learning_rate=0.01, iterations=1000
    )
    
    print(f"Final parameters: w = {final_w:.4f}, b = {final_b:.4f}")
    
    # Create visualizations
    plot_data_and_fit(X, y, final_w, final_b)
    plot_loss_history(loss_history)
    plot_parameter_evolution(w_history, b_history)
    
    # Print final results
    print("Gradient descent completed successfully!")
    print(f"True parameters: w = 2.0, b = 1.0")
    print(f"Learned parameters: w = {final_w:.4f}, b = {final_b:.4f}")
    print(f"Final loss: {loss_history[-1]:.6f}")
    
    # Save final loss value to file
    with open('/root/shiyou/AI_Pedia_Local_stream/data/generated_code/run_68dadac37f91488f81658ac1bd385654/assets/final_loss.txt', 'w') as f:
        f.write(f"Final Loss: {loss_history[-1]:.6f}\n")
        f.write(f"Final Parameters: w={final_w:.4f}, b={final_b:.4f}\n")