"""
Univariate Linear Regression Testing and Visualization

This module provides functions to test and visualize univariate linear regression,
including data plotting, regression line visualization, and objective function surface plots.

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from pathlib import Path

from linreg import LinearRegression


def plot_data_1d(X: np.ndarray, y: np.ndarray, title: str = "Univariate Data",
                 xlabel: str = "X", ylabel: str = "y") -> None:
    """
    Plot 1D scatter plot of training data.
    
    Args:
        X (np.ndarray): Feature values
        y (np.ndarray): Target values
        title (str): Plot title
        xlabel (str): X-axis label
        ylabel (str): Y-axis label
    """
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.plot(X, y, 'rx', markersize=8, label='Training Data')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_regression_line_1d(lr_model: LinearRegression, X: np.ndarray, 
                            y: np.ndarray, save_path: str = None) -> None:
    """
    Plot training data with the fitted regression line.
    
    Args:
        lr_model (LinearRegression): Trained linear regression model
        X (np.ndarray): Feature matrix (includes bias column)
        y (np.ndarray): Target values
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    plt.title("Linear Regression Fit", fontsize=14, fontweight='bold')
    plt.xlabel("X", fontsize=12)
    plt.ylabel("y", fontsize=12)
    
    # Plot training data
    plt.plot(X[:, 1], y, 'rx', markersize=8, label='Training Data')
    
    # Plot regression line
    predictions = lr_model.predict(X)
    plt.plot(X[:, 1], predictions, 'b-', linewidth=2, label='Regression Line')
    
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_objective(lr_model: LinearRegression, t1_vals: np.ndarray, 
                       t2_vals: np.ndarray, X: np.ndarray, y: np.ndarray,
                       save_path: str = None) -> None:
    """
    Visualize the cost function surface and contour plots with gradient descent path.
    
    This function creates two plots:
    1. 3D surface plot showing the cost function landscape
    2. Contour plot with the final theta position marked
    
    Args:
        lr_model (LinearRegression): Trained model with cost history
        t1_vals (np.ndarray): Range of theta0 values to evaluate
        t2_vals (np.ndarray): Range of theta1 values to evaluate
        X (np.ndarray): Feature matrix
        y (np.ndarray): Target values
        save_path (str, optional): Path to save the figure
    """
    T1, T2 = np.meshgrid(t1_vals, t2_vals)
    n, p = T1.shape
    
    # Compute cost function over the parameter space
    Z = np.zeros(T1.shape)
    for i in range(n):
        for j in range(p):
            theta_temp = np.array([[T1[i, j]], [T2[i, j]]])
            Z[i, j] = lr_model.compute_cost(X, y, theta_temp)
    
    # 3D Surface Plot
    fig = plt.figure(figsize=(14, 6))
    
    # Surface plot
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(T1, T2, Z, rstride=1, cstride=1, 
                           cmap=cm.coolwarm, linewidth=0, alpha=0.8)
    
    ax1.zaxis.set_major_locator(LinearLocator(10))
    ax1.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    ax1.set_xlabel('θ₀', fontsize=12)
    ax1.set_ylabel('θ₁', fontsize=12)
    ax1.set_zlabel('Cost J(θ)', fontsize=12)
    ax1.set_title('Cost Function Surface', fontsize=14, fontweight='bold')
    
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Plot gradient descent path
    if lr_model.cost_history is not None:
        for ii in range(len(lr_model.cost_history) - 1):
            t1_current = lr_model.cost_history[ii][1][0, 0]
            t2_current = lr_model.cost_history[ii][1][1, 0]
            J_current = lr_model.cost_history[ii][0]
            
            t1_next = lr_model.cost_history[ii + 1][1][0, 0]
            t2_next = lr_model.cost_history[ii + 1][1][1, 0]
            J_next = lr_model.cost_history[ii + 1][0]
            
            ax1.plot3D([t1_current, t1_next], [t2_current, t2_next], 
                      [J_current, J_next], 'b-', linewidth=1, alpha=0.6)
        
        # Mark key points
        for J, theta in lr_model.cost_history[::50]:  # Plot every 50th point
            ax1.plot3D([theta[0, 0]], [theta[1, 0]], [J], 'mo', markersize=4)
    
    # Contour Plot
    ax2 = fig.add_subplot(122)
    CS = ax2.contour(T1, T2, Z, levels=20, cmap='viridis')
    ax2.clabel(CS, inline=True, fontsize=8, fmt='%.1f')
    ax2.set_xlabel('θ₀', fontsize=12)
    ax2.set_ylabel('θ₁', fontsize=12)
    ax2.set_title('Cost Function Contours', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Mark the final theta
    ax2.plot(lr_model.theta[0, 0], lr_model.theta[1, 0], 'r*', 
            markersize=15, label='Optimal θ')
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def main():
    """Main function to test univariate linear regression."""
    
    # Load the data
    data_path = Path("data/univariateData.dat")
    
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure the data file is in the correct location.")
        return
    
    all_data = np.loadtxt(data_path, delimiter=',')
    
    X = all_data[:, :-1]
    y = all_data[:, -1].reshape(-1, 1)
    
    n, d = X.shape
    print(f"Dataset shape: {n} samples, {d} features")
    
    # Add bias term (column of ones)
    X = np.c_[np.ones((n, 1)), X]
    
    # Initialize model parameters
    # Note: Starting near [10, 10] for better gradient descent visualization
    init_theta = np.ones((d + 1, 1)) * 10
    n_iter = 1500
    alpha = 0.01
    
    print(f"\nTraining Configuration:")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Iterations: {n_iter}")
    print(f"  Initial θ: {init_theta.T}")
    
    # Create and train model
    print("\n" + "="*60)
    print("Training Linear Regression Model")
    print("="*60)
    
    lr_model = LinearRegression(init_theta=init_theta, alpha=alpha, n_iter=n_iter)
    
    # Visualize original data
    plot_data_1d(X[:, 1], y)
    
    # Train the model
    lr_model.fit(X, y)
    
    # Display results
    print("\n" + "="*60)
    print("Training Complete")
    print("="*60)
    print(f"Final θ: {lr_model.theta.T}")
    print(f"Final cost: {lr_model.cost_history[-1][0]:.6f}")
    print(f"R² score: {lr_model.score(X, y):.6f}")
    
    # Plot regression line
    plot_regression_line_1d(lr_model, X, y)
    
    # Visualize cost function surface
    theta1_vals = np.linspace(-10, 10, 100)
    theta2_vals = np.linspace(-10, 10, 100)
    visualize_objective(lr_model, theta1_vals, theta2_vals, X, y)


if __name__ == "__main__":
    main()