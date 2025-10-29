"""
Multivariate Linear Regression Testing and Evaluation

This module tests multivariate linear regression with feature standardization
and evaluates the model on a holdout test set.

"""

import numpy as np
from pathlib import Path
from typing import Tuple

from linreg import LinearRegression


def load_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data from file.

    Args:
        file_path (str): Path to the data file

    Returns:
        Tuple[np.ndarray, np.ndarray]: Feature matrix X and target vector y
    """
    all_data = np.loadtxt(file_path, delimiter=',')
    X = all_data[:, :-1]
    y = all_data[:, -1].reshape(-1, 1)
    return X, y


def standardize_features(X: np.ndarray, mean: np.ndarray = None,
                         std: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using Z-score normalization.

    Args:
        X (np.ndarray): Feature matrix
        mean (np.ndarray, optional): Mean values for standardization
        std (np.ndarray, optional): Standard deviation values for standardization

    Returns:
        Tuple containing:
            - Standardized feature matrix
            - Mean values used
            - Standard deviation values used
    """
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)

    # Avoid division by zero
    std[std == 0] = 1

    X_standardized = (X - mean) / std
    return X_standardized, mean, std


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """
    Add bias term (column of ones) to feature matrix.

    Args:
        X (np.ndarray): Feature matrix of shape (n, d)

    Returns:
        np.ndarray: Feature matrix with bias term of shape (n, d+1)
    """
    n = X.shape[0]
    return np.c_[np.ones((n, 1)), X]


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: RMSE value
    """
    mse = np.mean(np.square(y_pred - y_true))
    return np.sqrt(mse)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        y_true (np.ndarray): True target values
        y_pred (np.ndarray): Predicted values

    Returns:
        float: MAE value
    """
    return np.mean(np.abs(y_pred - y_true))


def main():
    """Main function to test multivariate linear regression."""

    # Load training data
    train_path = Path('data/multivariateData.dat')

    if not train_path.exists():
        print(f"Error: Training data file not found at {train_path}")
        return

    print("Multivariate Linear Regression")

    X_train, y_train = load_data(train_path)
    n, d = X_train.shape

    print(f"\nTraining Data:")
    print(f"  Samples: {n}")
    print(f"  Features: {d}")
    print(f"  Target shape: {y_train.shape}")

    # Standardize features
    print("\nStandardizing features...")
    X_train_std, mean, std = standardize_features(X_train)

    # Add bias term
    X_train_std = add_bias_term(X_train_std)

    # Initialize model parameters
    init_theta = np.random.randn(d + 1, 1)
    n_iter = 2000
    alpha = 0.1

    print(f"\nTraining Configuration:")
    print(f"  Learning rate (α): {alpha}")
    print(f"  Iterations: {n_iter}")
    print(f"  Initial θ shape: {init_theta.shape}")

    # Train the model
    print("Training Model")

    lr_model = LinearRegression(init_theta=init_theta, alpha=alpha, n_iter=n_iter)
    lr_model.fit(X_train_std, y_train)

    # Training results
    print("Training Results")
    print(f"Final cost: {lr_model.cost_history[-1][0]:.6f}")

    # Evaluate on training set
    y_train_pred = lr_model.predict(X_train_std)
    train_rmse = calculate_rmse(y_train, y_train_pred)
    train_mae = calculate_mae(y_train, y_train_pred)
    train_r2 = lr_model.score(X_train_std, y_train)

    print(f"\nTraining Set Performance:")
    print(f"  RMSE: {train_rmse:.6f}")
    print(f"  MAE:  {train_mae:.6f}")
    print(f"  R²:   {train_r2:.6f}")

    # Load and evaluate on test set
    test_path = Path('data/holdout.npz')

    if not test_path.exists():
        print(f"\nWarning: Test data file not found at {test_path}")
        print("Skipping test set evaluation.")
        return

    print("Test Set Evaluation")

    test_data = np.load(test_path)['arr_0']
    X_test = test_data[:, :-1]
    y_test = test_data[:, -1].reshape(-1, 1)

    print(f"\nTest Data:")
    print(f"  Samples: {X_test.shape[0]}")
    print(f"  Features: {X_test.shape[1]}")

    # Standardize test data using training statistics
    X_test_std, _, _ = standardize_features(X_test, mean=mean, std=std)
    X_test_std = add_bias_term(X_test_std)

    # Make predictions
    y_test_pred = lr_model.predict(X_test_std)

    # Calculate metrics
    test_rmse = calculate_rmse(y_test, y_test_pred)
    test_mae = calculate_mae(y_test, y_test_pred)
    test_r2 = lr_model.score(X_test_std, y_test)

    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_rmse:.6f}")
    print(f"  MAE:  {test_mae:.6f}")
    print(f"  R²:   {test_r2:.6f}")

    # Check for overfitting/underfitting
    print("Model Analysis")

    rmse_diff = abs(train_rmse - test_rmse)
    rmse_ratio = test_rmse / train_rmse if train_rmse > 0 else float('inf')

    print(f"\nRMSE Comparison:")
    print(f"  Train RMSE: {train_rmse:.6f}")
    print(f"  Test RMSE:  {test_rmse:.6f}")
    print(f"  Difference: {rmse_diff:.6f}")
    print(f"  Ratio (Test/Train): {rmse_ratio:.4f}")

    if rmse_ratio > 1.2:
        print("\n Warning: Possible overfitting detected (test RMSE >> train RMSE)")
    elif rmse_ratio < 0.8:
        print("\n Warning: Unusual pattern (test RMSE << train RMSE)")
    else:
        print("\n Model generalizes well to test data")

    print("Evaluation Complete")

if __name__ == "__main__":
    main()