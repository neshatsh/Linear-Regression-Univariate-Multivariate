import numpy as np
from typing import Optional, List, Tuple


class LinearRegression:
    """
    Linear Regression model using gradient descent optimization.
    
    This class implements a linear regression model that learns parameters theta
    by minimizing the mean squared error cost function using gradient descent.
    
    Attributes:
        alpha (float): Learning rate for gradient descent
        n_iter (int): Number of iterations for gradient descent
        theta (np.ndarray): Model parameters (weights)
        cost_history (List[Tuple[float, np.ndarray]]): History of cost values and theta values
    """

    def __init__(self, init_theta: Optional[np.ndarray] = None, 
                 alpha: float = 0.01, 
                 n_iter: int = 100):
        """
        Initialize the Linear Regression model.
        
        Args:
            init_theta (np.ndarray, optional): Initial theta values. Defaults to None.
            alpha (float): Learning rate. Defaults to 0.01.
            n_iter (int): Number of gradient descent iterations. Defaults to 100.
        """
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.cost_history = None
    

    def compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        '''
        Computes the mean squared error cost function.

        The cost function is defined as:
        J(θ) = (1/2n) * Σ(h(x^(i)) - y^(i))^2
        where h(x) = θ^T * x

        Args:
            X (np.ndarray): Feature matrix of shape (n, d) where n is number of samples
                           and d is number of features
            y (np.ndarray): Target vector of shape (n, 1)
            theta (np.ndarray): Parameter vector of shape (d, 1)

        Returns:
            float: The computed cost value
        '''
        n = len(y)
        predictions = X @ theta 
        cost = (1 / (2 * n)) * np.sum(np.square(predictions - y))

        return float(cost)


    def gradient_descent(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        '''
        Optimize model parameters using gradient descent algorithm.

        The update rule is:
        θ_{i+1} = θ_i - alpha * (1/n) * X^T * (X*theta - y)

        Args:
            X (np.ndarray): Feature matrix of shape (n, d)
            y (np.ndarray): Target vector of shape (n, 1)
            theta (np.ndarray): Initial parameter vector of shape (d, 1)
        Returns:
            np.ndarray: Optimized theta values  
        '''
        n, d = X.shape
        self.cost_history = []

        for i in range(self.n_iter):
            # Compute and store cost
            cost = self.compute_cost(X, y, theta)
            self.cost_history.append((cost, theta.copy()))

            # Print progress every 100 iterations
            if (i + 1) % 100 == 0 or i == 0:
                print(f"Iteration {i + 1:4d} | Cost: {cost:.6f}")
            
            # Compute gradient 
            gradient = (1 / n) * X.T @ (X @ theta - y)

            # Update theta
            theta = theta - self.alpha * gradient

        return theta

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        '''
        Train the linear regression model on the provided data.

        Args:
            X (np.ndarray): Feature matrix of shape (n, d)
            y (np.ndarray): Target vector of shape (n, 1) or (n,)

        Returns:
            LinearRegression: The fitted model (self)
        '''

        # Ensure y is a column vector
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n, d = X.shape

        # Initialize theta if not provided  
        if self.theta is None:
            self.theta = np.zeros((d, 1))

        # Run gradient descent
        self.theta = self.gradient_descent(X, y, self.theta)
        
        return self 

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Make predictions using the trained model.

        Args:
            X (np.ndarray): Feature matrix of shape (n, d)

        Returns:
            np.ndarray: Predicted values of shape (n, 1)

        Raises:
            ValueError: If the model has not been trained yet
        '''
        if self.theta is None:
            raise ValueError("Model must be trained before making predictions. Call fit() first.")
        
        return X @ self.theta


    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score (coefficient of determination).
        
        Args:
            X (np.ndarray): Feature matrix of shape (n, d)
            y (np.ndarray): True target values of shape (n, 1) or (n,)
        
        Returns:
            float: R² score
        """
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        predictions = self.predict(X)
        ss_res = np.sum(np.square(y - predictions))
        ss_tot = np.sum(np.square(y - np.mean(y)))
        
        return float(1 - (ss_res / ss_tot))
    
    def get_cost_history(self) -> Optional[List[Tuple[float, np.ndarray]]]:
        """
        Get the history of cost values during training.
        
        Returns:
            List[Tuple[float, np.ndarray]]: List of (cost, theta) tuples
        """
        return self.cost_history
