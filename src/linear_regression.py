"""
Linear Regression Implementation from Scratch
Author: [Your Name]
Date: [Date]

Mathematical Foundation:
------------------------
1. Hypothesis: hθ(x) = θ₀ + θ₁x₁ + θ₂x₂ + ... + θₙxₙ
2. Vector form: hθ(x) = Xθ
3. Cost Function: J(θ) = 1/2m * Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)²
4. Gradient: ∇J(θ) = 1/m * Xᵀ(Xθ - y)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import pandas as pd


class LinearRegression:
    """
    Linear Regression implementation with:
    - Gradient Descent (Batch, Mini-batch, Stochastic)
    - Normal Equation (Closed-form solution)
    - Regularization (Ridge/L2)
    - Learning rate scheduling
    """
    
    def __init__(self, method: str = 'gradient_descent', 
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 regularization: float = 0.0,
                 batch_size: Optional[int] = None):
        """
        Parameters:
        -----------
        method : str
            'gradient_descent' or 'normal_equation'
        learning_rate : float
            Step size for gradient descent
        n_iterations : int
            Number of iterations for gradient descent
        regularization : float
            L2 regularization parameter (lambda)
        batch_size : int or None
            For mini-batch gradient descent
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.batch_size = batch_size
        
        # Parameters to be learned
        self.theta = None  # Coefficients
        self.intercept = None  # Bias term
        self.cost_history = []  # To track learning
        self.gradient_history = []  # To track gradient norms
        
    def _add_intercept(self, X: np.ndarray) -> np.ndarray:
        """Add intercept term (column of ones) to features."""
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate([intercept, X], axis=1)
    
    def _compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Mean Squared Error cost function.
        
        J(θ) = 1/(2m) * Σ(hθ(x⁽ⁱ⁾) - y⁽ⁱ⁾)² + λ/(2m) * Σθⱼ²
        
        where hθ(x) = Xθ
        """
        m = X.shape[0]
        predictions = X.dot(self.theta)
        error = predictions - y
        
        # Mean Squared Error
        mse = (1/(2*m)) * np.sum(error ** 2)
        
        # Add regularization (excluding bias term θ₀)
        if self.regularization > 0:
            reg_term = (self.regularization/(2*m)) * np.sum(self.theta[1:]**2)
            mse += reg_term
            
        return mse
    
    def _compute_gradient(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of cost function.
        
        ∇J(θ) = 1/m * Xᵀ(Xθ - y) + λ/m * θ
        (Note: regularization excludes bias term)
        """
        m = X.shape[0]
        predictions = X.dot(self.theta)
        error = predictions - y
        gradient = (1/m) * X.T.dot(error)
        
        # Add regularization gradient (excluding bias)
        if self.regularization > 0:
            gradient[1:] += (self.regularization/m) * self.theta[1:]
            
        return gradient
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            verbose: bool = False, early_stopping: bool = False,
            tolerance: float = 1e-6) -> 'LinearRegression':
        """
        Train the linear regression model.
        
        Parameters:
        -----------
        X : np.ndarray
            Training features of shape (m, n)
        y : np.ndarray
            Target values of shape (m,)
        verbose : bool
            Print training progress
        early_stopping : bool
            Stop if cost doesn't improve significantly
        tolerance : float
            Minimum cost improvement for early stopping
        """
        # Add intercept term
        X_b = self._add_intercept(X)
        
        # Initialize parameters (θ₀, θ₁, θ₂, ...)
        self.theta = np.random.randn(X_b.shape[1]) * 0.01
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X_b, y)
        else:
            self._fit_gradient_descent(X_b, y, verbose, early_stopping, tolerance)
            
        # Separate intercept from coefficients
        self.intercept = self.theta[0]
        self.coefficients = self.theta[1:]
        
        return self
    
    def _fit_normal_equation(self, X_b: np.ndarray, y: np.ndarray) -> None:
        """
        Closed-form solution using normal equation.
        
        θ = (XᵀX)⁻¹ Xᵀy
        
        Time Complexity: O(n³) - Good for small n (< 1000)
        Space Complexity: O(n²)
        
        Advantages: No iterations, no learning rate
        Disadvantages: Doesn't scale well, requires XᵀX to be invertible
        """
        m, n = X_b.shape
        
        # Normal equation with regularization (Ridge regression)
        XTX = X_b.T.dot(X_b)
        
        # Add regularization term (excluding bias)
        if self.regularization > 0:
            reg_matrix = self.regularization * np.eye(n)
            reg_matrix[0, 0] = 0  # Don't regularize intercept
            XTX += reg_matrix
        
        try:
            # Solve for theta
            self.theta = np.linalg.inv(XTX).dot(X_b.T).dot(y)
            
            # Compute cost for consistency
            cost = self._compute_cost(X_b, y)
            self.cost_history.append(cost)
            
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            print("Warning: XᵀX is singular, using pseudo-inverse")
            self.theta = np.linalg.pinv(X_b).dot(y)
    
    def _fit_gradient_descent(self, X_b: np.ndarray, y: np.ndarray,
                            verbose: bool, early_stopping: bool,
                            tolerance: float) -> None:
        """
        Iterative optimization using gradient descent.
        
        Update rule: θ := θ - α * ∇J(θ)
        
        Types:
        - Batch: Uses all training examples
        - Stochastic: Uses one example per iteration  
        - Mini-batch: Uses batch_size examples per iteration
        """
        m = X_b.shape[0]
        
        # Determine batch size
        if self.batch_size is None:
            batch_size = m  # Batch gradient descent
        elif self.batch_size == 1:
            batch_size = 1  # Stochastic gradient descent
        else:
            batch_size = self.batch_size  # Mini-batch
        
        for iteration in range(self.n_iterations):
            if batch_size < m:
                # Mini-batch or Stochastic GD
                indices = np.random.choice(m, batch_size, replace=False)
                X_batch = X_b[indices]
                y_batch = y[indices]
            else:
                # Batch GD
                X_batch = X_b
                y_batch = y
            
            # Compute gradient and update parameters
            gradient = self._compute_gradient(X_batch, y_batch)
            self.theta -= self.learning_rate * gradient
            
            # Compute and store cost
            cost = self._compute_cost(X_b, y)
            self.cost_history.append(cost)
            self.gradient_history.append(np.linalg.norm(gradient))
            
            # Early stopping
            if (early_stopping and iteration > 0 and 
                abs(self.cost_history[-2] - cost) < tolerance):
                if verbose:
                    print(f"Early stopping at iteration {iteration}")
                break
            
            # Progress reporting
            if verbose and iteration % 100 == 0:
                print(f"Iteration {iteration}: Cost = {cost:.6f}, "
                      f"Gradient Norm = {np.linalg.norm(gradient):.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        X_b = self._add_intercept(X)
        return X_b.dot(self.theta)
    
    def r2_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² (coefficient of determination).
        
        R² = 1 - SS_res / SS_tot
        where SS_res = Σ(y - ŷ)², SS_tot = Σ(y - ȳ)²
        """
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
    
    def plot_learning_curve(self, figsize: Tuple[int, int] = (12, 4)):
        """Visualize cost function and gradient during training."""
        if not self.cost_history:
            print("Model not trained yet!")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Cost function over iterations
        axes[0].plot(self.cost_history)
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Cost')
        axes[0].set_title('Cost Function Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Gradient norm over iterations
        if self.gradient_history:
            axes[1].plot(self.gradient_history)
            axes[1].set_xlabel('Iteration')
            axes[1].set_ylabel('Gradient Norm')
            axes[1].set_title('Gradient Norm Over Time')
            axes[1].grid(True, alpha=0.3)
        
        # Log scale cost (for better visualization)
        axes[2].plot(np.log(self.cost_history))
        axes[2].set_xlabel('Iteration')
        axes[2].set_ylabel('Log(Cost)')
        axes[2].set_title('Log Cost Function')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_regression_line(self, X: np.ndarray, y: np.ndarray):
        """For simple linear regression (one feature), plot the fit."""
        if X.shape[1] != 1:
            print("Can only plot regression line for single feature!")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot of data
        plt.scatter(X, y, alpha=0.6, label='Data points')
        
        # Regression line
        X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_line = self.predict(X_line)
        plt.plot(X_line, y_line, 'r-', linewidth=2, 
                label=f'Regression line: y = {self.intercept:.2f} + {self.coefficients[0]:.2f}x')
        
        plt.xlabel('Feature')
        plt.ylabel('Target')
        plt.title('Linear Regression Fit')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class PolynomialRegression(LinearRegression):
    """
    Polynomial Regression by adding polynomial features
    before applying linear regression.
    """
    
    def __init__(self, degree: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.degree = degree
        self.poly_features = None
        
    def _create_polynomial_features(self, X: np.ndarray) -> np.ndarray:
        """Create polynomial features up to specified degree."""
        from sklearn.preprocessing import PolynomialFeatures
        self.poly_features = PolynomialFeatures(degree=self.degree, 
                                                include_bias=False)
        return self.poly_features.fit_transform(X)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'PolynomialRegression':
        """Fit polynomial regression model."""
        X_poly = self._create_polynomial_features(X)
        super().fit(X_poly, y, **kwargs)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with polynomial features."""
        if self.poly_features is None:
            raise ValueError("Model must be fitted first!")
        X_poly = self.poly_features.transform(X)
        return super().predict(X_poly)