"""
Normal Equation Examples
Closed-form solution and its advantages/limitations
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import LinearRegression

def normal_equation_advantages():
    """Show advantages of normal equation"""
    print("=" * 60)
    print("NORMAL EQUATION: Advantages")
    print("=" * 60)
    
    # Generate small dataset where normal equation excels
    np.random.seed(42)
    X = np.array([
        [1, 50],    # Study hours, IQ
        [2, 55],
        [3, 60],
        [4, 65],
        [5, 70],
    ])
    y = np.array([60, 65, 70, 75, 80])  # Exam scores
    
    print("Dataset: [Study Hours, IQ] → Exam Score")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # Normal Equation (exact solution)
    model_ne = LinearRegression(method='normal_equation')
    start_time = time.time()
    model_ne.fit(X, y)
    ne_time = time.time() - start_time
    
    # Gradient Descent (approximate)
    model_gd = LinearRegression(method='gradient_descent',
                               learning_rate=0.01,
                               n_iterations=10000)
    start_time = time.time()
    model_gd.fit(X, y)
    gd_time = time.time() - start_time
    
    print(f"\nNormal Equation:")
    print(f"  Time: {ne_time:.6f} seconds")
    print(f"  θ₀ (intercept): {model_ne.intercept:.6f}")
    print(f"  θ₁ (study hours): {model_ne.coefficients[0]:.6f}")
    print(f"  θ₂ (IQ): {model_ne.coefficients[1]:.6f}")
    
    print(f"\nGradient Descent (after {model_gd.n_iterations} iterations):")
    print(f"  Time: {gd_time:.6f} seconds")
    print(f"  θ₀ (intercept): {model_gd.intercept:.6f}")
    print(f"  θ₁ (study hours): {model_gd.coefficients[0]:.6f}")
    print(f"  θ₂ (IQ): {model_gd.coefficients[1]:.6f}")
    
    print(f"\nParameter Differences:")
    print(f"  θ₀ diff: {abs(model_ne.intercept - model_gd.intercept):.6f}")
    print(f"  θ₁ diff: {abs(model_ne.coefficients[0] - model_gd.coefficients[0]):.6f}")
    print(f"  θ₂ diff: {abs(model_ne.coefficients[1] - model_gd.coefficients[1]):.6f}")
    
    print(f"\nR² Scores (identical):")
    print(f"  Normal Equation: {model_ne.r2_score(X, y):.6f}")
    print(f"  Gradient Descent: {model_gd.r2_score(X, y):.6f}")

def singular_matrix_issue():
    """Demonstrate issue with singular/non-invertible matrices"""
    print("\n" + "=" * 60)
    print("SINGULAR MATRIX ISSUE")
    print("=" * 60)
    
    # Create dataset with perfect multicollinearity
    np.random.seed(42)
    X = np.random.randn(100, 2)
    X[:, 1] = X[:, 0] * 2  # Perfect linear relationship between features
    y = 3 + 2*X[:, 0] + 1.5*X[:, 1] + np.random.randn(100)*0.1
    
    print("Features: x2 = 2*x1 (perfect multicollinearity)")
    print("X shape:", X.shape)
    print("Correlation between x1 and x2:", np.corrcoef(X[:, 0], X[:, 1])[0, 1])
    
    try:
        # This should fail with normal equation
        model = LinearRegression(method='normal_equation')
        model.fit(X, y)
        print("\nNormal equation succeeded (unexpected!)")
    except np.linalg.LinAlgError as e:
        print(f"\nNormal Equation Failed: {e}")
        print("This is expected because XᵀX is singular (non-invertible)")
    
    # Show that gradient descent still works
    model_gd = LinearRegression(method='gradient_descent',
                               learning_rate=0.01,
                               n_iterations=1000)
    model_gd.fit(X, y)
    
    print(f"\n✓ Gradient Descent still works:")
    print(f"  θ₀: {model_gd.intercept:.4f}")
    print(f"  θ₁: {model_gd.coefficients[0]:.4f}")
    print(f"  θ₂: {model_gd.coefficients[1]:.4f}")
    print(f"  R²: {model_gd.r2_score(X, y):.4f}")
    
    # Add regularization to fix normal equation
    print("\n" + "=" * 40)
    print("SOLUTION: Add Regularization (Ridge Regression)")
    print("=" * 40)
    
    model_ridge = LinearRegression(method='normal_equation',
                                  regularization=0.1)  # Small lambda
    model_ridge.fit(X, y)
    
    print(f"With regularization (λ=0.1):")
    print(f"  θ₀: {model_ridge.intercept:.4f}")
    print(f"  θ₁: {model_ridge.coefficients[0]:.4f}")
    print(f"  θ₂: {model_ridge.coefficients[1]:.4f}")
    print(f"  R²: {model_ridge.r2_score(X, y):.4f}")

def scaling_insensitivity():
    """Show that normal equation doesn't need feature scaling"""
    print("\n" + "=" * 60)
    print("FEATURE SCALING INSENSITIVITY")
    print("=" * 60)
    
    # Create features with very different scales
    np.random.seed(42)
    n_samples = 100
    
    # Feature 1: Small scale (0-1)
    X1 = np.random.rand(n_samples, 1)
    
    # Feature 2: Large scale (1000-2000)
    X2 = 1000 + 1000 * np.random.rand(n_samples, 1)
    
    # Feature 3: Very large scale (10000-20000)
    X3 = 10000 + 10000 * np.random.rand(n_samples, 1)
    
    X = np.hstack([X1, X2, X3])
    y = 5 + 0.5*X1.flatten() + 0.001*X2.flatten() + 0.0001*X3.flatten() + np.random.randn(n_samples)*0.1
    
    print("Feature scales:")
    print(f"  X1 range: [{X1.min():.2f}, {X1.max():.2f}]")
    print(f"  X2 range: [{X2.min():.2f}, {X2.max():.2f}]")
    print(f"  X3 range: [{X3.min():.2f}, {X3.max():.2f}]")
    print(f"  Ratio X3/X1: {X3.max()/X1.max():.0f}x difference")
    
    # Normal Equation (no scaling needed)
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X, y)
    
    # Gradient Descent (needs scaling)
    from sklearn.preprocessing import StandardScaler
    
    # Without scaling (poor convergence)
    model_gd_no_scale = LinearRegression(method='gradient_descent',
                                        learning_rate=0.000001,  # Very small LR
                                        n_iterations=1000)
    model_gd_no_scale.fit(X, y, verbose=False)
    
    # With scaling (better)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_gd_scaled = LinearRegression(method='gradient_descent',
                                      learning_rate=0.1,
                                      n_iterations=1000)
    model_gd_scaled.fit(X_scaled, y, verbose=False)
    
    print(f"\nNormal Equation Results (no scaling needed):")
    print(f"  θ₀: {model_ne.intercept:.4f}")
    print(f"  θ₁: {model_ne.coefficients[0]:.4f}")
    print(f"  θ₂: {model_ne.coefficients[1]:.4f}")
    print(f"  θ₃: {model_ne.coefficients[2]:.4f}")
    
    print(f"\nGradient Descent Without Scaling:")
    if model_gd_no_scale.cost_history:
        improvement = ((model_gd_no_scale.cost_history[0] - model_gd_no_scale.cost_history[-1]) / 
                      model_gd_no_scale.cost_history[0]) * 100
        print(f"  Cost improvement: {improvement:.1f}%")
        print(f"  Final cost: {model_gd_no_scale.cost_history[-1]:.4f}")
    
    print(f"\nGradient Descent With Scaling:")
    if model_gd_scaled.cost_history:
        improvement = ((model_gd_scaled.cost_history[0] - model_gd_scaled.cost_history[-1]) / 
                      model_gd_scaled.cost_history[0]) * 100
        print(f"  Cost improvement: {improvement:.1f}%")
        print(f"  Final cost: {model_gd_scaled.cost_history[-1]:.4f}")
    
    # Visualize scaling effect on convergence
    if model_gd_no_scale.cost_history and model_gd_scaled.cost_history:
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(model_gd_no_scale.cost_history, 'r-', label='No Scaling (α=1e-6)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('GD Without Feature Scaling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(model_gd_scaled.cost_history, 'g-', label='With Scaling (α=0.1)')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('GD With Feature Scaling')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def large_dataset_limitation():
    """Show normal equation doesn't scale to large features"""
    print("\n" + "=" * 60)
    print("SCALING LIMITATION: Large Number of Features")
    print("=" * 60)
    
    # Create dataset with increasing number of features
    np.random.seed(42)
    n_samples = 100
    
    feature_counts = [10, 50, 100, 500]
    times_ne = []
    times_gd = []
    
    for n_features in feature_counts:
        print(f"\nTesting with {n_features} features...")
        
        X = np.random.randn(n_samples, n_features)
        # True coefficients
        true_coeffs = np.random.randn(n_features + 1)
        true_coeffs[0] = 5  # Intercept
        
        # Generate y
        X_b = np.c_[np.ones((n_samples, 1)), X]  # Add intercept
        y = X_b.dot(true_coeffs) + np.random.randn(n_samples) * 0.1
        
        # Normal Equation
        model_ne = LinearRegression(method='normal_equation')
        start = time.time()
        try:
            model_ne.fit(X, y)
            ne_time = time.time() - start
            times_ne.append(ne_time)
            print(f"  Normal Equation: {ne_time:.4f} seconds")
        except MemoryError:
            print(f"  Normal Equation: Memory Error!")
            times_ne.append(float('inf'))
        
        # Gradient Descent (few iterations)
        model_gd = LinearRegression(method='gradient_descent',
                                   learning_rate=0.01,
                                   n_iterations=100)
        start = time.time()
        model_gd.fit(X, y, verbose=False)
        gd_time = time.time() - start
        times_gd.append(gd_time)
        print(f"  Gradient Descent: {gd_time:.4f} seconds")
    
    # Plot scaling
    plt.figure(figsize=(10, 5))
    
    plt.plot(feature_counts[:len(times_ne)], times_ne, 'ro-', label='Normal Equation', linewidth=2)
    plt.plot(feature_counts[:len(times_gd)], times_gd, 'bo-', label='Gradient Descent (100 iter)', linewidth=2)
    
    plt.xlabel('Number of Features')
    plt.ylabel('Training Time (seconds)')
    plt.title('Scaling: Normal Equation vs Gradient Descent')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add theoretical complexity
    x_theory = np.array(feature_counts)
    y_ne_theory = x_theory**3 / 1e7  # O(n³) scaled
    y_gd_theory = x_theory / 100     # O(n) scaled
    
    plt.plot(x_theory, y_ne_theory, 'r--', alpha=0.5, label='O(n³) trend')
    plt.plot(x_theory, y_gd_theory, 'b--', alpha=0.5, label='O(n) trend')
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("Normal Equation is O(n³) - doesn't scale well with many features")
    print("Gradient Descent is O(n) - scales linearly with features")

if __name__ == "__main__":
    normal_equation_advantages()
    singular_matrix_issue()
    scaling_insensitivity()
    large_dataset_limitation()