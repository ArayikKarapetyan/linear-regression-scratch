"""
Basic Linear Regression Usage
Simple examples to get started
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import LinearRegression, PolynomialRegression

def example_1_simple_linear():
    """Simple linear relationship"""
    print("=" * 60)
    print("EXAMPLE 1: Simple Linear Relationship")
    print("=" * 60)
    
    # Generate data: y = 2x + 1 + noise
    np.random.seed(42)
    X = np.array([[1], [2], [3], [4], [5], [6]])
    y = np.array([3.1, 5.2, 7.1, 8.8, 10.9, 12.8])
    
    # Fit model
    model = LinearRegression(method='gradient_descent', 
                            learning_rate=0.01,
                            n_iterations=1000)
    model.fit(X, y)
    
    print(f"True relationship: y = 1 + 2x")
    print(f"Estimated: y = {model.intercept:.3f} + {model.coefficients[0]:.3f}x")
    print(f"R² Score: {model.r2_score(X, y):.4f}")
    
    # Predict
    X_new = np.array([[2.5], [3.5], [4.5]])
    predictions = model.predict(X_new)
    print(f"\nPredictions:")
    for x, y_pred in zip(X_new, predictions):
        print(f"  X = {x[0]}: ŷ = {y_pred:.2f}")
    
    return model, X, y

def example_2_multiple_features():
    """Multiple linear regression"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multiple Features")
    print("=" * 60)
    
    # House price prediction example
    # Features: [size(sqft), bedrooms, age]
    X = np.array([
        [1400, 3, 5],   # House 1
        [1600, 3, 3],   # House 2
        [1700, 4, 2],   # House 3
        [1875, 4, 4],   # House 4
        [1100, 2, 8],   # House 5
        [1550, 3, 6],   # House 6
        [2350, 4, 1],   # House 7
        [2450, 5, 2],   # House 8
    ])
    
    # Prices in $1000s
    y = np.array([300, 350, 375, 400, 250, 325, 475, 500])
    
    # Normalize features for better convergence
    X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
    
    model = LinearRegression(method='gradient_descent',
                            learning_rate=0.01,  # Reduced from 0.1
                            n_iterations=2000)
    model.fit(X_normalized, y)
    
    print(f"Features: [Size(sqft), Bedrooms, Age]")
    print(f"Intercept (base price): ${model.intercept:.2f}K")
    print(f"Coefficients: {model.coefficients}")
    print(f"Interpretation: Each std increase in size adds ${model.coefficients[0]:.2f}K")
    print(f"R² Score: {model.r2_score(X_normalized, y):.4f}")
    
    # Predict price for a new house
    new_house = np.array([[1800, 3, 5]])  # 1800 sqft, 3 bedrooms, 5 years old
    new_house_normalized = (new_house - X.mean(axis=0)) / X.std(axis=0)
    predicted_price = model.predict(new_house_normalized)[0]
    
    print(f"\nNew house prediction:")
    print(f"  Features: {new_house[0]}")
    print(f"  Predicted price: ${predicted_price:.2f}K (${predicted_price*1000:.0f})")
    
    return model, X, y

def example_3_method_comparison():
    """Compare gradient descent vs normal equation"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Method Comparison")
    print("=" * 60)
    
    # Generate data - FIXED: Make X values smaller
    np.random.seed(42)
    X = 2 * np.random.rand(50, 1)  # Changed from 10 to 2
    y = 2 + 3*X + np.random.randn(50, 1)*0.5
    
    # IMPORTANT: Add column of ones for intercept
    X_with_intercept = np.c_[np.ones((X.shape[0], 1)), X]
    
    # Gradient Descent - with MUCH smaller learning rate
    model_gd = LinearRegression(method='gradient_descent',
                               learning_rate=0.001,  # CRITICAL FIX: Reduced from 0.1
                               n_iterations=5000)    # Increased iterations
    model_gd.fit(X, y.flatten())
    
    # Normal Equation
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X, y.flatten())
    
    print(f"{'Method':<20} {'Intercept':<10} {'Slope':<10} {'R²':<10} {'MSE':<10}")
    print("-" * 60)
    
    for name, model in [("Gradient Descent", model_gd), ("Normal Equation", model_ne)]:
        r2 = model.r2_score(X, y.flatten())
        mse = model.mse(X, y.flatten())
        print(f"{name:<20} {model.intercept:<10.4f} {model.coefficients[0]:<10.4f} "
              f"{r2:<10.4f} {mse:<10.4f}")
    
    # Check if results are reasonable
    print("\n" + "-" * 60)
    print("ANALYSIS:")
    
    if abs(model_gd.intercept - model_ne.intercept) > 0.1:
        print("WARNING: Gradient descent not converging properly!")
        print(f"  Intercept difference: {abs(model_gd.intercept - model_ne.intercept):.4f}")
        
    if abs(model_gd.coefficients[0] - model_ne.coefficients[0]) > 0.1:
        print("WARNING: Gradient descent slopes don't match!")
        print(f"  Slope difference: {abs(model_gd.coefficients[0] - model_ne.coefficients[0]):.4f}")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_line, model_gd.predict(X_line), 'r-', linewidth=2, label='Gradient Descent')
    plt.plot(X_line, model_ne.predict(X_line), 'g--', linewidth=2, label='Normal Equation')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if hasattr(model_gd, 'cost_history') and model_gd.cost_history:
        plt.plot(model_gd.cost_history, 'b-')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Learning Curve (Gradient Descent)')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale to better see convergence
    else:
        plt.text(0.5, 0.5, 'No cost history available', 
                horizontalalignment='center', verticalalignment='center',
                transform=plt.gca().transAxes)
        plt.title('Learning Curve (Not Available)')
    
    plt.tight_layout()
    plt.show()
    
    return model_gd, model_ne, X, y

def example_4_fixed_gradient_descent():
    """Demonstrate working gradient descent with proper setup"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Fixed Gradient Descent with Feature Scaling")
    print("=" * 60)
    
    # Generate more realistic data
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 1) * 2  # Mean 0, std 2
    y = 2.5 + 1.8 * X + np.random.randn(n_samples, 1) * 0.5
    
    # Scale the features (CRITICAL for gradient descent)
    X_mean = X.mean()
    X_std = X.std()
    X_scaled = (X - X_mean) / X_std
    
    # Also scale y for stability (optional but helpful)
    y_mean = y.mean()
    y_std = y.std()
    y_scaled = (y - y_mean) / y_std
    
    # Gradient Descent on scaled data
    model_gd = LinearRegression(method='gradient_descent',
                               learning_rate=0.01,
                               n_iterations=10000)
    model_gd.fit(X_scaled, y_scaled.flatten())
    
    # Convert back to original scale
    # Transform coefficients back
    model_gd.intercept = model_gd.intercept * y_std + y_mean - np.sum(model_gd.coefficients * y_std * X_mean / X_std)
    model_gd.coefficients = model_gd.coefficients * y_std / X_std
    
    # Normal Equation on original data
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X, y.flatten())
    
    print(f"{'Method':<20} {'Intercept':<10} {'Slope':<10} {'R²':<10} {'MSE':<10}")
    print("-" * 60)
    
    for name, model in [("Gradient Descent", model_gd), ("Normal Equation", model_ne)]:
        r2 = model.r2_score(X, y.flatten())
        mse = model.mse(X, y.flatten())
        print(f"{name:<20} {model.intercept:<10.4f} {model.coefficients[0]:<10.4f} "
              f"{r2:<10.4f} {mse:<10.4f}")
    
    # Visualize learning
    if hasattr(model_gd, 'cost_history') and model_gd.cost_history:
        plt.figure(figsize=(8, 4))
        plt.plot(model_gd.cost_history[:1000])  # First 1000 iterations
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Gradient Descent Convergence (First 1000 iterations)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    return model_gd, model_ne, X, y

if __name__ == "__main__":
    print("BASIC LINEAR REGRESSION EXAMPLES")
    print("=" * 60)
    
    # Run examples
    try:
        model1, X1, y1 = example_1_simple_linear()
        model2, X2, y2 = example_2_multiple_features()
        model_gd, model_ne, X3, y3 = example_3_method_comparison()
        model_gd_fixed, model_ne_fixed, X4, y4 = example_4_fixed_gradient_descent()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your LinearRegression implementation")
        print("2. Make sure gradient descent has a small learning rate (0.001-0.01)")
        print("3. Add feature scaling if X values are large")
        print("4. Check that cost is decreasing with each iteration")