"""
Basic usage example for your Linear Regression
"""

import numpy as np
import sys
import os

# Add path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.linear_regression import LinearRegression
    print("✅ Import successful!")
except ImportError:
    print("❌ Could not import LinearRegression")
    print("Make sure linear_regression.py is in the src folder")
    sys.exit(1)

def main():
    """Basic demonstration"""
    
    # 1. Simple example
    print("=" * 50)
    print("SIMPLE LINEAR REGRESSION")
    print("=" * 50)
    
    # Create simple data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 5, 4, 5])
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    # 2. Using Gradient Descent
    print("\n--- Gradient Descent ---")
    model_gd = LinearRegression(method='gradient_descent', 
                                learning_rate=0.01,
                                n_iterations=1000)
    model_gd.fit(X, y)
    
    print(f"Parameters: θ₀ = {model_gd.intercept:.3f}, θ₁ = {model_gd.coefficients[0]:.3f}")
    print(f"R² Score: {model_gd.r2_score(X, y):.3f}")
    
    # 3. Using Normal Equation
    print("\n--- Normal Equation ---")
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X, y)
    
    print(f"Parameters: θ₀ = {model_ne.intercept:.3f}, θ₁ = {model_ne.coefficients[0]:.3f}")
    print(f"R² Score: {model_ne.r2_score(X, y):.3f}")
    
    # 4. Make predictions
    print("\n--- Predictions ---")
    X_new = np.array([[1.5], [2.5], [3.5]])
    predictions = model_gd.predict(X_new)
    
    for x, y_pred in zip(X_new, predictions):
        print(f"X = {x[0]:.1f} → Predicted y = {y_pred:.2f}")
    
    # 5. Compare methods
    print("\n--- Method Comparison ---")
    print(f"Parameter difference:")
    print(f"  θ₀: {abs(model_gd.intercept - model_ne.intercept):.6f}")
    print(f"  θ₁: {abs(model_gd.coefficients[0] - model_ne.coefficients[0]):.6f}")
    
    if hasattr(model_gd, 'cost_history'):
        print(f"\nGradient Descent converged in {len(model_gd.cost_history)} iterations")
        print(f"Final cost: {model_gd.cost_history[-1]:.6f}")

if __name__ == "__main__":
    main()