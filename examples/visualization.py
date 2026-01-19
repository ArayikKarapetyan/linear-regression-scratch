"""
Simple visualization for Linear Regression
Customize this with your own results!
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add path to import your linear regression
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import YOUR linear regression
try:
    from src.linear_regression import LinearRegression
    print("✅ Successfully imported your LinearRegression")
except ImportError as e:
    print(f"❌ Error importing: {e}")
    print("Please make sure your linear_regression.py is in the src folder")
    sys.exit(1)

def create_simple_visualization():
    """Create basic visualization with your implementation"""
    
    # Generate simple data
    np.random.seed(42)
    X = 2 * np.random.rand(50, 1)
    y = 4 + 3 * X + np.random.randn(50, 1) * 0.5
    
    # Flatten y for your implementation
    y_flat = y.flatten()
    
    # Create figure
    plt.figure(figsize=(12, 4))
    
    # 1. Train and plot Gradient Descent
    plt.subplot(1, 3, 1)
    
    model_gd = LinearRegression(method='gradient_descent', 
                                learning_rate=0.1,
                                n_iterations=500)
    model_gd.fit(X, y_flat)
    
    # Scatter plot
    plt.scatter(X, y, alpha=0.6, label='Data points')
    
    # Regression line
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model_gd.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, 
             label=f'y = {model_gd.intercept:.2f} + {model_gd.coefficients[0]:.2f}x')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Gradient Descent Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Learning Curve
    plt.subplot(1, 3, 2)
    if hasattr(model_gd, 'cost_history') and model_gd.cost_history:
        plt.plot(model_gd.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Learning Curve')
        plt.grid(True, alpha=0.3)
        
        # Annotate final cost
        plt.annotate(f'Final Cost: {model_gd.cost_history[-1]:.4f}', 
                    xy=(0.6, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Compare with Normal Equation
    plt.subplot(1, 3, 3)
    
    model_ne = LinearRegression(method='normal_equation')
    model_ne.fit(X, y_flat)
    
    plt.scatter(X, y, alpha=0.6, label='Data points')
    
    # Both regression lines
    y_line_gd = model_gd.predict(X_line)
    y_line_ne = model_ne.predict(X_line)
    
    plt.plot(X_line, y_line_gd, 'r-', linewidth=2, label='Gradient Descent')
    plt.plot(X_line, y_line_ne, 'g--', linewidth=2, label='Normal Equation')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Method Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('assets/regression_visualization.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved to assets/regression_visualization.png")
    plt.show()
    
    # Print results
    print("\n📊 Results:")
    print("-" * 40)
    print(f"Gradient Descent:")
    print(f"  θ₀ = {model_gd.intercept:.4f}, θ₁ = {model_gd.coefficients[0]:.4f}")
    print(f"  R² = {model_gd.r2_score(X, y_flat):.4f}")
    
    print(f"\nNormal Equation:")
    print(f"  θ₀ = {model_ne.intercept:.4f}, θ₁ = {model_ne.coefficients[0]:.4f}")
    print(f"  R² = {model_ne.r2_score(X, y_flat):.4f}")

def create_cost_function_3d():
    """Create 3D visualization of cost function (optional)"""
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Generate data
        np.random.seed(42)
        X = 2 * np.random.rand(20, 1)
        y = 4 + 3 * X + np.random.randn(20, 1)
        
        # Create grid of theta values
        theta0_vals = np.linspace(0, 8, 50)
        theta1_vals = np.linspace(0, 6, 50)
        theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)
        
        # Compute cost for each combination
        cost_grid = np.zeros_like(theta0_grid)
        m = len(X)
        
        for i in range(len(theta0_vals)):
            for j in range(len(theta1_vals)):
                predictions = theta0_grid[i, j] + theta1_grid[i, j] * X
                cost_grid[i, j] = (1/(2*m)) * np.sum((predictions - y) ** 2)
        
        # Plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(theta0_grid, theta1_grid, cost_grid, 
                              cmap='viridis', alpha=0.8)
        
        # Mark optimal point (true values)
        ax.scatter([4], [3], [0], color='red', s=100, label='True parameters')
        
        ax.set_xlabel('θ₀ (Intercept)')
        ax.set_ylabel('θ₁ (Slope)')
        ax.set_zlabel('Cost J(θ)')
        ax.set_title('Cost Function Surface')
        plt.legend()
        
        plt.savefig('assets/cost_function_3d.png', dpi=300, bbox_inches='tight')
        print("✅ 3D Visualization saved to assets/cost_function_3d.png")
        plt.show()
        
    except ImportError:
        print("⚠️  3D plotting requires mpl_toolkits. Skipping 3D visualization.")

if __name__ == "__main__":
    print("Creating visualizations for your Linear Regression implementation...")
    
    # Create assets folder if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Create visualizations
    create_simple_visualization()
    
    # Uncomment if you want 3D plot
    # create_cost_function_3d()
    
    print("\n🎉 All visualizations created! Check the 'assets' folder.")