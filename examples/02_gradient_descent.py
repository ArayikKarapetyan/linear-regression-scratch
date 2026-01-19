"""
Gradient Descent Variations
Demonstrate different GD variants and learning rates
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import LinearRegression

def compare_learning_rates():
    """Compare different learning rates"""
    print("=" * 60)
    print("GRADIENT DESCENT: Learning Rate Comparison")
    print("=" * 60)
    
    # Generate data
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) * 0.5
    
    learning_rates = [0.001, 0.01, 0.1, 0.5, 1.0]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    plt.figure(figsize=(15, 5))
    
    for i, (lr, color) in enumerate(zip(learning_rates, colors)):
        model = LinearRegression(method='gradient_descent',
                                learning_rate=lr,
                                n_iterations=100)
        model.fit(X, y.flatten(), verbose=False)
        
        # Plot cost history
        plt.subplot(1, 3, 1)
        if model.cost_history:
            plt.plot(model.cost_history, color=color, label=f'α={lr}')
    
    plt.subplot(1, 3, 1)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Learning Rate Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show effects of too high learning rate
    plt.subplot(1, 3, 2)
    model_bad = LinearRegression(method='gradient_descent',
                                learning_rate=1.5,  # Too high!
                                n_iterations=20)
    model_bad.fit(X, y.flatten(), verbose=False)
    
    if model_bad.cost_history:
        plt.plot(model_bad.cost_history, 'r-', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Learning Rate Too High (Diverges)')
        plt.grid(True, alpha=0.3)
    
    # Show optimal learning rate
    plt.subplot(1, 3, 3)
    model_good = LinearRegression(method='gradient_descent',
                                 learning_rate=0.1,
                                 n_iterations=200)
    model_good.fit(X, y.flatten(), verbose=False)
    
    plt.scatter(X, y, alpha=0.6)
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    plt.plot(X_line, model_good.predict(X_line), 'r-', linewidth=2)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Optimal Fit (α=0.1)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print convergence analysis
    print("\nConvergence Analysis:")
    print("-" * 40)
    for lr in learning_rates:
        model = LinearRegression(method='gradient_descent',
                                learning_rate=lr,
                                n_iterations=100)
        model.fit(X, y.flatten(), verbose=False)
        
        if model.cost_history:
            initial_cost = model.cost_history[0]
            final_cost = model.cost_history[-1]
            improvement = ((initial_cost - final_cost) / initial_cost) * 100
            
            status = "✓ Converged" if final_cost < initial_cost else "✗ Diverged"
            print(f"α={lr:<6} {status:<15} Cost: {initial_cost:.4f} → {final_cost:.4f} "
                  f"({improvement:+.1f}%)")

def batch_vs_stochastic():
    """Compare batch, mini-batch, and stochastic GD"""
    print("\n" + "=" * 60)
    print("BATCH vs MINI-BATCH vs STOCHASTIC GRADIENT DESCENT")
    print("=" * 60)
    
    # Larger dataset
    np.random.seed(42)
    X = 2 * np.random.rand(1000, 1)
    y = 4 + 3 * X + np.random.randn(1000, 1) * 0.5
    
    methods = [
        ('Batch GD', None),        # Uses all data
        ('Mini-batch GD (32)', 32),  # Batch size 32
        ('Stochastic GD (1)', 1),  # Batch size 1
    ]
    
    plt.figure(figsize=(12, 4))
    
    for idx, (name, batch_size) in enumerate(methods):
        model = LinearRegression(method='gradient_descent',
                                learning_rate=0.1,
                                n_iterations=100,
                                batch_size=batch_size)
        model.fit(X, y.flatten(), verbose=False)
        
        # Plot cost history
        plt.subplot(1, 3, idx + 1)
        if model.cost_history:
            plt.plot(model.cost_history)
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title(f'{name}')
            plt.grid(True, alpha=0.3)
            
            # Add final values
            plt.annotate(f'Final: {model.cost_history[-1]:.4f}',
                        xy=(0.7, 0.9), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Print stats
        print(f"\n{name}:")
        print(f"  Final cost: {model.cost_history[-1]:.6f}")
        print(f"  Parameters: θ₀={model.intercept:.4f}, θ₁={model.coefficients[0]:.4f}")
        print(f"  R² Score: {model.r2_score(X, y.flatten()):.4f}")
    
    plt.tight_layout()
    plt.show()

def early_stopping_demo():
    """Demonstrate early stopping"""
    print("\n" + "=" * 60)
    print("EARLY STOPPING DEMONSTRATION")
    print("=" * 60)
    
    np.random.seed(42)
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1) * 0.5
    
    # With early stopping
    model_early = LinearRegression(method='gradient_descent',
                                  learning_rate=0.1,
                                  n_iterations=2000)
    model_early.fit(X, y.flatten(), early_stopping=True, tolerance=1e-6, verbose=False)
    
    # Without early stopping
    model_full = LinearRegression(method='gradient_descent',
                                 learning_rate=0.1,
                                 n_iterations=2000)
    model_full.fit(X, y.flatten(), early_stopping=False, verbose=False)
    
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(model_early.cost_history, 'b-', label='With Early Stopping')
    plt.plot(model_full.cost_history, 'r--', label='Full Training', alpha=0.7)
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title('Early Stopping vs Full Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Mark early stopping point
    if len(model_early.cost_history) < len(model_full.cost_history):
        plt.axvline(x=len(model_early.cost_history)-1, color='green', 
                   linestyle=':', label='Stopped Early')
    
    plt.subplot(1, 2, 2)
    # Zoom in on convergence
    plt.plot(model_early.cost_history[-100:], 'b-')
    plt.plot(model_full.cost_history[-100:], 'r--', alpha=0.7)
    plt.xlabel('Iteration (last 100)')
    plt.ylabel('Cost')
    plt.title('Zoom: Final Convergence')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nEarly Stopping Results:")
    print(f"  Stopped at iteration: {len(model_early.cost_history)}")
    print(f"  Final cost: {model_early.cost_history[-1]:.6f}")
    print(f"  Saved iterations: {len(model_full.cost_history) - len(model_early.cost_history)}")
    print(f"  Parameters match: {np.allclose(model_early.theta, model_full.theta, rtol=1e-3)}")

if __name__ == "__main__":
    compare_learning_rates()
    batch_vs_stochastic()
    early_stopping_demo()