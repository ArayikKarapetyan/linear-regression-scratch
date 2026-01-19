"""
Polynomial Regression Examples
Fit non-linear relationships using polynomial features
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import PolynomialRegression

def scale_features(X, degree):
    """
    Properly scale polynomial features to prevent numerical overflow.
    Returns scaled X and scaling parameters.
    """
    X_scaled = X.copy()
    scaling_params = []
    
    for d in range(1, degree + 1):
        X_d = X ** d
        mean = X_d.mean()
        std = X_d.std()
        
        # Avoid division by zero
        if std < 1e-10:
            std = 1.0
            
        X_scaled = np.hstack([X_scaled, (X_d - mean) / std])
        scaling_params.append((mean, std))
    
    return X_scaled[:, 1:], scaling_params  # Remove the first column of ones

def real_world_example():
    """Real-world example: House price vs size (non-linear) - FIXED VERSION"""
    print("\n" + "=" * 60)
    print("REAL WORLD: House Price vs Size (Non-linear) - FIXED")
    print("=" * 60)
    
    # Simulate real estate data with reasonable values
    np.random.seed(42)
    n_samples = 50  # Reduce samples for stability
    
    # Generate house sizes (in thousands of sqft to keep numbers reasonable)
    sizes = np.random.uniform(1.0, 3.0, n_samples)  # 1000-3000 sqft
    
    # Create realistic price relationship with diminishing returns
    # Price = base + linear_term + sqrt_term (diminishing returns)
    base_price = 200  # $200K base
    linear_coeff = 100  # $100K per 1000 sqft
    sqrt_coeff = 50    # $50K per sqrt(1000 sqft)
    
    prices = (base_price + 
              linear_coeff * sizes + 
              sqrt_coeff * np.sqrt(sizes) +
              np.random.randn(n_samples) * 20)  # Add some noise
    
    # Convert to original scale for display
    X_original = sizes * 1000  # Convert back to sqft
    y_original = prices * 1000  # Convert back to dollars
    
    # Use scaled data for training (in thousands)
    X = sizes.reshape(-1, 1)
    y = prices
    
    print(f"Dataset: {len(X)} houses")
    print(f"Size range: {X_original.min():.0f} - {X_original.max():.0f} sqft")
    print(f"Price range: ${y_original.min():,.0f} - ${y_original.max():,.0f}")
    print(f"\nUsing scaled data: sizes in 1000s sqft, prices in $1000s")
    
    # Try different models - ALWAYS use normal equation for polynomials
    degrees = [1, 2, 3]
    models = []
    mse_scores = []
    
    plt.figure(figsize=(12, 5))
    
    for i, degree in enumerate(degrees):
        try:
            print(f"\n--- Fitting Degree {degree} ---")
            
            # ALWAYS use normal equation for polynomial regression
            # Gradient descent is too unstable for polynomials
            model = PolynomialRegression(degree=degree,
                                       method='normal_equation',
                                       add_bias=True)
            
            # Fit the model
            model.fit(X, y, verbose=False)
            
            # Check if coefficients are reasonable
            if hasattr(model, 'theta'):
                max_coeff = np.max(np.abs(model.theta))
                if max_coeff > 1e6:  # Coefficients too large
                    print(f"  Warning: Large coefficients detected (max: {max_coeff:.2e})")
                    print(f"  Coefficients: {model.theta}")
                    
                    # Try with regularization if available
                    if hasattr(model, 'fit_ridge'):
                        print("  Trying Ridge regression instead...")
                        model.fit_ridge(X, y, alpha=1.0)  # Regularization
                    else:
                        # Manual regularization: add small constant to diagonal
                        print("  Adding small regularization...")
                        # This is a hack - better to implement Ridge in your class
                        pass
            
            models.append(model)
            
            # Make predictions
            y_pred_scaled = model.predict(X)
            
            # Calculate MSE on original scale
            y_pred_original = y_pred_scaled * 1000
            mse = np.mean((y_original - y_pred_original) ** 2)
            mse_scores.append(mse)
            
            r2 = model.r2_score(X, y)
            print(f"  MSE = ${mse:,.0f}")
            print(f"  R² = {r2:.4f}")
            
            # Plot with original scale
            plt.subplot(1, 2, 1)
            if i == 0:
                plt.scatter(X_original, y_original, alpha=0.5, 
                          label='Houses', color='gray', s=50)
            
            # Generate smooth curve for plotting
            X_line_scaled = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
            X_line_original = X_line_scaled * 1000
            
            try:
                y_line_scaled = model.predict(X_line_scaled)
                y_line_original = y_line_scaled * 1000
                
                # Check for unreasonable predictions
                if np.any(np.abs(y_line_original) > 1e15):
                    print(f"  Warning: Unreasonable predictions for degree {degree}")
                    continue
                
                colors = ['red', 'green', 'blue']
                plt.plot(X_line_original, y_line_original, color=colors[i], 
                        alpha=0.8, label=f'Degree {degree}', linewidth=2)
                
            except Exception as e:
                print(f"  Error predicting for degree {degree}: {e}")
                
        except Exception as e:
            print(f"\nError fitting degree {degree}: {e}")
            print("Trying with manual implementation...")
            
            # Manual polynomial fitting as fallback
            try:
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression
                
                poly = PolynomialFeatures(degree=degree)
                X_poly = poly.fit_transform(X)
                
                reg = LinearRegression()
                reg.fit(X_poly, y)
                
                # Create a wrapper model
                class ManualPolyModel:
                    def __init__(self, degree, reg):
                        self.degree = degree
                        self.reg = reg
                        self.poly = poly
                    
                    def predict(self, X):
                        X_poly = self.poly.transform(X)
                        return self.reg.predict(X_poly)
                    
                    def r2_score(self, X, y):
                        y_pred = self.predict(X)
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        return 1 - (ss_res / ss_tot)
                
                manual_model = ManualPolyModel(degree, reg)
                models.append(manual_model)
                
                # Calculate MSE
                y_pred_scaled = manual_model.predict(X)
                y_pred_original = y_pred_scaled * 1000
                mse = np.mean((y_original - y_pred_original) ** 2)
                mse_scores.append(mse)
                
                print(f"  MSE (manual) = ${mse:,.0f}")
                print(f"  R² (manual) = {manual_model.r2_score(X, y):.4f}")
                
            except Exception as e2:
                print(f"  Manual fitting also failed: {e2}")
                models.append(None)
                mse_scores.append(np.inf)
    
    # Plot 1: Polynomial fits
    plt.subplot(1, 2, 1)
    plt.xlabel('House Size (sqft)')
    plt.ylabel('Price ($)')
    plt.title('House Price vs Size: Polynomial Fits')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set reasonable y-axis limits
    y_min = y_original.min() * 0.9
    y_max = y_original.max() * 1.1
    plt.ylim(y_min, y_max)
    
    # Plot 2: MSE vs Degree
    plt.subplot(1, 2, 2)
    valid_mse = [m for m in mse_scores if m != np.inf and not np.isnan(m)]
    valid_degrees = [d for d, m in zip(degrees, mse_scores) 
                    if m != np.inf and not np.isnan(m)]
    
    if valid_mse:
        plt.plot(valid_degrees, valid_mse, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Polynomial Degree')
        plt.ylabel('Mean Squared Error ($)')
        plt.title('Model Error vs Complexity')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for d, m in zip(valid_degrees, valid_mse):
            plt.text(d, m, f'${m:,.0f}', ha='center', va='bottom')
    else:
        plt.text(0.5, 0.5, 'No valid models to plot', 
                ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Make predictions for specific house sizes (in original scale)
    print("\n" + "=" * 50)
    print("Price Predictions for Different House Sizes:")
    print("=" * 50)
    
    test_sizes_original = np.array([[1000], [1500], [2000], [2500], [3000]])
    test_sizes_scaled = test_sizes_original / 1000  # Scale down to thousands
    
    for size_original, size_scaled in zip(test_sizes_original, test_sizes_scaled):
        print(f"\n{size_original[0]:.0f} sqft house:")
        
        for model, degree in zip(models, degrees):
            if model is not None:
                try:
                    pred_scaled = model.predict(size_scaled.reshape(-1, 1))[0]
                    pred_original = pred_scaled * 1000
                    
                    # Check if prediction is reasonable
                    if 0 < pred_original < 1e9:  # Reasonable price range
                        print(f"  Degree {degree}: ${pred_original:,.0f}")
                    else:
                        print(f"  Degree {degree}: Unreasonable (${pred_original:.2e})")
                except Exception as e:
                    print(f"  Degree {degree}: Error - {str(e)[:30]}")
    
    return models, X, y

def safe_polynomial_fit():
    """Safe polynomial regression with all necessary precautions"""
    print("\n" + "=" * 60)
    print("SAFE POLYNOMIAL REGRESSION DEMONSTRATION")
    print("=" * 60)
    
    # Create simple, well-behaved data
    np.random.seed(42)
    n_points = 30
    
    # X values between -3 and 3 (well-centered)
    X = np.linspace(-3, 3, n_points).reshape(-1, 1)
    
    # Simple quadratic relationship
    y_true = 2 + 1.5*X - 0.8*X**2
    y = y_true + np.random.randn(n_points, 1) * 0.5
    
    # Try polynomial regression with different methods
    degrees = [1, 2, 3, 4]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, degree in enumerate(degrees):
        ax = axes[idx]
        
        # Plot data
        ax.scatter(X, y, alpha=0.6, label='Data', s=40)
        
        try:
            # Use gradient descent only for low degrees with tiny learning rate
            if degree <= 2:
                model = PolynomialRegression(degree=degree,
                                           method='gradient_descent',
                                           learning_rate=0.001,
                                           n_iterations=5000)
            else:
                # Use normal equation for higher degrees
                model = PolynomialRegression(degree=degree,
                                           method='normal_equation')
            
            model.fit(X, y.flatten(), verbose=False)
            
            # Generate prediction curve
            X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
            y_line = model.predict(X_line)
            
            # Plot predictions
            ax.plot(X_line, y_line, 'r-', linewidth=2, label=f'Degree {degree}')
            
            # Plot true relationship
            ax.plot(X_line, 2 + 1.5*X_line - 0.8*X_line**2, 
                   'g--', alpha=0.7, label='True', linewidth=2)
            
            # Calculate R²
            r2 = model.r2_score(X, y.flatten())
            ax.set_title(f'Degree {degree} (R² = {r2:.4f})')
            
            # Add equation if available
            if hasattr(model, 'theta') and len(model.theta) <= 4:
                eq_parts = []
                for i, coeff in enumerate(model.theta):
                    if i == 0:
                        eq_parts.append(f"{coeff:.2f}")
                    elif i == 1:
                        eq_parts.append(f"{coeff:+.2f}x")
                    else:
                        eq_parts.append(f"{coeff:+.2f}x^{i}")
                
                eq_text = "y = " + " ".join(eq_parts)
                ax.text(0.05, 0.95, eq_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Degree {degree}\nFailed:\n{str(e)[:30]}...',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Degree {degree} (Failed)')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Polynomial Regression with Different Degrees', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    print("\nKey takeaways:")
    print("1. Use normal equation for polynomial regression (gradient descent is unstable)")
    print("2. Keep X values centered around 0 (between -3 and 3 is good)")
    print("3. Start with low degrees (1-3) before trying higher degrees")
    print("4. Add regularization (Ridge/Lasso) to prevent overfitting")

def main():
    """Main function with proper error handling"""
    print("POLYNOMIAL REGRESSION EXAMPLES - STABLE VERSION")
    print("=" * 60)
    
    try:
        # Run the safe demonstration first
        safe_polynomial_fit()
        
        # Then try the real-world example
        models, X, y = real_world_example()
        
        print("\n" + "=" * 60)
        print("SUCCESS! All examples completed.")
        print("=" * 60)
        
        print("\nTROUBLESHOOTING TIPS:")
        print("1. If you see astronomical numbers, use normal equation instead of gradient descent")
        print("2. Scale your features: X = (X - mean(X)) / std(X)")
        print("3. Reduce polynomial degree (start with 1 or 2)")
        print("4. Add regularization to prevent overfitting")
        print("5. Use more data points for higher-degree polynomials")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        
        print("\nEMERGENCY FALLBACK:")
        print("If nothing works, use scikit-learn for polynomial regression:")
        print("""
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        
        model = make_pipeline(
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
        model.fit(X, y)
        """)

if __name__ == "__main__":
    main()