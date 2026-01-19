"""
Polynomial Regression Examples - FIXED VERSION
Properly scaled and visualized
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import PolynomialRegression

def create_realistic_housing_data():
    """Create realistic housing data with proper scaling"""
    np.random.seed(42)
    n_samples = 50
    
    # House sizes in sqft (realistic range)
    sizes_sqft = np.random.uniform(800, 3000, n_samples)
    
    # Create realistic price relationship
    # Base price + linear term + diminishing returns (log/sqrt)
    base_price = 100000  # $100K base
    price_per_sqft = 150  # $150 per sqft
    
    # Prices with some non-linearity and noise
    prices = (base_price + 
              price_per_sqft * sizes_sqft +
              50 * np.sqrt(sizes_sqft) * 100 +  # Diminishing returns
              np.random.randn(n_samples) * 20000)  # Add noise
    
    return sizes_sqft, prices

def scale_data_for_training(X_raw, y_raw):
    """Scale data for training but keep track of scaling for inverse transform"""
    # Scale X to reasonable range (0-1 is good for polynomials)
    X_min, X_max = X_raw.min(), X_raw.max()
    X_scaled = (X_raw - X_min) / (X_max - X_min)
    
    # Scale y to reasonable range (also 0-1)
    y_min, y_max = y_raw.min(), y_raw.max()
    y_scaled = (y_raw - y_min) / (y_max - y_min)
    
    scaling_params = {
        'X_min': X_min, 'X_max': X_max,
        'y_min': y_min, 'y_max': y_max
    }
    
    return X_scaled.reshape(-1, 1), y_scaled, scaling_params

def inverse_transform_predictions(X_scaled, y_scaled_pred, scaling_params):
    """Convert scaled predictions back to original scale"""
    X_original = X_scaled * (scaling_params['X_max'] - scaling_params['X_min']) + scaling_params['X_min']
    y_original = y_scaled_pred * (scaling_params['y_max'] - scaling_params['y_min']) + scaling_params['y_min']
    return X_original, y_original

def real_world_example_fixed():
    """Real-world example: House price vs size - PROPERLY FIXED"""
    print("\n" + "=" * 60)
    print("HOUSE PRICE PREDICTION - PROPERLY SCALED")
    print("=" * 60)
    
    # Create realistic data
    sizes_sqft, prices = create_realistic_housing_data()
    
    print(f"Dataset: {len(sizes_sqft)} houses")
    print(f"Size range: {sizes_sqft.min():.0f} - {sizes_sqft.max():.0f} sqft")
    print(f"Price range: ${prices.min():,.0f} - ${prices.max():,.0f}")
    
    # Scale data for training
    X_scaled, y_scaled, scaling_params = scale_data_for_training(sizes_sqft, prices)
    
    # Try different polynomial degrees
    degrees = [1, 2, 3]
    models = []
    mse_scores = []
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Scatter plot of original data
    ax1.scatter(sizes_sqft, prices, alpha=0.6, color='gray', s=60, label='Houses')
    ax1.set_xlabel('House Size (sqft)', fontsize=12)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('House Price vs Size: Actual Data', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')  # Disable scientific notation
    
    colors = ['red', 'green', 'blue']
    
    for i, degree in enumerate(degrees):
        print(f"\n--- Fitting Degree {degree} Polynomial ---")
        
        try:
            # Always use normal equation for stability
            model = PolynomialRegression(degree=degree,
                                       method='normal_equation')
            
            # Fit on scaled data
            model.fit(X_scaled, y_scaled, verbose=False)
            
            # Generate predictions on scaled data
            X_line_scaled = np.linspace(0, 1, 300).reshape(-1, 1)
            y_line_scaled = model.predict(X_line_scaled)
            
            # Convert back to original scale for plotting
            X_line_original, y_line_original = inverse_transform_predictions(
                X_line_scaled, y_line_scaled, scaling_params
            )
            
            # Calculate MSE on original scale
            y_pred_scaled = model.predict(X_scaled)
            _, y_pred_original = inverse_transform_predictions(
                X_scaled, y_pred_scaled, scaling_params
            )
            
            mse = np.mean((prices - y_pred_original) ** 2)
            r2 = model.r2_score(X_scaled, y_scaled)
            
            mse_scores.append(mse)
            models.append(model)
            
            print(f"  MSE: ${mse:,.0f}")
            print(f"  R²:  {r2:.4f}")
            
            # Plot the polynomial fit
            ax1.plot(X_line_original, y_line_original, 
                    color=colors[i], linewidth=2.5, 
                    label=f'Degree {degree}', alpha=0.8)
            
        except Exception as e:
            print(f"  Error with degree {degree}: {e}")
            models.append(None)
            mse_scores.append(np.inf)
    
    ax1.legend(fontsize=11)
    
    # Set reasonable axis limits
    ax1.set_xlim(sizes_sqft.min() * 0.9, sizes_sqft.max() * 1.1)
    ax1.set_ylim(prices.min() * 0.8, prices.max() * 1.2)
    
    # Format y-axis as currency
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Plot 2: MSE vs Degree
    ax2.plot(degrees[:len(mse_scores)], mse_scores, 'bo-', 
            linewidth=3, markersize=10, markerfacecolor='red')
    ax2.set_xlabel('Polynomial Degree', fontsize=12)
    ax2.set_ylabel('Mean Squared Error ($)', fontsize=12)
    ax2.set_title('Model Error vs Complexity', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on the MSE plot
    for d, m in zip(degrees[:len(mse_scores)], mse_scores):
        if m != np.inf and not np.isnan(m):
            ax2.text(d, m, f'${m:,.0f}', 
                    ha='center', va='bottom', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('house_price_polynomial_fits.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Make predictions for specific house sizes
    print("\n" + "=" * 60)
    print("PRICE PREDICTIONS")
    print("=" * 60)
    
    test_sizes = np.array([[1000], [1500], [2000], [2500], [3000]])
    
    # Scale the test sizes
    test_sizes_scaled = (test_sizes - scaling_params['X_min']) / (scaling_params['X_max'] - scaling_params['X_min'])
    
    print(f"\n{'Size (sqft)':<12} {'Degree 1':<15} {'Degree 2':<15} {'Degree 3':<15}")
    print("-" * 60)
    
    for size_original, size_scaled in zip(test_sizes, test_sizes_scaled):
        predictions = []
        
        for model, degree in zip(models, degrees):
            if model is not None:
                try:
                    pred_scaled = model.predict(size_scaled.reshape(-1, 1))[0]
                    # Convert back to original scale
                    pred_original = pred_scaled * (scaling_params['y_max'] - scaling_params['y_min']) + scaling_params['y_min']
                    predictions.append(pred_original)
                except:
                    predictions.append(np.nan)
            else:
                predictions.append(np.nan)
        
        # Format the row
        row = f"{size_original[0]:<12,.0f}"
        for pred in predictions:
            if not np.isnan(pred):
                row += f" ${pred:,.0f}"
            else:
                row += " " * 15
        print(row)
    
    # Show some statistics
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    
    print(f"\n{'Degree':<10} {'MSE':<15} {'RMSE':<15} {'R²':<10}")
    print("-" * 50)
    
    for degree, model, mse in zip(degrees, models, mse_scores):
        if model is not None and mse != np.inf:
            rmse = np.sqrt(mse)
            r2 = model.r2_score(X_scaled, y_scaled)
            print(f"{degree:<10} ${mse:,.0f}  ${rmse:,.0f}  {r2:.4f}")
    
    return models, X_scaled, y_scaled, scaling_params

def simple_quadratic_example():
    """Simple example to verify polynomial regression works"""
    print("\n" + "=" * 60)
    print("SIMPLE QUADRATIC EXAMPLE (Verification)")
    print("=" * 60)
    
    # Create simple quadratic data
    np.random.seed(42)
    X = np.linspace(-2, 2, 50).reshape(-1, 1)
    y = 1 + 2*X - 0.5*X**2 + np.random.randn(50, 1)*0.3
    
    # Create and fit model
    model = PolynomialRegression(degree=2, method='normal_equation')
    model.fit(X, y.flatten(), verbose=False)
    
    # Print coefficients
    if hasattr(model, 'theta'):
        print(f"\nTrue relationship: y = 1 + 2x - 0.5x²")
        print(f"Estimated: y = {model.theta[0]:.3f} + {model.theta[1]:.3f}x + {model.theta[2]:.3f}x²")
    
    # Calculate metrics
    mse = model.mse(X, y.flatten())
    r2 = model.r2_score(X, y.flatten())
    
    print(f"\nMSE: {mse:.4f}")
    print(f"R²:  {r2:.4f}")
    
    # Simple plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.6, label='Data')
    X_line = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Quadratic fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Quadratic Relationship')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    y_pred = model.predict(X)
    residuals = y.flatten() - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return model, X, y

def main():
    """Main execution"""
    try:
        # First, verify with simple example
        model_simple, X_simple, y_simple = simple_quadratic_example()
        
        # Then run the real-world example
        models_housing, X_scaled, y_scaled, scaling_params = real_world_example_fixed()
        
        print("\n" + "=" * 60)
        print("SUCCESS! All examples completed correctly.")
        print("=" * 60)
        
        print("\nKey improvements in this version:")
        print("1. Proper data scaling (0-1 range for training)")
        print("2. Correct inverse transformation for predictions")
        print("3. Use of normal equation for stability")
        print("4. Proper axis labeling and formatting")
        print("5. Realistic house sizes (800-3000 sqft) displayed correctly")
        
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("\nMake sure you have all required packages:")
        print("pip install numpy matplotlib")
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()