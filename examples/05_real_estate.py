"""
Real Estate Price Prediction
Using Boston Housing Dataset
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_boston_dataset():
    """Load Boston housing dataset"""
    print("=" * 60)
    print("REAL ESTATE: California Housing Dataset")
    print("=" * 60)
    
    # Load dataset
    housing = fetch_california_housing()
    X = housing.data
    y = housing.target
    feature_names = housing.feature_names
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"\nFeatures:")
    for i, name in enumerate(feature_names):
        print(f"  {i+1}. {name}")
    
    print(f"\nTarget: Median house value (in $100,000s)")
    print(f"Sample prices: ${y[:5]*100000:,.0f}")
    
    return X, y, feature_names

def analyze_features(X, y, feature_names):
    """Analyze feature importance and correlations"""
    print("\n" + "=" * 60)
    print("FEATURE ANALYSIS")
    print("=" * 60)
    
    # Create DataFrame for analysis
    df = pd.DataFrame(X, columns=feature_names)
    df['PRICE'] = y * 100000  # Convert to actual dollars
    
    print("\nFeature Statistics:")
    print(df.describe().round(2))
    
    print("\nCorrelation with Price:")
    correlations = df.corr()['PRICE'].sort_values(ascending=False)
    for feature, corr in correlations.items():
        if feature != 'PRICE':
            print(f"  {feature:15} : {corr:+.3f}")
    
    # Visualize top correlations
    top_features = correlations.index[1:5]  # Skip PRICE itself
    
    plt.figure(figsize=(12, 3))
    for i, feature in enumerate(top_features):
        plt.subplot(1, 4, i + 1)
        plt.scatter(df[feature], df['PRICE'], alpha=0.5, s=10)
        plt.xlabel(feature)
        plt.ylabel('Price ($)')
        plt.title(f'Corr: {correlations[feature]:.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return df

def train_model(X, y, feature_names):
    """Train linear regression model"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features (important for gradient descent)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train model
    model = LinearRegression(method='gradient_descent',
                            learning_rate=0.01,
                            n_iterations=5000)
    model.fit(X_train_scaled, y_train, verbose=True)
    
    # Evaluate
    train_r2 = model.r2_score(X_train_scaled, y_train)
    test_r2 = model.r2_score(X_test_scaled, y_test)
    train_mse = model.mse(X_train_scaled, y_train)
    test_mse = model.mse(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Test R²:     {test_r2:.4f}")
    print(f"  Training MSE: {train_mse:.4f} (${train_mse*100000:,.0f})")
    print(f"  Test MSE:     {test_mse:.4f} (${test_mse*100000:,.0f})")
    
    # Feature importance (coefficients)
    print(f"\nFeature Importance (Standardized Coefficients):")
    for i, (name, coef) in enumerate(zip(feature_names, model.coefficients)):
        print(f"  {name:15} : {coef:+.6f}")
    
    # Intercept (in standardized space)
    print(f"  {'Intercept':15} : {model.intercept:.6f}")
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test, scaler

def make_predictions(model, X_test_scaled, y_test, scaler, feature_names):
    """Make and visualize predictions"""
    print("\n" + "=" * 60)
    print("PREDICTIONS")
    print("=" * 60)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Convert back to actual dollars
    y_test_actual = y_test * 100000
    y_pred_actual = y_pred * 100000
    
    # Calculate prediction errors
    errors = y_pred_actual - y_test_actual
    abs_errors = np.abs(errors)
    percentage_errors = (abs_errors / y_test_actual) * 100
    
    print(f"\nPrediction Statistics:")
    print(f"  Average error: ${errors.mean():,.0f}")
    print(f"  Average absolute error: ${abs_errors.mean():,.0f}")
    print(f"  Average percentage error: {percentage_errors.mean():.1f}%")
    print(f"  Max error: ${abs_errors.max():,.0f}")
    print(f"  Min error: ${abs_errors.min():,.0f}")
    
    # Show sample predictions
    print(f"\nSample Predictions (first 10 houses):")
    print("-" * 60)
    print(f"{'Actual':>10} {'Predicted':>10} {'Error':>10} {'% Error':>10}")
    print("-" * 60)
    
    for i in range(min(10, len(y_test))):
        actual = y_test_actual[i]
        predicted = y_pred_actual[i]
        error = predicted - actual
        pct_error = (abs(error) / actual) * 100
        print(f"${actual:>9,.0f} ${predicted:>9,.0f} ${error:>9,.0f} {pct_error:>9.1f}%")
    
    # Visualizations
    plt.figure(figsize=(15, 4))
    
    # 1. Actual vs Predicted
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_actual, y_pred_actual, alpha=0.6)
    plt.plot([y_test_actual.min(), y_test_actual.max()],
             [y_test_actual.min(), y_test_actual.max()],
             'r--', label='Perfect Prediction', linewidth=2)
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Actual vs Predicted Prices')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Error distribution
    plt.subplot(1, 3, 2)
    plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error ($)')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    # 3. Learning curve
    plt.subplot(1, 3, 3)
    if hasattr(model, 'cost_history') and model.cost_history:
        plt.plot(model.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Learning Curve')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def predict_custom_house(model, scaler, feature_names):
    """Predict price for a custom house"""
    print("\n" + "=" * 60)
    print("CUSTOM HOUSE PREDICTION")
    print("=" * 60)
    
    # Example house features (from dataset description)
    print("\nEnter house features:")
    print("(Based on California Housing dataset)")
    
    # Get average values for reference
    avg_values = {
        'MedInc': 3.87,          # Median income in block group
        'HouseAge': 28.64,       # Median house age
        'AveRooms': 5.43,        # Average rooms per household
        'AveBedrms': 1.10,       # Average bedrooms per household
        'Population': 1425.48,   # Block group population
        'AveOccup': 3.07,        # Average household members
        'Latitude': 35.63,       # Block group latitude
        'Longitude': -119.57,    # Block group longitude
    }
    
    custom_house = []
    print("\nFeature Values (enter your values or press Enter for average):")
    
    for feature, avg in zip(feature_names, avg_values.values()):
        value = input(f"  {feature} (avg: {avg:.2f}): ")
        if value.strip() == "":
            custom_house.append(avg)
            print(f"    Using average: {avg:.2f}")
        else:
            custom_house.append(float(value))
    
    custom_house = np.array(custom_house).reshape(1, -1)
    
    # Scale features
    custom_house_scaled = scaler.transform(custom_house)
    
    # Predict
    predicted_value = model.predict(custom_house_scaled)[0]
    predicted_price = predicted_value * 100000
    
    print(f"\nPrediction Results:")
    print("-" * 40)
    print(f"Predicted median house value: ${predicted_price:,.0f}")
    print(f"Range (±15%): ${predicted_price*0.85:,.0f} - ${predicted_price*1.15:,.0f}")
    
    return predicted_price

if __name__ == "__main__":
    # Load and analyze data
    X, y, feature_names = load_boston_dataset()
    df = analyze_features(X, y, feature_names)
    
    # Train model
    model, X_train, X_test, y_train, y_test, scaler = train_model(
        X, y, feature_names
    )
    
    # Make predictions
    make_predictions(model, X_test, y_test, scaler, feature_names)
    
    # Optional: Predict custom house
    try:
        predict_custom_house(model, scaler, feature_names)
    except:
        print("\nSkipping custom prediction (input required)")