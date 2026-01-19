"""
Diabetes Progression Prediction
Using the Diabetes Dataset from scikit-learn
Predict disease progression one year after baseline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.linear_regression import LinearRegression

def load_diabetes_data():
    """Load and explore diabetes dataset"""
    print("=" * 70)
    print("DIABETES PROGRESSION PREDICTION")
    print("=" * 70)
    
    # Load dataset
    diabetes = load_diabetes()
    X = diabetes.data
    y = diabetes.target
    feature_names = diabetes.feature_names
    
    print(f"\nDataset Overview:")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target: Quantitative measure of disease progression one year after baseline")
    print(f"   Target range: {y.min()} to {y.max()}")
    
    # Create DataFrame for exploration
    df = pd.DataFrame(X, columns=feature_names)
    df['disease_progression'] = y
    
    print(f"\nFeatures (all normalized):")
    for i, feature in enumerate(feature_names, 1):
        print(f"   {i:2}. {feature}")
    
    # Feature descriptions
    feature_descriptions = {
        'age': 'Age (years)',
        'sex': 'Gender',
        'bmi': 'Body Mass Index',
        'bp': 'Average Blood Pressure',
        's1': 'Total Serum Cholesterol',
        's2': 'Low-Density Lipoproteins (LDL)',
        's3': 'High-Density Lipoproteins (HDL)',
        's4': 'Total Cholesterol / HDL Ratio',
        's5': 'Log of Serum Triglycerides Level',
        's6': 'Blood Sugar Level'
    }
    
    print(f"\nFeature Descriptions:")
    for feature, desc in feature_descriptions.items():
        if feature in feature_names:
            print(f"   • {feature}: {desc}")
    
    return X, y, feature_names, df

def explore_data(df, feature_names):
    """Explore and visualize the data"""
    print("\n" + "=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe().round(2))
    
    # Correlation analysis
    print(f"\nCorrelation with Disease Progression:")
    correlations = df.corr()['disease_progression'].sort_values(ascending=False)
    
    for feature, corr in correlations.items():
        if feature != 'disease_progression':
            # Interpret correlation strength
            strength = "very strong" if abs(corr) > 0.7 else \
                      "strong" if abs(corr) > 0.5 else \
                      "moderate" if abs(corr) > 0.3 else \
                      "weak" if abs(corr) > 0.1 else "very weak"
            
            direction = "positive" if corr > 0 else "negative"
            print(f"   {feature:5} : {corr:+.3f} ({strength} {direction} correlation)")
    
    # Top correlations visualization
    top_features = correlations.index[1:6]  # Top 5 features excluding target
    
    plt.figure(figsize=(15, 10))
    
    # 1. Correlation heatmap
    plt.subplot(2, 3, 1)
    corr_matrix = df.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap')
    
    # 2. Target distribution
    plt.subplot(2, 3, 2)
    plt.hist(df['disease_progression'], bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Disease Progression')
    plt.ylabel('Frequency')
    plt.title('Distribution of Disease Progression')
    plt.grid(True, alpha=0.3)
    
    # 3. Top features vs target
    for i, feature in enumerate(top_features[:3], 3):
        plt.subplot(2, 3, i)
        plt.scatter(df[feature], df['disease_progression'], alpha=0.5, s=20)
        
        # Add regression line
        z = np.polyfit(df[feature], df['disease_progression'], 1)
        p = np.poly1d(z)
        plt.plot(df[feature].sort_values(), p(df[feature].sort_values()), 
                "r--", alpha=0.8, linewidth=2)
        
        plt.xlabel(feature)
        plt.ylabel('Disease Progression')
        plt.title(f'{feature} vs Progression\nCorr: {correlations[feature]:.3f}')
        plt.grid(True, alpha=0.3)
    
    # 4. Feature importance from correlation
    plt.subplot(2, 3, 6)
    top_corrs = correlations[1:6]  # Top 5 features
    colors = ['green' if c > 0 else 'red' for c in top_corrs.values]
    plt.barh(range(len(top_corrs)), top_corrs.values, color=colors)
    plt.yticks(range(len(top_corrs)), top_corrs.index)
    plt.xlabel('Correlation Coefficient')
    plt.title('Top 5 Predictive Features')
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.show()
    
    return top_features.tolist()

def train_diabetes_model(X, y, feature_names):
    """Train linear regression model on diabetes data"""
    print("\n" + "=" * 70)
    print("MODEL TRAINING")
    print("=" * 70)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nData Split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples:  {X_test.shape[0]}")
    print(f"   Features per sample: {X_train.shape[1]}")
    
    # Scale features (important for interpretation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTraining Model...")
    
    # Train with Gradient Descent
    model_gd = LinearRegression(
        method='gradient_descent',
        learning_rate=0.01,
        n_iterations=5000,
        regularization=0.1  # Small regularization to prevent overfitting
    )
    
    model_gd.fit(X_train_scaled, y_train, verbose=True, early_stopping=True)
    
    # Also train with Normal Equation for comparison
    model_ne = LinearRegression(
        method='normal_equation',
        regularization=0.1
    )
    model_ne.fit(X_train_scaled, y_train)
    
    print(f"\nTraining Complete!")

    # Evaluate both models
    print(f"\nModel Performance:")
    print("-" * 70)
    print(f"{'Metric':<20} {'Gradient Descent':<20} {'Normal Equation':<20}")
    print("-" * 70)
    
    models = [('Gradient Descent', model_gd), ('Normal Equation', model_ne)]
    
    for name, model in models:
        # Training performance
        y_train_pred = model.predict(X_train_scaled)
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        
        # Testing performance
        y_test_pred = model.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        print(f"{name:<20}")
        print(f"  Training:")
        print(f"    MSE:  {train_mse:>17.2f}")
        print(f"    MAE:  {train_mae:>17.2f}")
        print(f"    R²:   {train_r2:>17.4f}")
        print(f"  Testing:")
        print(f"    MSE:  {test_mse:>17.2f}")
        print(f"    MAE:  {test_mae:>17.2f}")
        print(f"    R²:   {test_r2:>17.4f}")
        print()
    
    # Feature importance analysis
    print(f"\nFeature Importance (from Gradient Descent model):")
    print("-" * 70)
    
    # Get coefficients
    coefficients = model_gd.coefficients
    intercept = model_gd.intercept
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'description': [get_feature_description(f) for f in feature_names]
    })
    
    # Sort by absolute coefficient value
    importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
    
    for idx, row in importance_df.iterrows():
        direction = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"   {row['feature']:5} : {row['coefficient']:+.4f} ({direction} progression)")
        print(f"        {row['description']}")
    
    print(f"\n   Intercept (baseline progression): {intercept:.2f}")
    
    return model_gd, model_ne, X_train_scaled, X_test_scaled, y_train, y_test, scaler, importance_df

def get_feature_description(feature):
    """Get human-readable description of features"""
    descriptions = {
        'age': 'Age of patient (years)',
        'sex': 'Gender (male/female)',
        'bmi': 'Body Mass Index - higher BMI increases risk',
        'bp': 'Average Blood Pressure',
        's1': 'Total Serum Cholesterol',
        's2': 'Low-Density Lipoproteins (LDL - "bad" cholesterol)',
        's3': 'High-Density Lipoproteins (HDL - "good" cholesterol)',
        's4': 'Total Cholesterol / HDL Ratio',
        's5': 'Log of Serum Triglycerides Level',
        's6': 'Blood Sugar Level'
    }
    return descriptions.get(feature, feature)

def visualize_predictions(model, X_test, y_test, feature_names, importance_df):
    """Visualize model predictions and performance"""
    print("\n" + "=" * 70)
    print("PREDICTION VISUALIZATION")
    print("=" * 70)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    plt.figure(figsize=(15, 10))
    
    # 1. Actual vs Predicted
    plt.subplot(2, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, s=30)
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', 
             label='Perfect Prediction', linewidth=2)
    
    plt.xlabel('Actual Disease Progression')
    plt.ylabel('Predicted Disease Progression')
    plt.title('Actual vs Predicted Values\nR² = {:.4f}'.format(r2_score(y_test, y_pred)))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Residual plot
    plt.subplot(2, 3, 2)
    residuals = y_test - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6, s=30)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot\nMAE = {:.2f}'.format(mean_absolute_error(y_test, y_pred)))
    plt.grid(True, alpha=0.3)
    
    # 3. Error distribution
    plt.subplot(2, 3, 3)
    plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution\nStd Dev = {:.2f}'.format(residuals.std()))
    plt.grid(True, alpha=0.3)
    
    # 4. Learning curve (if available)
    plt.subplot(2, 3, 4)
    if hasattr(model, 'cost_history') and model.cost_history:
        plt.plot(model.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.title('Learning Curve')
        plt.grid(True, alpha=0.3)
        
        # Add final cost annotation
        final_cost = model.cost_history[-1]
        plt.annotate(f'Final MSE: {final_cost:.2f}', 
                    xy=(0.6, 0.9), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 5. Feature importance bar chart
    plt.subplot(2, 3, 5)
    top_n = min(8, len(importance_df))
    top_features = importance_df.head(top_n)
    
    colors = ['green' if c > 0 else 'red' for c in top_features['coefficient']]
    y_pos = np.arange(len(top_features))
    
    plt.barh(y_pos, top_features['coefficient'], color=colors)
    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Coefficient Value')
    plt.title(f'Top {top_n} Most Important Features')
    plt.grid(True, alpha=0.3, axis='x')
    
    # Add coefficient values on bars
    for i, (coef, feature) in enumerate(zip(top_features['coefficient'], top_features['feature'])):
        plt.text(coef + (0.01 if coef >= 0 else -0.05), i, 
                f'{coef:+.2f}', va='center',
                color='black', fontweight='bold')
    
    # 6. Prediction error vs actual value
    plt.subplot(2, 3, 6)
    abs_errors = np.abs(residuals)
    plt.scatter(y_test, abs_errors, alpha=0.6, s=30)
    
    # Add moving average
    sorted_idx = np.argsort(y_test)
    window_size = 20
    moving_avg = np.convolve(abs_errors[sorted_idx], 
                            np.ones(window_size)/window_size, mode='valid')
    plt.plot(y_test[sorted_idx][window_size-1:], moving_avg, 
             'r-', linewidth=2, label=f'{window_size}-point moving average')
    
    plt.xlabel('Actual Disease Progression')
    plt.ylabel('Absolute Prediction Error')
    plt.title('Error vs Actual Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistical analysis of errors
    print(f"\nPrediction Error Analysis:")
    print("-" * 70)
    print(f"   Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"   Root Mean Squared Error (RMSE): {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
    print(f"   Mean Absolute Percentage Error (MAPE): {np.mean(np.abs(residuals / y_test)) * 100:.1f}%")
    print(f"   Error Standard Deviation: {residuals.std():.2f}")
    print(f"   95% of predictions within: ±{np.percentile(np.abs(residuals), 95):.2f} units")

def predict_new_patient(model, scaler, feature_names):
    """Predict disease progression for a new patient"""
    print("\n" + "=" * 70)
    print("NEW PATIENT PREDICTION")
    print("=" * 70)
    
    print("\nEnter patient information (normalized values):")
    print("   Note: Values are normalized (mean=0, std=1)")
    print("   Positive values = above average, Negative = below average")
    print("   Press Enter to use average values (0)")
    
    patient_data = []
    
    for feature in feature_names:
        desc = get_feature_description(feature)
        while True:
            try:
                value = input(f"\n   {feature} ({desc}): ").strip()
                if value == "":
                    value = 0.0  # Use average
                    print(f"     Using average value: 0.0")
                    break
                value = float(value)
                break
            except ValueError:
                print("     Please enter a valid number or press Enter for average")
        
        patient_data.append(value)
    
    # Convert to numpy array and scale
    patient_array = np.array(patient_data).reshape(1, -1)
    patient_scaled = scaler.transform(patient_array)
    
    # Make prediction
    prediction = model.predict(patient_scaled)[0]
    
    print(f"\nPrediction Results:")
    print("-" * 70)
    print(f"   Predicted disease progression: {prediction:.2f}")
    
    # Interpret the prediction
    print(f"\nInterpretation:")
    if prediction < 100:
        print(f"   LOW RISK: Disease progression likely to be below average")
        print(f"   Recommended: Continue healthy lifestyle, annual checkup")
    elif prediction < 150:
        print(f"   MODERATE RISK: Average disease progression expected")
        print(f"   Recommended: Regular monitoring, lifestyle modifications")
    elif prediction < 200:
        print(f"   HIGH RISK: Above average progression expected")
        print(f"   Recommended: Medical consultation, closer monitoring")
    else:
        print(f"   VERY HIGH RISK: Significant disease progression expected")
        print(f"   Recommended: Immediate medical attention, aggressive management")
    
    # Show feature contributions
    print(f"\n🔍 Feature Contributions:")
    print("-" * 70)
    
    coefficients = model.coefficients
    contributions = patient_scaled[0] * coefficients
    
    for feature, value, coef, contrib in zip(feature_names, patient_scaled[0], 
                                            coefficients, contributions):
        if abs(contrib) > 1:  # Show only significant contributions
            effect = "increases" if contrib > 0 else "decreases"
            print(f"   {feature:5}: {value:+.2f} × {coef:+.2f} = {contrib:+.2f} ({effect} risk)")
    
    baseline = model.intercept
    total = baseline + contributions.sum()
    print(f"\n   Baseline progression: {baseline:.2f}")
    print(f"   Total contributions:  {contributions.sum():+.2f}")
    print(f"   Final prediction:     {total:.2f}")
    
    return prediction

def cross_validation_analysis(X, y):
    """Perform cross-validation to assess model stability"""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION ANALYSIS")
    print("=" * 70)
    
    # Scale entire dataset
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Custom cross-validation
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_scores = []
    fold_coefficients = []
    
    print(f"\nPerforming 5-fold cross-validation...")
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model = LinearRegression(
            method='gradient_descent',
            learning_rate=0.01,
            n_iterations=2000,
            regularization=0.1
        )
        model.fit(X_train, y_train, verbose=False)
        
        # Evaluate
        y_pred = model.predict(X_val)
        fold_r2 = r2_score(y_val, y_pred)
        fold_mse = mean_squared_error(y_val, y_pred)
        
        fold_scores.append({'fold': fold, 'r2': fold_r2, 'mse': fold_mse})
        fold_coefficients.append(model.coefficients)
        
        print(f"   Fold {fold}: R² = {fold_r2:.4f}, MSE = {fold_mse:.2f}")
    
    # Analyze results
    scores_df = pd.DataFrame(fold_scores)
    
    print(f"\nCross-Validation Results:")
    print("-" * 70)
    print(f"   Average R²:  {scores_df['r2'].mean():.4f} (±{scores_df['r2'].std():.4f})")
    print(f"   Average MSE: {scores_df['mse'].mean():.2f} (±{scores_df['mse'].std():.2f})")
    print(f"   Min R²:      {scores_df['r2'].min():.4f}")
    print(f"   Max R²:      {scores_df['r2'].max():.4f}")
    
    # Coefficient stability across folds
    coeff_array = np.array(fold_coefficients)
    coeff_stability = pd.DataFrame({
        'mean': coeff_array.mean(axis=0),
        'std': coeff_array.std(axis=0),
        'cv': coeff_array.std(axis=0) / np.abs(coeff_array.mean(axis=0))  # Coefficient of variation
    })
    
    print(f"\nCoefficient Stability:")
    print(f"   Most stable coefficient: CV = {coeff_stability['cv'].min():.3f}")
    print(f"   Least stable coefficient: CV = {coeff_stability['cv'].max():.3f}")
    
    return scores_df, coeff_stability

def clinical_recommendations(importance_df, model_performance):
    """Generate clinical recommendations based on model insights"""
    print("\n" + "=" * 70)
    print("CLINICAL INSIGHTS & RECOMMENDATIONS")
    print("=" * 70)
    
    print(f"\nBased on the linear regression analysis:")
    
    # Top risk factors
    top_risks = importance_df.head(3)
    
    print(f"\nTOP 3 RISK FACTORS:")
    for idx, row in top_risks.iterrows():
        effect = "INCREASES" if row['coefficient'] > 0 else "DECREASES"
        print(f"   1. {row['feature'].upper()} - {effect} disease progression")
        print(f"      {get_feature_description(row['feature'])}")
    
    # Protective factors
    protective = importance_df[importance_df['coefficient'] < 0].head(3)
    
    if len(protective) > 0:
        print(f"\n🟢 TOP PROTECTIVE FACTORS:")
        for idx, row in protective.iterrows():
            print(f"   • {row['feature'].upper()} - DECREASES disease progression")
            print(f"     {get_feature_description(row['feature'])}")
    
    # Model performance implications
    print(f"\nMODEL RELIABILITY:")
    r2_score = model_performance.get('test_r2', 0.5)
    
    if r2_score > 0.6:
        print(f"   ✅ High predictive power (R² = {r2_score:.3f})")
        print(f"   Model explains {r2_score*100:.1f}% of variance in disease progression")
        print(f"   Suitable for individual risk assessment")
    elif r2_score > 0.4:
        print(f"   ⚠️  Moderate predictive power (R² = {r2_score:.3f})")
        print(f"   Model explains {r2_score*100:.1f}% of variance")
        print(f"   Best for population-level insights")
    else:
        print(f"   ⚠️  Limited predictive power (R² = {r2_score:.3f})")
        print(f"   Use with caution for individual predictions")
    
    # Clinical recommendations
    print(f"\n💡 CLINICAL RECOMMENDATIONS:")
    print(f"   1. Focus interventions on modifiable risk factors (BMI, BP, blood lipids)")
    print(f"   2. Monitor high-risk patients more frequently")
    print(f"   3. Use model as screening tool, not diagnostic")
    print(f"   4. Combine with clinical judgment and other tests")
    
    print(f"\n⚠️  LIMITATIONS:")
    print(f"   • Linear model assumes linear relationships")
    print(f"   • Does not capture complex interactions")
    print(f"   • Based on normalized data - real values may differ")
    print(f"   • Cannot replace comprehensive medical evaluation")

def main():
    """Main function to run diabetes prediction analysis"""
    
    # Load and explore data
    X, y, feature_names, df = load_diabetes_data()
    
    # Data exploration
    top_features = explore_data(df, feature_names)
    
    # Train model
    model_gd, model_ne, X_train, X_test, y_train, y_test, scaler, importance_df = train_diabetes_model(
        X, y, feature_names
    )
    
    # Visualize predictions
    visualize_predictions(model_gd, X_test, y_test, feature_names, importance_df)
    
    # Cross-validation
    cv_scores, coeff_stability = cross_validation_analysis(X, y)
    
    # Generate clinical insights
    model_performance = {
        'test_r2': r2_score(y_test, model_gd.predict(X_test))
    }
    clinical_recommendations(importance_df, model_performance)
    
    # Optional: Predict for new patient
    try:
        predict_new = input("\n👤 Predict for a new patient? (y/n): ").strip().lower()
        if predict_new == 'y':
            prediction = predict_new_patient(model_gd, scaler, feature_names)
    except:
        print("\nSkipping new patient prediction")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nSummary:")
    print(f"• Dataset: {X.shape[0]} patients with {X.shape[1]} clinical measurements")
    print(f"• Best model R²: {r2_score(y_test, model_gd.predict(X_test)):.4f}")
    print(f"• Most important feature: {importance_df.iloc[0]['feature']}")
    print(f"• Clinical utility: Screening tool for diabetes progression risk")
    print(f"\n🚨 IMPORTANT: This is for educational purposes only.")
    print("   Not for actual medical diagnosis or treatment decisions.")

if __name__ == "__main__":
    # Set matplotlib style
    plt.style.use('seaborn-v0_8-darkgrid')
    

    main()
