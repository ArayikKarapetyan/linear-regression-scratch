# Linear Regression from Scratch - Complete Guide

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/ArayikKarapetyan/linear-regression-scratch?style=social)](https://github.com/ArayikKarapetyan/linear-regression-scratch)

**Complete implementation of linear regression algorithms from first principles with mathematical explanations, visualizations, and real-world examples.**

---

## Table of Contents

1. [Core Implementation](#core-implementation)
2. [Examples Guide](#examples-guide)
3. [Getting Started](#getting-started)
4. [Testing](#testing)
5. [Documentation](#documentation)
6. [Learning Path](#learning-path)
7. [Project Structure](#project-structure)
8. [Contributing](#contributing)
9. [Quick Links](#quick-links)

---

## Core Implementation

### Main Algorithm File

**Location**: `src/linear_regression.py`

**What it contains**:
- `LinearRegression` class with Gradient Descent and Normal Equation
- `PolynomialRegression` class for non-linear relationships
- Ridge Regression (L2 regularization) support
- Learning curve visualization methods
- Performance metrics (R², MSE)

### Key Classes & Methods

```python
# Main classes
from src.linear_regression import LinearRegression, PolynomialRegression

# Core methods
model.fit(X, y)                    # Train the model
model.predict(X)                   # Make predictions
model.r2_score(X, y)              # Calculate R²
model.mse(X, y)                   # Calculate Mean Squared Error
model.plot_learning_curve()       # Visualize training progress
model.plot_regression_line(X, y)  # Plot fitted line
```

---

## Examples Guide

### Where are examples?

All examples are in the `examples/` folder:

```
examples/
├── 01_basic_usage.py            # Start here - basic concepts
├── 02_gradient_descent.py       # Optimization methods
├── 03_normal_equation.py        # Closed-form solution
├── 04_polynomial_regression.py  # Non-linear relationships
├── 05_real_estate.py            # Real-world: Housing prices
├── 06_diabetes_prediction.py    # Real-world: Medical analytics
└── 07_stock_prediction.py       # Real-world: Financial forecasting
```

### What Each Example Shows

#### 1. Basic Usage (`01_basic_usage.py`)

**Run it**: `python examples/01_basic_usage.py`

- Simple linear regression implementation
- Model training and prediction
- Gradient Descent vs Normal Equation comparison
- Visualizations: Regression line plots, learning curves

#### 2. Gradient Descent (`02_gradient_descent.py`)

**Run it**: `python examples/02_gradient_descent.py`

- Learning rate effects analysis
- Batch vs Mini-batch vs Stochastic GD
- Early stopping implementation
- Visualizations: Convergence plots, rate comparisons

#### 3. Normal Equation (`03_normal_equation.py`)

**Run it**: `python examples/03_normal_equation.py`

- Closed-form solution mathematics
- Handling singular matrices
- Ridge regression for regularization
- Visualizations: Method comparison, complexity analysis

#### 4. Polynomial Regression (`04_polynomial_regression.py`)

**Run it**: `python examples/04_polynomial_regression.py`

- Fitting non-linear relationships
- Degree selection and overfitting
- Real-world: House price vs size
- Visualizations: Polynomial fits, R² vs degree plots

#### 5. Real Estate Prediction (`05_real_estate.py`)

**Run it**: `python examples/05_real_estate.py`

- California Housing dataset
- Feature importance analysis
- Error metrics for regression
- Visualizations: Correlation heatmaps, prediction plots

#### 6. Diabetes Prediction (`06_diabetes_prediction.py`)

**Run it**: `python examples/06_diabetes_prediction.py`

- Medical dataset analysis
- Clinical feature interpretation
- Risk assessment modeling
- Visualizations: Feature importance, residual analysis

#### 7. Stock Prediction (`07_stock_prediction.py`)

**Run it**: `python examples/07_stock_prediction.py`

- Time series forecasting
- Lag feature engineering
- Financial data analysis
- Visualizations: Price history, prediction accuracy

---

## Getting Started

### Step 1: Clone & Setup

```bash
# Clone the repository
git clone https://github.com/ArayikKarapetyan/linear-regression-scratch.git
cd linear-regression-scratch

# Install core dependencies
pip install numpy matplotlib pandas
```

### Step 2: Run Your First Example

```bash
# Start with basic usage
python examples/01_basic_usage.py

# Or run gradient descent analysis
python examples/02_gradient_descent.py
```

### Step 3: Explore the Implementation

Open and examine the main algorithm file:

```bash
# View the core implementation
cat src/linear_regression.py

# Or open in your editor
code src/linear_regression.py
```

### Step 4: Run All Examples

```bash
# Run all examples sequentially
for file in examples/*.py; do
    echo "Running $file..."
    python "$file"
    echo "----------------------"
done
```

---

## Testing

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_linear_regression.py

# Run with verbose output
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

### Test Structure

**Location**: `tests/test_linear_regression.py`

**What's tested**:
- Gradient Descent convergence
- Normal Equation accuracy
- Prediction correctness
- Edge cases and error handling

---

## Documentation

### Mathematical Documentation

**Location**: `docs/mathematical_derivations.md`

**Contents**: Complete mathematical proofs and derivations

### Algorithm Comparison

**Location**: `docs/algorithm_comparison.md`

**Contents**: Gradient Descent vs Normal Equation analysis

### Interactive Tutorials

**Location**: `notebooks/demo.ipynb` (if available)

**Contents**: Jupyter notebook with visual examples

---

## Learning Path

### For Beginners

1. Start with `01_basic_usage.py`
2. Read the mathematical derivations in `docs/`
3. Experiment with different learning rates in `02_gradient_descent.py`
4. Try polynomial regression with `04_polynomial_regression.py`

### For Intermediate Users

1. Analyze real-world examples (`05_real_estate.py`, `06_diabetes_prediction.py`)
2. Compare performance with scikit-learn
3. Modify algorithm parameters and observe effects
4. Add your own features to the models

### For Advanced Users

1. Implement additional optimization algorithms
2. Add L1 regularization (Lasso)
3. Extend to logistic regression
4. Implement cross-validation

---

## 📁 Project Structure

```
linear-regression-scratch/
│
├── 📂 src/                    # CORE ALGORITHMS
│   └── linear_regression.py  # Main implementation
│
├── 📂 examples/              # PRACTICAL APPLICATIONS
│   ├── 01-03: Foundational
│   ├── 04: Polynomial fitting
│   └── 05-07: Real-world cases
│
├── 📂 tests/                 # TEST SUITE
│   └── test_linear_regression.py
│
├── 📂 docs/                  # MATHEMATICAL DOCS
│   ├── mathematical_derivations.md
│   └── algorithm_comparison.md
│
└── 📂 notebooks/             # INTERACTIVE TUTORIALS
    └── demo.ipynb
```

---

## Contributing

### How to Add New Examples

1. Create new file in `examples/` folder (e.g., `08_new_example.py`)
2. Follow existing structure with clear documentation
3. Include visualizations if applicable
4. Add to this README's examples guide

### Reporting Issues

1. Check if issue already exists in Issues
2. Provide example code to reproduce
3. Include expected vs actual behavior

---

## Quick Links

- **Main Algorithm**: `src/linear_regression.py`
- **All Examples**: `examples/`
- **Tests**: `tests/`
- **Documentation**: `docs/`

---

## 📞 Need Help?

- Check the examples for similar use cases
- Review mathematical derivations in `docs/`
- Run tests to verify your implementation
- Open an issue for specific questions

---

<div align="center">

## 🚀 Start Exploring!

```bash
# Pick an example and run it
python examples/01_basic_usage.py
```

### ⭐ Found this helpful? Star the repo!

[![GitHub stars](https://img.shields.io/github/stars/ArayikKarapetyan/linear-regression-scratch?style=social)](https://github.com/ArayikKarapetyan/linear-regression-scratch)

**Built for learning with ❤️**


</div>
