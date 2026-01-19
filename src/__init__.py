"""
Linear Regression from Scratch Package

A comprehensive implementation of linear regression algorithms
from first principles with mathematical explanations.

Author: Karapetyan Arayik
"""

__version__ = "0.1.0"
__author__ = "Karapetyan Arayik"
__email__ = "arayik.1098@gmail.com"

from .linear_regression import LinearRegression, PolynomialRegression

__all__ = [
    "LinearRegression",
    "PolynomialRegression",
]

# Package metadata
__description__ = "Linear Regression algorithms implemented from scratch"
__url__ = "https://github.com/ArayikKarapetyan/linear-regression-scratch"
__keywords__ = ["machine-learning", "linear-regression", "gradient-descent", "python"]

print(f"Linear Regression from Scratch v{__version__}")
print("Use: from src.linear_regression import LinearRegression")