# Mathematical Derivations for Linear Regression

## 1. Problem Formulation

Given a dataset with $m$ examples and $n$ features:
- Input: $X \in \mathbb{R}^{m \times n}$
- Output: $y \in \mathbb{R}^{m}$
- Parameters: $\theta \in \mathbb{R}^{n+1}$ (including bias term $\theta_0$)

## 2. Hypothesis Function

The linear hypothesis is:

$$
h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

In vector form (with $x_0 = 1$ for the bias term):

$$
h_{\theta}(x) = \theta^T x = \sum_{j=0}^{n} \theta_j x_j
$$

## 3. Cost Function: Mean Squared Error (MSE)

We want to minimize the average squared error:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2
$$

The $\frac{1}{2}$ factor is for mathematical convenience when taking derivatives.

### Matrix Notation

Let $X \in \mathbb{R}^{m \times (n+1)}$ be the design matrix with a column of ones added:

$$
X = \begin{bmatrix}
1 & x_1^{(1)} & \dots & x_n^{(1)} \\
1 & x_1^{(2)} & \dots & x_n^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(m)} & \dots & x_n^{(m)}
\end{bmatrix}
$$

Then:
- Predictions: $\hat{y} = X\theta$
- Errors: $\epsilon = X\theta - y$
- Cost: $J(\theta) = \frac{1}{2m} \epsilon^T \epsilon = \frac{1}{2m} (X\theta - y)^T (X\theta - y)$

## 4. Gradient Descent Derivation

### Partial Derivatives

For each parameter $\theta_j$:

$$
\frac{\partial J}{\partial \theta_j} = \frac{\partial}{\partial \theta_j} \left[ \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 \right]
$$

Using chain rule:

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) \cdot \frac{\partial}{\partial \theta_j} h_{\theta}(x^{(i)})
$$

Since $\frac{\partial}{\partial \theta_j} h_{\theta}(x^{(i)}) = x_j^{(i)}$:

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)}
$$

### Gradient Vector

In vector form, the gradient is:

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y)
$$

### Gradient Descent Update Rule

Update each parameter simultaneously:

$$
\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}
$$

In vector form:

$$
\theta := \theta - \alpha \nabla J(\theta)
$$

Where $\alpha$ is the learning rate.

## 5. Normal Equation Derivation

### Setting Gradient to Zero

For the optimal $\theta$, the gradient should be zero:

$$
\nabla J(\theta) = \frac{1}{m} X^T (X\theta - y) = 0
$$

Multiply both sides by $m$:

$$
X^T (X\theta - y) = 0
$$

$$
X^T X \theta - X^T y = 0
$$

$$
X^T X \theta = X^T y
$$

### Solving for θ

Assuming $X^T X$ is invertible:

$$
\theta = (X^T X)^{-1} X^T y
$$

This is the **Normal Equation**.

## 6. Regularization: Ridge Regression

### Regularized Cost Function

Add L2 penalty to prevent overfitting:

$$
J(\theta) = \frac{1}{2m} \left[ \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{j=1}^{n} \theta_j^2 \right]
$$

Note: We don't regularize $\theta_0$ (bias term).

### Gradient with Regularization

$$
\frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x_j^{(i)} + \frac{\lambda}{m} \theta_j \quad \text{for } j \geq 1
$$

$$
\frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)}) x_0^{(i)}
$$

### Normal Equation with Regularization

$$
\theta = (X^T X + \lambda I)^{-1} X^T y
$$

Where $I$ is the identity matrix with $I_{00} = 0$ to exclude bias term from regularization.

## 7. Polynomial Regression

Transform features to polynomial basis:

For a single feature $x$, create polynomial features:
$$
\phi(x) = [1, x, x^2, x^3, \dots, x^d]
$$

For multiple features, include interaction terms.

## 8. Time Complexity Analysis

| Method | Training Time | Prediction Time | Space |
|--------|---------------|-----------------|--------|
| Gradient Descent | $O(k \cdot m \cdot n)$ | $O(n)$ | $O(n)$ |
| Normal Equation | $O(n^3)$ | $O(n)$ | $O(n^2)$ |
| Ridge Regression | $O(n^3)$ | $O(n)$ | $O(n^2)$ |

Where:
- $m$: number of training examples
- $n$: number of features
- $k$: number of iterations

## 9. Convergence Proof (Gradient Descent)

For convex cost functions like MSE, gradient descent converges to global minimum with appropriate learning rate.

### Lipschitz Continuity
The gradient $\nabla J$ is L-Lipschitz continuous, so:

$$
J(\theta_{t+1}) \leq J(\theta_t) - \alpha \left(1 - \frac{\alpha L}{2}\right) \|\nabla J(\theta_t)\|^2
$$

For $\alpha < \frac{2}{L}$, the cost decreases monotonically.

## 10. Geometric Interpretation

- **Cost Function**: Paraboloid in parameter space
- **Gradient**: Direction of steepest ascent
- **Gradient Descent**: Moving opposite to gradient
- **Normal Equation**: Finding stationary point analytically

## References

1. Boyd & Vandenberghe, "Convex Optimization"
2. Bishop, "Pattern Recognition and Machine Learning"
3. Stanford CS229 Lecture Notes