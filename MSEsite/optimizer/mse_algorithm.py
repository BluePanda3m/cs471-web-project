import numpy as np


def compute_mse(y_true, y_pred):
    """Mean Squared Error: (1/n) * sum((y - y_hat)^2)"""
    n = len(y_true)
    return (1 / n) * np.sum((y_true - y_pred) ** 2)


def compute_gradient(y_true, X, theta):
    """Gradient of MSE w.r.t. theta: (2/n) * X^T * (X*theta - y)"""
    n = len(y_true)
    y_pred = X @ theta
    grad = (2 / n) * X.T @ (y_pred - y_true)
    return grad


def gradient_descent(X, y_true, learning_rate=0.01, iterations=500, seed=42):
    """
    Minimize MSE using Gradient Descent.

    Returns
    -------
    theta       : final parameters [intercept, slope]
    cost_history: list of MSE at each iteration
    theta_history: list of theta at each iteration (for animation)
    """
    np.random.seed(seed)
    n_features = X.shape[1]
    theta = np.zeros(n_features)

    cost_history = []
    theta_history = []

    for i in range(iterations):
        y_pred = X @ theta
        cost = compute_mse(y_true, y_pred)
        cost_history.append(cost)
        theta_history.append(theta.copy())

        grad = compute_gradient(y_true, X, theta)
        theta = theta - learning_rate * grad

    return theta, cost_history, theta_history


def build_dataset(n_samples=80, noise=8.0, seed=42):
    """Generate a simple linear regression dataset."""
    np.random.seed(seed)
    X_raw = np.linspace(0, 10, n_samples)
    y = 3.5 * X_raw + 7 + np.random.randn(n_samples) * noise

    # Add bias column
    X = np.column_stack([np.ones(n_samples), X_raw])
    return X_raw, X, y
