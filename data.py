"""
neuralkit.data — synthetic classification and regression datasets.
"""

import numpy as np


def make_spiral(N=300, classes=3, noise=0.15, seed=42):
    """N samples per class arranged in interlocking spirals."""
    rng = np.random.RandomState(seed)
    X, y = [], []
    for c in range(classes):
        ix = np.arange(N)
        r  = ix / N
        t  = ix / N * 4.0 + c * (4.0 / classes) + rng.randn(N) * noise
        X.append(np.column_stack([r * np.sin(t), r * np.cos(t)]))
        y.append(np.full(N, c))
    X = np.vstack(X).astype(np.float64)
    y = np.hstack(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def make_moons(N=500, noise=0.1, seed=42):
    """Two interleaving half-circles."""
    rng = np.random.RandomState(seed)
    n   = N // 2
    t   = np.linspace(0, np.pi, n)
    x1  = np.column_stack([np.cos(t), np.sin(t)])
    x2  = np.column_stack([1.0 - np.cos(t), 0.5 - np.sin(t)])
    X   = np.vstack([x1, x2]) + rng.randn(N, 2) * noise
    y   = np.array([0] * n + [1] * n)
    idx = rng.permutation(N)
    return X[idx].astype(np.float64), y[idx]


def make_circles(N=500, noise=0.05, factor=0.5, seed=42):
    """Two concentric circles — outer class 0, inner class 1."""
    rng = np.random.RandomState(seed)
    n   = N // 2
    t   = np.linspace(0, 2 * np.pi, n)
    outer = np.column_stack([np.cos(t), np.sin(t)])
    inner = factor * np.column_stack([np.cos(t), np.sin(t)])
    X   = np.vstack([outer, inner]) + rng.randn(N, 2) * noise
    y   = np.array([0] * n + [1] * n)
    idx = rng.permutation(N)
    return X[idx].astype(np.float64), y[idx]


def make_xor(N=500, noise=0.1, seed=42):
    """XOR pattern — 4 quadrants, linearly inseparable."""
    rng = np.random.RandomState(seed)
    X   = rng.randn(N, 2)
    y   = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X  += rng.randn(N, 2) * noise
    return X.astype(np.float64), y


def make_regression(N=500, noise=0.3, seed=42):
    """1-D regression: y = sin(3x) + Gaussian noise."""
    rng = np.random.RandomState(seed)
    X   = rng.uniform(-np.pi, np.pi, (N, 1))
    y   = np.sin(3 * X) + rng.randn(N, 1) * noise
    return X.astype(np.float64), y.astype(np.float64)


def train_val_split(X, y, val_ratio=0.2, seed=42):
    rng   = np.random.RandomState(seed)
    idx   = rng.permutation(len(X))
    split = int(len(X) * (1.0 - val_ratio))
    return X[idx[:split]], y[idx[:split]], X[idx[split:]], y[idx[split:]]


def normalize(X_train, X_val=None):
    """Z-score using training statistics only (no data leakage)."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    if X_val is not None:
        return (X_train - mean) / std, (X_val - mean) / std
    return (X_train - mean) / std
