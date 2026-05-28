"""
neuralkit — neural network from scratch, NumPy only.
No PyTorch. No TensorFlow. Just math.
"""

import numpy as np


# ── Activations ─────────────────────────────────────────────────────────────

class ReLU:
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask

    def parameters(self): return []


class Sigmoid:
    def forward(self, x):
        self._out = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad):
        return grad * self._out * (1 - self._out)

    def parameters(self): return []


class Softmax:
    def forward(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self._out = e / e.sum(axis=1, keepdims=True)
        return self._out

    def backward(self, grad):
        # Combined with cross-entropy this simplifies to (pred - y) / N
        return grad

    def parameters(self): return []


# ── Layers ───────────────────────────────────────────────────────────────────

class Dense:
    def __init__(self, in_features, out_features):
        # He initialisation for ReLU networks
        scale = np.sqrt(2.0 / in_features)
        self.W  = np.random.randn(in_features, out_features) * scale
        self.b  = np.zeros((1, out_features))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad):
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T

    def parameters(self):
        return [(self.W, self.dW), (self.b, self.db)]


# ── Loss functions ───────────────────────────────────────────────────────────

class CrossEntropyLoss:
    """Softmax + cross-entropy in one pass for numerical stability."""

    def __call__(self, logits, y_true):
        N = logits.shape[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        self._grad = probs.copy()
        self._grad[np.arange(N), y_true] -= 1
        self._grad /= N
        loss = -np.log(probs[np.arange(N), y_true] + 1e-12).mean()
        return loss, self._grad


class MSELoss:
    def __call__(self, pred, y_true):
        diff = pred - y_true
        loss = (diff ** 2).mean()
        grad = 2 * diff / y_true.size
        return loss, grad


# ── Sequential model ──────────────────────────────────────────────────────────

class Sequential:
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad):
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
