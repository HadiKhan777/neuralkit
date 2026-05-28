"""
neuralkit — neural network framework from scratch, NumPy only.
Layers: Dense, BatchNorm1d, LayerNorm, Dropout, Flatten
Activations: ReLU, LeakyReLU, GELU, Sigmoid, Tanh, Softmax
Loss: CrossEntropyLoss, MSELoss, BCELoss
"""

import numpy as np


# ── Activations ───────────────────────────────────────────────────────────────

class ReLU:
    def forward(self, x):
        self._mask = x > 0
        return x * self._mask

    def backward(self, grad):
        return grad * self._mask

    def parameters(self): return []
    def __repr__(self): return 'ReLU()'


class LeakyReLU:
    def __init__(self, negative_slope=0.01):
        self.alpha = negative_slope

    def forward(self, x):
        self._x = x
        return np.where(x > 0, x, self.alpha * x)

    def backward(self, grad):
        return grad * np.where(self._x > 0, 1.0, self.alpha)

    def parameters(self): return []
    def __repr__(self): return f'LeakyReLU(alpha={self.alpha})'


class GELU:
    """Gaussian Error Linear Unit — tanh approximation used in GPT/BERT."""
    _C = 0.7978845608028654   # sqrt(2/pi)

    def forward(self, x):
        self._x = x
        inner = self._C * (x + 0.044715 * x ** 3)
        self._tanh = np.tanh(inner)
        self._cdf  = 0.5 * (1.0 + self._tanh)
        return x * self._cdf

    def backward(self, grad):
        x     = self._x
        inner = self._C * (x + 0.044715 * x ** 3)
        dtanh = 1.0 - self._tanh ** 2
        dcdf  = 0.5 * dtanh * self._C * (1.0 + 3 * 0.044715 * x ** 2)
        return grad * (self._cdf + x * dcdf)

    def parameters(self): return []
    def __repr__(self): return 'GELU()'


class Sigmoid:
    def forward(self, x):
        self._out = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self._out

    def backward(self, grad):
        return grad * self._out * (1.0 - self._out)

    def parameters(self): return []
    def __repr__(self): return 'Sigmoid()'


class Tanh:
    def forward(self, x):
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad):
        return grad * (1.0 - self._out ** 2)

    def parameters(self): return []
    def __repr__(self): return 'Tanh()'


class Softmax:
    def forward(self, x):
        e = np.exp(x - x.max(axis=1, keepdims=True))
        self._out = e / e.sum(axis=1, keepdims=True)
        return self._out

    def backward(self, grad):
        return grad

    def parameters(self): return []
    def __repr__(self): return 'Softmax()'


# ── Layers ────────────────────────────────────────────────────────────────────

class Dense:
    def __init__(self, in_features, out_features, bias=True):
        # He initialisation for ReLU-family activations
        scale = np.sqrt(2.0 / in_features)
        self.W  = np.random.randn(in_features, out_features) * scale
        self.b  = np.zeros((1, out_features)) if bias else None
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        self._bias = bias

    def forward(self, x):
        self._x = x
        out = x @ self.W
        if self._bias:
            out = out + self.b
        return out

    def backward(self, grad):
        self.dW = self._x.T @ grad
        if self._bias:
            self.db = grad.sum(axis=0, keepdims=True)
        return grad @ self.W.T

    def parameters(self):
        p = [(self.W, self.dW)]
        if self._bias:
            p.append((self.b, self.db))
        return p

    def __repr__(self):
        return f'Dense({self.W.shape[0]}, {self.W.shape[1]})'


class BatchNorm1d:
    """
    Batch normalisation with running stats for eval mode.
    During training: normalise over the batch dimension.
    During eval: use exponential moving averages accumulated during training.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        self.gamma = np.ones((1, num_features))
        self.beta  = np.zeros((1, num_features))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)
        self.eps      = eps
        self.momentum = momentum
        self.running_mean = np.zeros((1, num_features))
        self.running_var  = np.ones((1, num_features))
        self.training = True

    def forward(self, x):
        if self.training:
            mean = x.mean(axis=0, keepdims=True)
            var  = x.var(axis=0, keepdims=True)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * var
        else:
            mean = self.running_mean
            var  = self.running_var

        self._x_hat = (x - mean) / np.sqrt(var + self.eps)
        self._var   = var
        self._N     = x.shape[0]
        return self.gamma * self._x_hat + self.beta

    def backward(self, grad):
        N   = self._N
        std = np.sqrt(self._var + self.eps)
        self.dgamma = (grad * self._x_hat).sum(axis=0, keepdims=True)
        self.dbeta  = grad.sum(axis=0, keepdims=True)
        dx_hat = grad * self.gamma
        # Full derivative of batch norm (not approximate)
        dvar   = (-0.5 * (dx_hat * self._x_hat).sum(axis=0, keepdims=True) / (self._var + self.eps))
        dmean  = (-dx_hat / std).sum(axis=0, keepdims=True)
        return dx_hat / std + (2 * dvar * self._x_hat * std) / N + dmean / N

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def __repr__(self):
        return f'BatchNorm1d({self.gamma.shape[1]})'


class LayerNorm:
    """Layer normalisation — normalises across the feature dimension per sample."""
    def __init__(self, normalized_shape, eps=1e-5):
        self.gamma  = np.ones((1, normalized_shape))
        self.beta   = np.zeros((1, normalized_shape))
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)
        self.eps    = eps

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var  = x.var(axis=-1, keepdims=True)
        self._x_hat = (x - mean) / np.sqrt(var + self.eps)
        self._var   = var
        return self.gamma * self._x_hat + self.beta

    def backward(self, grad):
        D   = self._x_hat.shape[-1]
        std = np.sqrt(self._var + self.eps)
        self.dgamma = (grad * self._x_hat).sum(axis=0, keepdims=True)
        self.dbeta  = grad.sum(axis=0, keepdims=True)
        dx_hat = grad * self.gamma
        return (1.0 / D) * (
            D * dx_hat
            - dx_hat.sum(axis=-1, keepdims=True)
            - self._x_hat * (dx_hat * self._x_hat).sum(axis=-1, keepdims=True)
        ) / std

    def parameters(self):
        return [(self.gamma, self.dgamma), (self.beta, self.dbeta)]

    def __repr__(self):
        return f'LayerNorm({self.gamma.shape[1]})'


class Dropout:
    """Inverted dropout — scales surviving units so inference needs no change."""
    def __init__(self, p=0.5):
        self.p        = p
        self.training = True
        self._mask    = None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        self._mask = (np.random.rand(*x.shape) > self.p) / (1.0 - self.p)
        return x * self._mask

    def backward(self, grad):
        if not self.training or self.p == 0.0:
            return grad
        return grad * self._mask

    def parameters(self): return []
    def __repr__(self): return f'Dropout(p={self.p})'


class Flatten:
    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad):
        return grad.reshape(self._shape)

    def parameters(self): return []
    def __repr__(self): return 'Flatten()'


# ── Loss functions ────────────────────────────────────────────────────────────

class CrossEntropyLoss:
    """Fused softmax + cross-entropy in one pass for numerical stability."""
    def __call__(self, logits, y_true):
        N = logits.shape[0]
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = e / e.sum(axis=1, keepdims=True)
        grad  = probs.copy()
        grad[np.arange(N), y_true] -= 1
        grad /= N
        loss = -np.log(probs[np.arange(N), y_true] + 1e-12).mean()
        return loss, grad


class MSELoss:
    def __call__(self, pred, y_true):
        diff = pred - y_true
        loss = (diff ** 2).mean()
        grad = 2.0 * diff / y_true.size
        return loss, grad


class BCELoss:
    """Binary cross-entropy for sigmoid outputs."""
    def __call__(self, pred, y_true):
        pred = np.clip(pred, 1e-12, 1.0 - 1e-12)
        loss = -(y_true * np.log(pred) + (1.0 - y_true) * np.log(1.0 - pred)).mean()
        grad = (pred - y_true) / (pred * (1.0 - pred) * y_true.size)
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

    def train(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True

    def eval(self):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False

    def save(self, path):
        arrays = {}
        for i, layer in enumerate(self.layers):
            for j, (param, _) in enumerate(layer.parameters()):
                arrays[f'l{i}_p{j}'] = param
        np.savez(path, **arrays)
        print(f'  Saved to {path}.npz')

    def load(self, path):
        data = np.load(path if path.endswith('.npz') else path + '.npz')
        for i, layer in enumerate(self.layers):
            for j, (param, _) in enumerate(layer.parameters()):
                key = f'l{i}_p{j}'
                if key in data:
                    param[:] = data[key]

    def summary(self):
        total = 0
        print(f'\n  {"Layer":<22} {"Params":>8}')
        print(f'  {"─"*22} {"─"*8}')
        for layer in self.layers:
            n = sum(p.size for p, _ in layer.parameters())
            total += n
            print(f'  {repr(layer):<22} {n:>8,}')
        print(f'  {"─"*30}')
        print(f'  {"Total":<22} {total:>8,}\n')

    def __repr__(self):
        lines = '\n'.join(f'  ({i}): {repr(l)}' for i, l in enumerate(self.layers))
        return f'Sequential(\n{lines}\n)'
