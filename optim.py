import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self._velocity    = {}

    def step(self, parameters):
        for i, (param, grad) in enumerate(parameters):
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            if self.momentum:
                v = self._velocity.get(i, np.zeros_like(param))
                v = self.momentum * v - self.lr * grad
                self._velocity[i] = v
                param += v
            else:
                param -= self.lr * grad

    def zero_grad(self, parameters):
        for param, grad in parameters:
            grad[:] = 0


class Adam:
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr           = lr
        self.beta1        = beta1
        self.beta2        = beta2
        self.eps          = eps
        self.weight_decay = weight_decay
        self._m  = {}
        self._v  = {}
        self._t  = 0

    def step(self, parameters):
        self._t += 1
        for i, (param, grad) in enumerate(parameters):
            if self.weight_decay:
                grad = grad + self.weight_decay * param
            m = self._m.get(i, np.zeros_like(param))
            v = self._v.get(i, np.zeros_like(param))
            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * grad ** 2
            self._m[i] = m
            self._v[i] = v
            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, parameters):
        for param, grad in parameters:
            grad[:] = 0
