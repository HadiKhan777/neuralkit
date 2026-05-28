import numpy as np


class SGD:
    def __init__(self, lr=0.01, momentum=0.9, weight_decay=0.0, nesterov=False):
        self.lr           = lr
        self.momentum     = momentum
        self.weight_decay = weight_decay
        self.nesterov     = nesterov
        self._v           = {}

    def step(self, parameters):
        for i, (param, grad) in enumerate(parameters):
            g = grad + self.weight_decay * param if self.weight_decay else grad
            if self.momentum:
                v = self.momentum * self._v.get(i, np.zeros_like(param)) - self.lr * g
                self._v[i] = v
                param += (self.momentum * v - self.lr * g) if self.nesterov else v
            else:
                param -= self.lr * g

    def zero_grad(self, parameters):
        for _, grad in parameters:
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
            g = grad + self.weight_decay * param if self.weight_decay else grad
            m = self.beta1 * self._m.get(i, np.zeros_like(param)) + (1 - self.beta1) * g
            v = self.beta2 * self._v.get(i, np.zeros_like(param)) + (1 - self.beta2) * g ** 2
            self._m[i], self._v[i] = m, v
            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, parameters):
        for _, grad in parameters:
            grad[:] = 0


class AdamW:
    """
    Adam with decoupled weight decay (Loshchilov & Hutter, 2019).
    Weight decay is applied directly to params rather than folded into gradient,
    which regularises differently from L2 and works better with adaptive lr.
    """
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
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
            m = self.beta1 * self._m.get(i, np.zeros_like(param)) + (1 - self.beta1) * grad
            v = self.beta2 * self._v.get(i, np.zeros_like(param)) + (1 - self.beta2) * grad ** 2
            self._m[i], self._v[i] = m, v
            m_hat = m / (1 - self.beta1 ** self._t)
            v_hat = v / (1 - self.beta2 ** self._t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            if self.weight_decay:
                param -= self.lr * self.weight_decay * param  # decoupled

    def zero_grad(self, parameters):
        for _, grad in parameters:
            grad[:] = 0


# ── LR Schedulers ─────────────────────────────────────────────────────────────

class LRScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self._base_lr  = optimizer.lr
        self._step     = 0

    def step(self):
        self._step += 1
        self.optimizer.lr = self._get_lr()

    def _get_lr(self):
        raise NotImplementedError

    def get_lr(self):
        return self.optimizer.lr


class StepLR(LRScheduler):
    """Multiply lr by gamma every step_size epochs."""
    def __init__(self, optimizer, step_size=10, gamma=0.5):
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma     = gamma

    def _get_lr(self):
        return self._base_lr * (self.gamma ** (self._step // self.step_size))


class CosineAnnealingLR(LRScheduler):
    """Cosine decay from base_lr to eta_min over T_max steps."""
    def __init__(self, optimizer, T_max=100, eta_min=1e-6):
        super().__init__(optimizer)
        self.T_max   = T_max
        self.eta_min = eta_min

    def _get_lr(self):
        t   = min(self._step, self.T_max)
        cos = np.cos(np.pi * t / self.T_max)
        return self.eta_min + 0.5 * (self._base_lr - self.eta_min) * (1.0 + cos)


class WarmupCosineScheduler(LRScheduler):
    """Linear warmup → cosine decay. Standard schedule for transformers."""
    def __init__(self, optimizer, warmup_steps=10, T_max=100, eta_min=1e-6):
        super().__init__(optimizer)
        self.warmup_steps = warmup_steps
        self.T_max        = T_max
        self.eta_min      = eta_min

    def _get_lr(self):
        if self._step <= self.warmup_steps:
            return self._base_lr * self._step / max(1, self.warmup_steps)
        t   = self._step - self.warmup_steps
        T   = self.T_max - self.warmup_steps
        cos = np.cos(np.pi * t / max(1, T))
        return self.eta_min + 0.5 * (self._base_lr - self.eta_min) * (1.0 + cos)
