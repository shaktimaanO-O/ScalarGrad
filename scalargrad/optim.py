import math


class Optimizer:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = 0.0

    def step(self):
        raise NotImplementedError("subclasses must implement step")


class SGD(Optimizer):
    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad


class Momentum(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = {id(p): 0.0 for p in self.params}

    def step(self):
        for p in self.params:
            key = id(p)
            self.velocity[key] = self.momentum * self.velocity[key] + p.grad
            p.data -= self.lr * self.velocity[key]


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.001, beta=0.9, eps=1e-8):
        super().__init__(params, lr)
        self.beta = beta
        self.eps = eps
        self.square_avg = {id(p): 0.0 for p in self.params}

    def step(self):
        for p in self.params:
            key = id(p)
            self.square_avg[key] = (
                self.beta * self.square_avg[key] + (1 - self.beta) * p.grad**2
            )
            p.data -= self.lr * p.grad / (math.sqrt(self.square_avg[key]) + self.eps)


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {id(p): 0.0 for p in self.params}
        self.v = {id(p): 0.0 for p in self.params}

    def step(self):
        self.t += 1
        for p in self.params:
            key = id(p)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * p.grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * p.grad**2
            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)
