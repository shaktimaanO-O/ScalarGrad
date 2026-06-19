import random

from .engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def parameters(self):
        return []


def apply_activation(value, activation):
    if activation == "tanh":
        return value.tanh()
    if activation == "relu":
        return value.relu()
    if activation == "sigmoid":
        return value.sigmoid()
    if activation in ("linear", None):
        return value
    raise ValueError(f"unsupported activation: {activation}")


class Neuron(Module):
    def __init__(self, nin, activation="tanh"):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
        self.activation = activation

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return apply_activation(act, self.activation)

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation="tanh"):
        self.neurons = [Neuron(nin, activation=activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activation="tanh", output_activation="tanh"):
        sizes = [nin] + nouts
        self.layers = []
        for i in range(len(nouts)):
            is_output = i == len(nouts) - 1
            layer_activation = output_activation if is_output else activation
            self.layers.append(
                Layer(sizes[i], sizes[i + 1], activation=layer_activation)
            )

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
