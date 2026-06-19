from .engine import Value
from .nn import MLP, Module


def binary_cross_entropy(y_pred, y_true):
    y_true = y_true if isinstance(y_true, Value) else Value(float(y_true))
    eps = 1e-12
    y_pred = y_pred * (1 - 2 * eps) + eps
    return -(y_true * y_pred.log() + (1 - y_true) * (1 - y_pred).log())


class BinaryMLPClassifier(Module):
    """Binary classifier backed by an MLP that emits one linear logit."""

    def __init__(self, nin, hidden_sizes, activation="tanh"):
        self.model = MLP(
            nin,
            hidden_sizes + [1],
            activation=activation,
            output_activation="linear",
        )

    def logit(self, x):
        return self.model(x)

    def predict_proba(self, x):
        return self.logit(x).sigmoid()

    def predict(self, x, threshold=0.5):
        return int(self.predict_proba(x).data >= threshold)

    def loss(self, xs, ys):
        losses = [
            binary_cross_entropy(self.predict_proba(x), y) for x, y in zip(xs, ys)
        ]
        return sum(losses) / len(losses)

    def accuracy(self, xs, ys):
        correct = sum(self.predict(x) == int(y) for x, y in zip(xs, ys))
        return correct / len(ys)

    def parameters(self):
        return self.model.parameters()
