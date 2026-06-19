import random

from scalargrad import Adam, BinaryMLPClassifier, Value, binary_cross_entropy
from scalargrad.datasets import make_noisy_xor, make_xor


def test_binary_cross_entropy_is_low_for_correct_confident_prediction():
    loss = binary_cross_entropy(Value(0.99), 1.0)

    assert loss.data < 0.02


def test_binary_classifier_learns_xor():
    xs, ys = make_xor()
    random.seed(42)
    model = BinaryMLPClassifier(2, [4, 4], activation="tanh")
    optimizer = Adam(model.parameters(), lr=0.03)

    for _ in range(250):
        loss = model.loss(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert model.accuracy(xs, ys) == 1.0


def test_noisy_xor_saturates_below_perfect_accuracy():
    xs, ys = make_noisy_xor(repeats_per_point=100, noise_rate=0.25, seed=42)
    random.seed(42)
    model = BinaryMLPClassifier(2, [4, 4], activation="tanh")
    optimizer = Adam(model.parameters(), lr=0.03)

    for _ in range(100):
        loss = model.loss(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert 0.70 <= model.accuracy(xs, ys) <= 0.80
