import csv
import random


def make_xor():
    return (
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        [0.0, 1.0, 1.0, 0.0],
    )


def make_noisy_xor(repeats_per_point=100, noise_rate=0.25, seed=42):
    rng = random.Random(seed)
    base_xs, base_ys = make_xor()
    xs = []
    ys = []

    for x, y in zip(base_xs, base_ys):
        for _ in range(repeats_per_point):
            xs.append(list(x))
            ys.append(1.0 - y if rng.random() < noise_rate else y)

    return xs, ys


def load_banknote(path):
    xs = []
    ys = []

    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            values = [float(v) for v in row]
            xs.append(values[:4])
            ys.append(values[4])

    return xs, ys


def deduplicate_rows(xs, ys):
    seen = set()
    clean_xs = []
    clean_ys = []

    for x, y in zip(xs, ys):
        key = tuple(x)
        if key in seen:
            continue
        seen.add(key)
        clean_xs.append(x)
        clean_ys.append(y)

    return clean_xs, clean_ys


def train_test_split(xs, ys, test_size=0.2, seed=42):
    rng = random.Random(seed)
    indices = list(range(len(xs)))
    rng.shuffle(indices)
    split = int(len(xs) * (1 - test_size))
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_xs = [xs[i] for i in train_idx]
    train_ys = [ys[i] for i in train_idx]
    test_xs = [xs[i] for i in test_idx]
    test_ys = [ys[i] for i in test_idx]

    return train_xs, train_ys, test_xs, test_ys


def fit_standardizer(xs):
    n_features = len(xs[0])
    means = []
    stds = []

    for j in range(n_features):
        col = [x[j] for x in xs]
        mean = sum(col) / len(col)
        variance = sum((v - mean) ** 2 for v in col) / len(col)
        std = variance**0.5
        means.append(mean)
        stds.append(std if std > 0 else 1.0)

    return means, stds


def transform_standardizer(xs, means, stds):
    return [[(x[j] - means[j]) / stds[j] for j in range(len(x))] for x in xs]


def standardize_train_test(train_xs, test_xs):
    means, stds = fit_standardizer(train_xs)
    return (
        transform_standardizer(train_xs, means, stds),
        transform_standardizer(test_xs, means, stds),
    )


def make_batches(xs, ys, batch_size=32, seed=None):
    rng = random.Random(seed)
    indices = list(range(len(xs)))
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield [xs[i] for i in batch_indices], [ys[i] for i in batch_indices]
