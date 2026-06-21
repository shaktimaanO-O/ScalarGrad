<p align="center">
  <img src="assets/scalargrad-logo.svg" alt="ScalarGrad logo" width="520">
</p>

<h1 align="center">ScalarGrad</h1>

<p align="center">
  A tiny scalar automatic differentiation engine and neural-network toolkit built from scratch in Python.
</p>

<p align="center">
  <img alt="Python 3.9+" src="https://img.shields.io/badge/python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="Tests: pytest" src="https://img.shields.io/badge/tests-pytest-0A9EDC?style=flat-square">
  <img alt="Package: pure Python" src="https://img.shields.io/badge/package-pure%20Python-16A34A?style=flat-square">
</p>

ScalarGrad is a compact learning-first implementation of reverse-mode autodiff. It starts with a micrograd-style `Value` object, then layers on neural-network modules, binary classification, optimizers, datasets, plots, and reproducible examples.

It is small enough to read in one sitting, but complete enough to train XOR models, compare optimizers, visualize decision boundaries, and run a real Banknote Authentication classifier.

## Features

- Reverse-mode automatic differentiation over scalar computation graphs
- A minimal neural-network stack: `Module`, `Neuron`, `Layer`, and `MLP`
- Activations: `tanh`, `relu`, `sigmoid`, and `linear`
- `BinaryMLPClassifier` with sigmoid probabilities and binary cross entropy
- Optimizers: `SGD`, `Momentum`, `RMSProp`, and `Adam`
- Dataset helpers for XOR, noisy XOR, batching, train/test splits, and standardization
- Graphviz computation graph visualization
- Reproducible examples that save CSV metrics and PNG plots to `outputs/`

## Project Layout

```text
scalargrad/
  engine.py        # scalar Value type and reverse-mode autodiff
  nn.py            # Module, Neuron, Layer, MLP
  classifier.py    # BinaryMLPClassifier and BCE loss
  optim.py         # SGD, Momentum, RMSProp, Adam
  datasets.py      # XOR, noisy XOR, banknote loading, splits, scaling
  plotting.py      # decision boundary plotting
  visualize.py     # Graphviz computation graph rendering
examples/
  train_mlp.py
  compare_xor_optimizers.py
  train_banknote.py
  noisy_xor_sanity_check.py
tests/
  test_engine.py
  test_classifier.py
  test_optim.py
notebooks/
  ScalarGrad.ipynb
data/
  banknote_authentication.txt
outputs/
  *.csv, *.png
```

## Installation

ScalarGrad supports Python 3.9+.

```bash
git clone https://github.com/shaktimaanO-O/ScalarGrad.git
cd ScalarGrad
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install the local package and example dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e . -r requirements.txt
```

Graph rendering uses the `graphviz` Python package and the Graphviz system executable. If you use `scalargrad.visualize`, make sure `dot` is available on your `PATH`.

## Quick Start

Run the test suite:

```bash
python -m pytest
```

Train the toy MLP demo:

```bash
python examples/train_mlp.py
```

Compare optimizers on XOR and write plots/metrics to `outputs/`:

```bash
python examples/compare_xor_optimizers.py
```

Train the Banknote Authentication classifier:

```bash
python examples/train_banknote.py
```

Run the noisy-label XOR sanity check:

```bash
python examples/noisy_xor_sanity_check.py
```

## Usage

### Minimal Autograd

```python
from scalargrad import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + 10

c.backward()

print(c.data)  # 4.0
print(a.grad)  # -3.0
print(b.grad)  # 2.0
```

### Binary Classification

```python
import random

from scalargrad import Adam, BinaryMLPClassifier
from scalargrad.datasets import make_xor

xs, ys = make_xor()

random.seed(42)
model = BinaryMLPClassifier(2, [4, 4], activation="tanh")
optimizer = Adam(model.parameters(), lr=0.03)

for _ in range(250):
    loss = model.loss(xs, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print([model.predict(x) for x in xs])
```

## Dataset Source

The Banknote Authentication example uses the [UCI Machine Learning Repository Banknote Authentication dataset](https://archive.ics.uci.edu/dataset/267/banknote+authentication). UCI lists the dataset as donated on April 15, 2013 by Volker Lohweg, with 1,372 instances, four real-valued features extracted from wavelet-transformed banknote images (`variance`, `skewness`, `curtosis`, `entropy`), and an integer class target.


## Results

The checked-in example outputs are generated with fixed seeds for reproducibility.

| Experiment | Command | Result |
| --- | --- | --- |
| XOR optimizer comparison | `python examples/compare_xor_optimizers.py` | All optimizers reach 100% accuracy on clean XOR; adaptive and momentum-based optimizers converge faster than plain SGD. |
| Banknote classifier | `python examples/train_banknote.py` | Adam reaches 100% train and test accuracy in the saved run after deduplication, train/test split, and train-only standardization. |
| Noisy XOR sanity check | `python examples/noisy_xor_sanity_check.py` | With 25% intentionally flipped labels, accuracy saturates near the expected noisy-label ceiling instead of reporting a misleading perfect score. |

## Visual Outputs

### XOR Optimizer Comparison

![XOR optimizer comparison](outputs/xor_optimizer_comparison.png)

### XOR Decision Boundary

![XOR decision boundary](outputs/xor_decision_boundary.png)

### Banknote Training Curves

![Banknote Adam training curves](outputs/banknote_adam_training.png)

## Public API

```python
from scalargrad import (
    Value,
    Module,
    Neuron,
    Layer,
    MLP,
    BinaryMLPClassifier,
    binary_cross_entropy,
    Optimizer,
    SGD,
    Momentum,
    RMSProp,
    Adam,
)
```

## Development

```bash
python -m pytest
python examples/compare_xor_optimizers.py
python examples/train_banknote.py
python examples/noisy_xor_sanity_check.py
```

Generated files are written to `outputs/`. The package itself has no runtime dependencies in `pyproject.toml`; plotting, testing, and visualization tools live in `requirements.txt`.
