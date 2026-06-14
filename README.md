# ScalarGrad

A tiny scalar-valued automatic differentiation engine and neural network library built from scratch in Python.

This project is inspired by the core idea behind deep learning frameworks: build a computation graph during the forward pass, then apply backpropagation to compute gradients automatically.

## What This Project Contains

- A `Value` class that supports scalar arithmetic and automatic differentiation
- Reverse-mode backpropagation over a computation graph
- Basic neural network modules: `Neuron`, `Layer`, and `MLP`
- Optional Graphviz visualization for computation graphs
- A small MLP training example on toy data
- Basic tests for engine behavior and model training

## Project Structure

```text
scalargrad/
  scalargrad/
    engine.py        # scalar autograd engine
    nn.py            # neuron, layer, and MLP classes
    visualize.py     # computation graph visualization
  examples/
    train_mlp.py     # small training demo
  tests/
    test_engine.py   # basic correctness checks
  requirements.txt
  pyproject.toml
  README.md
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the example:

```bash
python examples/train_mlp.py
```

Run tests:

```bash
python -m pytest
```

## Example

```python
from scalargrad import Value

a = Value(2.0)
b = Value(-3.0)
c = a * b + 10
c.backward()

print(c.data)
print(a.grad)
print(b.grad)
```

## Training A Tiny Neural Network

```python
from scalargrad.nn import MLP

model = MLP(3, [4, 4, 1])
x = [2.0, 3.0, -1.0]
y = model(x)
```

The example in `examples/train_mlp.py` trains a small MLP on four toy samples using mean squared error and gradient descent.

## Why This Matters

This project demonstrates:

- how computation graphs are constructed
- how local derivatives combine through the chain rule
- how backpropagation computes gradients
- how neural networks can be built from simple scalar operations
- how gradient descent updates model parameters

## Limitations

This is an educational engine. It works with scalar values and is intentionally small, so it is not designed to replace libraries like PyTorch or TensorFlow.

