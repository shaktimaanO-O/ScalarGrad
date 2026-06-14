import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalargrad.nn import MLP


def main():
    random.seed(42)

    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    model = MLP(3, [4, 4, 1])

    for step in range(100):
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data -= 0.05 * p.grad

        if step % 10 == 0 or step == 99:
            print(f"step {step:03d} loss {loss.data:.4f}")

    print("predictions:", [round(y.data, 3) for y in ypred])


if __name__ == "__main__":
    main()
