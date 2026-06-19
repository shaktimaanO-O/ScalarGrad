import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalargrad import Adam, BinaryMLPClassifier
from scalargrad.datasets import make_noisy_xor


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    xs, ys = make_noisy_xor(repeats_per_point=100, noise_rate=0.25, seed=42)

    random.seed(42)
    model = BinaryMLPClassifier(2, [4, 4], activation="tanh")
    optimizer = Adam(model.parameters(), lr=0.03)

    final_loss = None
    for epoch in range(100):
        loss = model.loss(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = loss.data

        if epoch % 10 == 0:
            print(f"epoch {epoch:02d} loss {loss.data:.4f} acc {model.accuracy(xs, ys):.4f}")

    final_accuracy = model.accuracy(xs, ys)
    with open(OUTPUT_DIR / "noisy_xor_sanity_check.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "noise_rate", "final_loss", "final_accuracy"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "dataset": "noisy_xor",
                "noise_rate": 0.25,
                "final_loss": final_loss,
                "final_accuracy": final_accuracy,
            }
        )


if __name__ == "__main__":
    main()
