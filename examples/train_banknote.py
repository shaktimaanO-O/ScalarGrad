import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for this example. Run: pip install -r requirements.txt"
    ) from exc

from scalargrad import Adam, BinaryMLPClassifier
from scalargrad.datasets import (
    deduplicate_rows,
    load_banknote,
    make_batches,
    standardize_train_test,
    train_test_split,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    xs, ys = load_banknote(ROOT / "data" / "banknote_authentication.txt")
    xs, ys = deduplicate_rows(xs, ys)
    train_xs, train_ys, test_xs, test_ys = train_test_split(xs, ys, seed=42)
    train_xs, test_xs = standardize_train_test(train_xs, test_xs)

    random.seed(42)
    model = BinaryMLPClassifier(4, [8, 4], activation="tanh")
    optimizer = Adam(model.parameters(), lr=0.01)
    history = {"train_loss": [], "train_accuracy": [], "test_accuracy": []}

    for epoch in range(31):
        batch_losses = []
        for batch_xs, batch_ys in make_batches(
            train_xs, train_ys, batch_size=32, seed=epoch
        ):
            loss = model.loss(batch_xs, batch_ys)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.data)

        avg_loss = sum(batch_losses) / len(batch_losses)
        train_acc = model.accuracy(train_xs, train_ys)
        test_acc = model.accuracy(test_xs, test_ys)
        history["train_loss"].append(avg_loss)
        history["train_accuracy"].append(train_acc)
        history["test_accuracy"].append(test_acc)

        if epoch % 5 == 0:
            print(
                f"epoch {epoch:02d} loss {avg_loss:.4f} "
                f"train_acc {train_acc:.4f} test_acc {test_acc:.4f}"
            )

    with open(OUTPUT_DIR / "banknote_adam_training.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_accuracy", "test_accuracy"]
        )
        writer.writeheader()
        for epoch in range(len(history["train_loss"])):
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": history["train_loss"][epoch],
                    "train_accuracy": history["train_accuracy"][epoch],
                    "test_accuracy": history["test_accuracy"][epoch],
                }
            )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Banknote Adam Train Loss")

    plt.subplot(1, 2, 2)
    plt.plot(history["train_accuracy"], label="train")
    plt.plot(history["test_accuracy"], label="test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Banknote Adam Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "banknote_adam_training.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
