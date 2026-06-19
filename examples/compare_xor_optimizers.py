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

from scalargrad import Adam, BinaryMLPClassifier, Momentum, RMSProp, SGD
from scalargrad.datasets import make_xor
from scalargrad.plotting import plot_decision_boundary


OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def train_classifier(model, optimizer, xs, ys, epochs=250):
    history = {"loss": [], "accuracy": []}

    for _ in range(epochs):
        loss = model.loss(xs, ys)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history["loss"].append(loss.data)
        history["accuracy"].append(model.accuracy(xs, ys))

    return history


def save_curves(histories, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["optimizer", "epoch", "loss", "accuracy"])
        writer.writeheader()
        for optimizer, history in histories.items():
            for epoch, (loss, accuracy) in enumerate(
                zip(history["loss"], history["accuracy"])
            ):
                writer.writerow(
                    {
                        "optimizer": optimizer,
                        "epoch": epoch,
                        "loss": loss,
                        "accuracy": accuracy,
                    }
                )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    xs, ys = make_xor()
    optimizer_builders = {
        "SGD": lambda params: SGD(params, lr=0.1),
        "Momentum": lambda params: Momentum(params, lr=0.05, momentum=0.9),
        "RMSProp": lambda params: RMSProp(params, lr=0.02),
        "Adam": lambda params: Adam(params, lr=0.03),
    }

    histories = {}
    final_models = {}
    for name, build_optimizer in optimizer_builders.items():
        random.seed(42)
        model = BinaryMLPClassifier(2, [4, 4], activation="tanh")
        optimizer = build_optimizer(model.parameters())
        histories[name] = train_classifier(model, optimizer, xs, ys)
        final_models[name] = model
        print(
            f"{name:8s} final loss {histories[name]['loss'][-1]:.4f} "
            f"accuracy {histories[name]['accuracy'][-1]:.2f}"
        )

    save_curves(histories, OUTPUT_DIR / "xor_optimizer_curves.csv")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    for name, history in histories.items():
        plt.plot(history["loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("XOR Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    for name, history in histories.items():
        plt.plot(history["accuracy"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("XOR Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "xor_optimizer_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    plot_decision_boundary(
        final_models["Adam"],
        xs,
        ys,
        title="XOR Decision Boundary",
        save_path=OUTPUT_DIR / "xor_decision_boundary.png",
    )
    plt.close()


if __name__ == "__main__":
    main()
