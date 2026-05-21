import os

import matplotlib.pyplot as plt

from config import RUN_DIR


def plot_metrics(metrics):
    epochs = metrics["epoch"]

    plt.figure(figsize=(10, 6))

    plt.plot(epochs, metrics["total"], label="Total")
    plt.plot(epochs, metrics["esr"], label="ESR")
    plt.plot(epochs, metrics["spec"], label="Spectral")

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Training Curves")
    plt.legend()
    plt.grid()

    out_path = os.path.join(RUN_DIR, "training_curve.png")
    plt.savefig(out_path, dpi=200)

    print(f"Saved plot → {out_path}")
