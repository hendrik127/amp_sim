from pathlib import Path

import matplotlib.pyplot as plt


def plot_metrics(metrics, run_dir: Path):
    epochs = metrics["epoch"]

    plt.figure(figsize=(12, 8))
    
    # Train losses (solid lines)
    plt.plot(epochs, metrics["train_total"], label="Train Total", linewidth=2)
    plt.plot(epochs, metrics["train_esr"], label="Train ESR", linewidth=2)
    plt.plot(epochs, metrics["train_spec"], label="Train Spectral", linewidth=2)
    
    # Validation losses (dashed lines)
    plt.plot(epochs, metrics["val_total"], label="Val Total", linewidth=2, linestyle='--')
    plt.plot(epochs, metrics["val_esr"], label="Val ESR", linewidth=2, linestyle='--')
    plt.plot(epochs, metrics["val_spec"], label="Val Spectral", linewidth=2, linestyle='--')

    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (log)")
    plt.title("Training vs Validation Curves")
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

    out_path = run_dir / "training_curve.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved plot → {out_path}")