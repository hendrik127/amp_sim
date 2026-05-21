import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    AMP_PATH,
    BATCH_SIZE,
    CHANNELS,
    DEVICE,
    DI_PATH,
    DILATIONS,
    EPOCHS,
    LR,
    RUN_DIR,
    SEGMENT_LENGTH,
    STACKS,
)
from dataset import LongAudioDataset
from loss import CombinedLoss
from model import AmpTCN
from plotting import plot_metrics


def main():
    dataset = LongAudioDataset(
        DI_PATH,
        AMP_PATH,
        SEGMENT_LENGTH,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = AmpTCN(CHANNELS, DILATIONS, STACKS).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    loss_fn = CombinedLoss().to(DEVICE)

    metrics = {
        "epoch": [],
        "total": [],
        "esr": [],
        "spec": [],
    }

    best_loss = float("inf")

    # ========================================================
    # Epoch loop
    # ========================================================

    for epoch in range(EPOCHS):
        model.train()

        total_sum = 0
        esr_sum = 0
        spec_sum = 0

        loader_tqdm = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for x, y in loader_tqdm:
            x = x.to(DEVICE).float()
            y = y.to(DEVICE).float()

            pred = model(x)

            loss, esr, spec = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_sum += loss.item()
            esr_sum += esr.item()
            spec_sum += spec.item()

            loader_tqdm.set_postfix(
                {
                    "loss": loss.item(),
                    "esr": esr.item(),
                    "spec": spec.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        n = len(loader)

        avg_total = total_sum / n
        avg_esr = esr_sum / n
        avg_spec = spec_sum / n

        scheduler.step(avg_total)

        metrics["epoch"].append(epoch + 1)
        metrics["total"].append(avg_total)
        metrics["esr"].append(avg_esr)
        metrics["spec"].append(avg_spec)

        print(f"\nEpoch {epoch + 1} summary:")
        print(f" total={avg_total:.6f}")
        print(f" esr  ={avg_esr:.6f}")
        print(f" spec ={avg_spec:.6f}")

        # Save best model
        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "best_model.pt"))

        # Save last model
        torch.save(model.state_dict(), os.path.join(RUN_DIR, "last_model.pt"))

    # ========================================================
    # Save CSV
    # ========================================================

    csv_path = os.path.join(RUN_DIR, "metrics.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "total", "esr", "spec"])

        for i in range(len(metrics["epoch"])):
            writer.writerow(
                [
                    metrics["epoch"][i],
                    metrics["total"][i],
                    metrics["esr"][i],
                    metrics["spec"][i],
                ]
            )

    print(f"Saved metrics → {csv_path}")

    # ========================================================
    # Plot
    # ========================================================

    plot_metrics(metrics)

    print("Training complete.")


if __name__ == "__main__":
    main()
