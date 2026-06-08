import csv
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import AMP_PATH, DEVICE, DI_PATH, EXPERIMENTS, ExperimentConfig
from dataset import LongAudioDataset
from loss import CombinedLoss
from models.amp_tcn import AmpTCN
from plotting import plot_metrics


def make_scheduler(optimizer, config: ExperimentConfig):
    kw = config.scheduler_kwargs
    if config.scheduler == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=5, factor=0.1, **kw
        )
    if config.scheduler == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, **kw)
    if config.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, **kw
        )
    raise ValueError(f"Unknown scheduler: {config.scheduler!r}")


def step_scheduler(scheduler, loss: float):
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(loss)
    else:
        scheduler.step()


def train_one(config: ExperimentConfig):
    run_dir = config.make_run_dir()

    with open(run_dir / "hparams.json", "w") as f:
        json.dump({**config.to_dict(), "device": DEVICE}, f, indent=2)

    print(f"\n{'=' * 60}\nRun: {run_dir.name}\n{'=' * 60}")

    dataset = LongAudioDataset(str(DI_PATH), str(AMP_PATH), config.segment_length)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    model = AmpTCN(config.channels, config.dilations, config.stacks).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = make_scheduler(optimizer, config)
    loss_fn = CombinedLoss().to(DEVICE)

    metrics = {"epoch": [], "total": [], "esr": [], "spec": []}
    best_loss = float("inf")

    for epoch in range(config.epochs):
        model.train()
        total_sum = esr_sum = spec_sum = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for x, y in pbar:
            x, y = x.to(DEVICE).float(), y.to(DEVICE).float()
            pred = model(x)
            loss, esr, spec = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_sum += loss.item()
            esr_sum += esr.item()
            spec_sum += spec.item()
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                esr=f"{esr.item():.4f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}",
            )

        n = len(loader)
        avg_total, avg_esr, avg_spec = total_sum / n, esr_sum / n, spec_sum / n

        step_scheduler(scheduler, avg_total)

        metrics["epoch"].append(epoch + 1)
        metrics["total"].append(avg_total)
        metrics["esr"].append(avg_esr)
        metrics["spec"].append(avg_spec)

        print(
            f"Epoch {epoch + 1}  total={avg_total:.6f}  esr={avg_esr:.6f}  spec={avg_spec:.6f}"
        )

        if avg_total < best_loss:
            best_loss = avg_total
            torch.save(model.state_dict(), run_dir / "best_model.pt")
        torch.save(model.state_dict(), run_dir / "last_model.pt")

    csv_path = run_dir / "metrics.csv"
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

    plot_metrics(metrics, run_dir)
    print(f"Done → {run_dir}")


def main():
    for i, config in enumerate(EXPERIMENTS):
        print(f"\nExperiment {i + 1}/{len(EXPERIMENTS)}: {config.slug()}")
        train_one(config)


if __name__ == "__main__":
    main()
