import csv
import json

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import AMP_PATH, DEVICE, DI_PATH, EXPERIMENTS, AMP_VAL_PATH, DI_VAL_PATH, BaseConfig
from dataset import LongAudioDataset
from loss import CombinedLoss
#from models.AmpTCN import AmpTCN
from plotting import plot_metrics


def make_scheduler(optimizer, config: BaseConfig):
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


@torch.no_grad()
def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_sum = esr_sum = spec_sum = 0.0

    for x, y in val_loader:
        x, y = x.to(device).float(), y.to(device).float()
        pred = model(x)
        loss, esr, spec = loss_fn(pred, y)

        total_sum += loss.item()
        esr_sum += esr.item()
        spec_sum += spec.item()

    n = len(val_loader)
    return total_sum / n, esr_sum / n, spec_sum / n


def train_one(config: BaseConfig, return_model=False):
    run_dir = config.make_run_dir()

    with open(run_dir / "hparams.json", "w") as f:
        json.dump({**config.to_dict(), "device": DEVICE}, f, indent=2)

    print(f"\n{'=' * 60}\nRun: {run_dir.name}\n{'=' * 60}")

    train_dataset = LongAudioDataset(
        str(DI_PATH), str(AMP_PATH),
        segment_length=config.segment_length,
        dataset_size=config.train_dataset_size,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    val_dataset = LongAudioDataset(
        str(DI_VAL_PATH), str(AMP_VAL_PATH),
        segment_length=config.segment_length,
        is_val=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, drop_last=False
    )

    model = config.create_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    scheduler = make_scheduler(optimizer, config)
    loss_fn = CombinedLoss().to(DEVICE)

    metrics = {
        "epoch": [],
        "train_total": [], "train_esr": [], "train_spec": [],
        "val_total": [], "val_esr": [], "val_spec": [],
    }
    best_val_loss = float("inf")

    early_stop_patience = 10
    epochs_without_improvement = 0

    for epoch in range(config.epochs):
        model.train()
        total_sum = esr_sum = spec_sum = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
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

        avg_train_total = total_sum / len(train_loader)
        avg_train_esr = esr_sum / len(train_loader)
        avg_train_spec = spec_sum / len(train_loader)

        avg_val_total, avg_val_esr, avg_val_spec = validate(
            model, val_loader, loss_fn, DEVICE
        )

        step_scheduler(scheduler, avg_val_total)

        metrics["epoch"].append(epoch + 1)
        metrics["train_total"].append(avg_train_total)
        metrics["train_esr"].append(avg_train_esr)
        metrics["train_spec"].append(avg_train_spec)
        metrics["val_total"].append(avg_val_total)
        metrics["val_esr"].append(avg_val_esr)
        metrics["val_spec"].append(avg_val_spec)

        print(
            f"Epoch {epoch + 1}: "
            f"train_total={avg_train_total:.6f} train_esr={avg_train_esr:.6f} | "
            f"val_total={avg_val_total:.6f} val_esr={avg_val_esr:.6f}"
        )

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            epochs_without_improvement = 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            print(f"New best model (val_loss={avg_val_total:.6f})")
        else:
            epochs_without_improvement += 1
            print(
                f"  No improvement for {epochs_without_improvement}/{early_stop_patience} epochs"
            )
            if epochs_without_improvement >= early_stop_patience:
                print(f"Early stopping triggered after epoch {epoch + 1}.")
                break

        torch.save(model.state_dict(), run_dir / "last_model.pt")

    csv_path = run_dir / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_total", "train_esr", "train_spec",
            "val_total", "val_esr", "val_spec",
        ])
        for i in range(len(metrics["epoch"])):
            writer.writerow([
                metrics["epoch"][i],
                metrics["train_total"][i],
                metrics["train_esr"][i],
                metrics["train_spec"][i],
                metrics["val_total"][i],
                metrics["val_esr"][i],
                metrics["val_spec"][i],
            ])

    plot_metrics(metrics, run_dir)
    print(f"Done → {run_dir}")
    if return_model:
        # Load best model
        model.load_state_dict(torch.load(run_dir / "best_model.pt"))
        return best_val_loss, model

    return best_val_loss


def main():
    for i, config in enumerate(EXPERIMENTS):
        print(f"\nExperiment {i + 1}/{len(EXPERIMENTS)}: {config.slug()}")
        train_one(config)


if __name__ == "__main__":
    main()