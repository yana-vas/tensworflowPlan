import argparse
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.data import get_dataloader
from src.model import OccupancyNetwork
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 3DScan occupancy network")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def build_optimizer(model: nn.Module, lr: float, weight_decay: float) -> optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim == 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return optim.AdamW(groups, lr=lr)


def build_scheduler(
    optimizer: optim.Optimizer, warmup_steps: int, total_steps: int
) -> optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def validate(model: nn.Module, loader, criterion: nn.Module, device: torch.device) -> tuple:
    model.eval()
    total_loss, total_iou, n = 0.0, 0.0, 0
    for batch in loader:
        images = batch["image"].to(device)
        points = batch["points"].to(device)
        occupancy = batch["occupancy"].to(device)
        logits = model(images, points)
        total_loss += criterion(logits, occupancy).item()
        pred = (torch.sigmoid(logits) > 0.5)
        gt = (occupancy > 0.5)
        inter = (pred & gt).sum(dim=1).float()
        union = (pred | gt).sum(dim=1).float().clamp(min=1.0)
        total_iou += (inter / union).mean().item()
        n += 1
    return total_loss / max(n, 1), total_iou / max(n, 1)


def train_epoch(
    model, loader, criterion, optimizer, scheduler, scaler, device, epoch, grad_clip, use_amp
) -> float:
    model.train()
    if not any(p.requires_grad for p in model.encoder.parameters()):
        model.encoder.backbone.eval()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images = batch["image"].to(device)
        points = batch["points"].to(device)
        occupancy = batch["occupancy"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, points)
            loss = criterion(logits, occupancy)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
    return total_loss / max(n, 1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    epochs = args.epochs or config.training.num_epochs
    batch_size = args.batch_size or config.training.batch_size
    lr = args.lr or config.training.learning_rate
    use_amp = bool(config.training.amp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.type == "cuda"
    print(f"Device: {device} | AMP: {use_amp} | epochs: {epochs} | bs: {batch_size} | lr: {lr}")

    model = OccupancyNetwork.from_config(config).to(device)
    print(f"Trainable params: {model.get_num_params(trainable_only=True):,} / "
          f"{model.get_num_params(trainable_only=False):,} total")

    train_loader = get_dataloader(
        root=args.data_root, split="train", batch_size=batch_size,
        num_workers=config.training.num_workers, num_points=config.training.num_points,
        augment=config.training.augment, max_samples=args.max_samples,
    )
    val_loader = get_dataloader(
        root=args.data_root, split="val", batch_size=batch_size,
        num_workers=config.training.num_workers, num_points=config.training.num_points,
        augment=False, max_samples=(args.max_samples // 5 if args.max_samples else None),
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = build_optimizer(model, lr, config.training.weight_decay)
    total_steps = epochs * max(len(train_loader), 1)
    scheduler = build_scheduler(optimizer, config.training.warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch, best_iou = 0, -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", -1.0)
        print(f"Resumed from epoch {start_epoch}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    for epoch in range(start_epoch, epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device,
            epoch + 1, config.training.grad_clip, use_amp,
        )
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("iou/val", val_iou, epoch)
        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch + 1}/{epochs} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | val IoU {val_iou:.4f}")

        ckpt = {
            "epoch": epoch, "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "config": model.config_snapshot, "best_iou": best_iou,
            "train_loss": train_loss, "val_loss": val_loss, "val_iou": val_iou,
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  New best (val IoU {val_iou:.4f}) -> best.pt")

    writer.close()
    print(f"Done. Best val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()