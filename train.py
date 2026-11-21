"""Train and evaluate ExpressionNet on a FER-2013 formatted CSV."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn import functional as F
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from model import EMOTIONS, ExpressionNet, IMAGE_SIZE, load_checkpoint


class FERDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, augment: bool = False):
        self.labels = frame["emotion"].astype(int).to_numpy()
        self.images = np.stack([
            np.fromstring(pixel_string, sep=" ", dtype=np.uint8).reshape(IMAGE_SIZE, IMAGE_SIZE)
            for pixel_string in frame["pixels"].astype(str)
        ])
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
            if random.random() < 0.25:
                image = np.clip(image.astype(np.float32) * random.uniform(0.88, 1.12), 0, 255).astype(np.uint8)
        x = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        x = (x - 0.5) / 0.5
        return x, torch.tensor(self.labels[index], dtype=torch.long)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply small geometric and exposure changes without changing the label."""
    count = len(images)
    params = torch.rand(count, 7, device=images.device)
    angle = (2 * params[:, 0] - 1) * math.radians(10)
    scale = 0.90 + 0.20 * params[:, 1]
    horizontal = torch.where(params[:, 2] < 0.5, -scale, scale)
    cosine, sine = angle.cos(), angle.sin()
    translate_x = (2 * params[:, 3] - 1) * 0.12
    translate_y = (2 * params[:, 4] - 1) * 0.12
    transform = torch.stack((
        horizontal * cosine, -scale * sine, translate_x,
        horizontal * sine, scale * cosine, translate_y,
    ), dim=1).reshape(count, 2, 3)
    grid = F.affine_grid(transform, images.shape, align_corners=False)
    warped = F.grid_sample(images, grid, padding_mode="border", align_corners=False)
    contrast = (0.82 + 0.36 * params[:, 5]).view(count, 1, 1, 1)
    offset = (-0.12 + 0.24 * params[:, 6]).view(count, 1, 1, 1)
    return (warped * contrast + offset).clamp_(-1, 1)


def prepare_splits(frame: pd.DataFrame, seed: int):
    frame = frame.dropna(subset=["emotion", "pixels"]).copy()
    frame["emotion"] = frame["emotion"].astype(int)
    valid = frame["emotion"].between(0, len(EMOTIONS) - 1)
    frame = frame.loc[valid]
    if "Usage" in frame.columns:
        usage = frame["Usage"].astype(str).str.lower()
        train = frame[usage.eq("training")]
        validation = frame[usage.eq("publictest")]
        test = frame[usage.eq("privatetest")]
        if min(len(train), len(validation), len(test)) > 0:
            return train, validation, test
    train, rest = train_test_split(frame, test_size=0.2, stratify=frame.emotion, random_state=seed)
    validation, test = train_test_split(rest, test_size=0.5, stratify=rest.emotion, random_state=seed)
    return train, validation, test


@torch.no_grad()
def evaluate(model, loader, device, criterion=None):
    model.eval()
    truth, predicted, losses = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        if criterion is not None:
            losses.append(criterion(logits, labels).item())
        truth.extend(labels.cpu().numpy().tolist())
        predicted.extend(logits.argmax(1).cpu().numpy().tolist())
    return {
        "loss": float(np.mean(losses)) if losses else None,
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "truth": truth,
        "predicted": predicted,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/fer2013.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--recipe", choices=("baseline", "robust"), default="baseline",
                        help="Robust uses affine/exposure augmentation and exponential weight averaging.")
    parser.add_argument("--warm-start", type=Path,
                        help="Initialize weights from an existing ExpressionNet checkpoint.")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if not args.csv.exists():
        parser.error(f"CSV not found: {args.csv}. Download FER-2013 and place it there, or pass --csv.")
    checkpoint_path = args.output_dir / "best_model.pt"
    if args.warm_start:
        if not args.warm_start.is_file():
            parser.error("warm-start must point to an existing checkpoint")
        if args.warm_start.resolve() == checkpoint_path.resolve():
            parser.error("warm-start requires a separate output directory to preserve its source")
    robust = args.recipe == "robust"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    frame = pd.read_csv(args.csv)
    if not {"emotion", "pixels"}.issubset(frame.columns):
        parser.error("CSV must contain 'emotion' and 'pixels' columns")
    train_frame, val_frame, test_frame = prepare_splits(frame, args.seed)
    print(f"Rows — train: {len(train_frame):,}, validation: {len(val_frame):,}, test: {len(test_frame):,}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(FERDataset(train_frame, not robust), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(FERDataset(val_frame), batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(FERDataset(test_frame), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin_memory)
    counts = np.bincount(train_frame.emotion.to_numpy(), minlength=len(EMOTIONS))
    weights = len(train_frame) / (len(EMOTIONS) * np.maximum(counts, 1))
    if robust:
        weights = np.sqrt(weights)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=0.05)
    provenance = {"recipe": args.recipe}
    if args.warm_start:
        model, source = load_checkpoint(args.warm_start, device)
        provenance["warm_start"] = {
            "sha256": hashlib.sha256(args.warm_start.read_bytes()).hexdigest(),
            "epoch": source.get("epoch"),
        }
    else:
        model = ExpressionNet().to(device)
    use_channels_last = robust and pin_memory
    if use_channels_last:
        model.to(memory_format=torch.channels_last)
    averaged = (AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.995), use_buffers=True)
                if robust else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    scaler = torch.cuda.amp.GradScaler(enabled=pin_memory)
    best_f1, patience, wait = -1.0, 8, 0
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_macro_f1": []}
    minimum_accuracy = -1.0
    if args.warm_start:
        baseline = evaluate(model, val_loader, device, criterion)
        minimum_accuracy = baseline["accuracy"]
        best_f1 = baseline["macro_f1"]
        provenance["warm_start"]["validation"] = {"accuracy": minimum_accuracy, "macro_f1": best_f1}
        print(f"Source validation accuracy {minimum_accuracy:.4f} | macro-F1 {best_f1:.4f}")
        torch.save({"model_state_dict": model.state_dict(), "emotions": list(EMOTIONS),
                    "image_size": IMAGE_SIZE, "best_val_macro_f1": best_f1, "seed": args.seed,
                    "epoch": source.get("epoch"), "training": provenance, "fine_tune_epochs": 0},
                   checkpoint_path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []
        progress = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)
        for images, labels in progress:
            images, labels = images.to(device, non_blocking=pin_memory), labels.to(device, non_blocking=pin_memory)
            if robust:
                images = augment_batch(images)
            if use_channels_last:
                images = images.contiguous(memory_format=torch.channels_last)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=pin_memory):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            if averaged is not None:
                averaged.update_parameters(model)
            batch_losses.append(loss.item())
            progress.set_postfix(loss=f"{loss.item():.3f}")
        evaluated_model = averaged.module if averaged is not None else model
        val = evaluate(evaluated_model, val_loader, device, criterion)
        history["train_loss"].append(float(np.mean(batch_losses)))
        history["val_loss"].append(val["loss"])
        history["val_accuracy"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])
        scheduler.step(val["macro_f1"])
        print(f"Epoch {epoch:02d}: train loss {history['train_loss'][-1]:.4f} | "
              f"val loss {val['loss']:.4f} | val accuracy {val['accuracy']:.4f} | val macro-F1 {val['macro_f1']:.4f}")
        if val["macro_f1"] > best_f1 and val["accuracy"] > minimum_accuracy:
            best_f1, wait = val["macro_f1"], 0
            torch.save({"model_state_dict": evaluated_model.state_dict(), "emotions": list(EMOTIONS),
                        "image_size": IMAGE_SIZE, "best_val_macro_f1": best_f1,
                        "seed": args.seed, "epoch": (source.get("epoch") or 0) + epoch if args.warm_start else epoch,
                        "training": provenance, "fine_tune_epochs": epoch if args.warm_start else None},
                       checkpoint_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping: validation macro-F1 did not improve for {patience} epochs.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test = evaluate(model, test_loader, device)
    precision, recall, f1, support = precision_recall_fscore_support(
        test["truth"], test["predicted"], labels=range(len(EMOTIONS)), zero_division=0)
    metrics = {"test_accuracy": test["accuracy"], "test_macro_f1": test["macro_f1"],
               "best_validation_macro_f1": best_f1, "test_samples": len(test["truth"]),
               "training": provenance, "fine_tune_epochs": checkpoint.get("fine_tune_epochs"),
               "classes": list(EMOTIONS),
               "per_class": {name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(n)}
                             for name, p, r, f, n in zip(EMOTIONS, precision, recall, f1, support)}}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (args.output_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    pd.DataFrame(classification_report(test["truth"], test["predicted"], labels=range(len(EMOTIONS)),
                                       target_names=EMOTIONS, output_dict=True, zero_division=0)).T.to_csv(
                                           args.output_dir / "classification_report.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss"); axes[0].legend(); axes[0].grid(alpha=.2)
    axes[1].plot(history["val_macro_f1"], label="Validation macro-F1", color="#7c3aed")
    axes[1].plot(history["val_accuracy"], label="Validation accuracy", color="#0891b2")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1)); axes[1].legend(); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(args.output_dir / "training_curves.png", dpi=160); plt.close(fig)
    cm = confusion_matrix(test["truth"], test["predicted"], labels=range(len(EMOTIONS)))
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(cm, cmap="Purples"); fig.colorbar(image, ax=ax)
    ax.set(xticks=range(len(EMOTIONS)), yticks=range(len(EMOTIONS)), xticklabels=EMOTIONS,
           yticklabels=EMOTIONS, xlabel="Predicted label", ylabel="True label", title="Held-out test confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    threshold = cm.max() / 2 if cm.size else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    color="white" if cm[row, col] > threshold else "#201634", fontsize=8)
    fig.tight_layout(); fig.savefig(args.output_dir / "confusion_matrix.png", dpi=160); plt.close(fig)
    print(f"Held-out test accuracy: {test['accuracy']:.4f} | macro-F1: {test['macro_f1']:.4f}")
    print(f"Saved model and reports to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
