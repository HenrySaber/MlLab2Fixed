from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from dataset_tools import HouseSegmentationDataset
from segmentation_utils import build_segmentation_model, compute_dice_score, compute_iou_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a house segmentation model on aerial imagery.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared dataset directory containing train/val/test splits.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--use-pretrained-backbone", action="store_true")
    return parser.parse_args()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    targets = targets.float()
    intersection = (probabilities * targets).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    score = (2 * intersection + eps) / (denominator + eps)
    return 1 - score.mean()


def batch_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    ious = []
    dices = []
    for prediction, target in zip(predictions, targets_np):
        ious.append(compute_iou_score(prediction, target))
        dices.append(compute_dice_score(prediction, target))
    return float(np.mean(ious)), float(np.mean(dices))


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer | None, device: torch.device) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(is_training)

    cross_entropy = nn.CrossEntropyLoss()
    running_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    step_count = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        if is_training:
            optimizer.zero_grad()

        logits = model(images)["out"]
        loss = cross_entropy(logits, masks) + dice_loss(logits, masks)

        if is_training:
            loss.backward()
            optimizer.step()

        iou, dice = batch_metrics(logits, masks)
        running_loss += loss.item()
        running_iou += iou
        running_dice += dice
        step_count += 1

    return {
        "loss": running_loss / max(step_count, 1),
        "iou": running_iou / max(step_count, 1),
        "dice": running_dice / max(step_count, 1),
    }


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_iou = [entry["train_iou"] for entry in history]
    val_iou = [entry["val_iou"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Validation Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()

    axes[1].plot(epochs, train_iou, label="Train IoU")
    axes[1].plot(epochs, val_iou, label="Validation IoU")
    axes[1].set_title("IoU Curves")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_prediction_grid(model: nn.Module, dataset: HouseSegmentationDataset, device: torch.device, output_path: Path, max_samples: int = 3) -> None:
    if len(dataset) == 0:
        return

    samples = min(max_samples, len(dataset))
    fig, axes = plt.subplots(samples, 3, figsize=(10, 4 * samples))
    if samples == 1:
        axes = np.expand_dims(axes, axis=0)

    model.eval()
    for row_index in range(samples):
        image_tensor, mask_tensor = dataset[row_index]
        image_batch = image_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            prediction = model(image_batch)["out"].argmax(dim=1).squeeze(0).cpu().numpy()

        image = np.transpose(image_tensor.numpy(), (1, 2, 0))
        axes[row_index][0].imshow(image)
        axes[row_index][0].set_title("Image")
        axes[row_index][0].axis("off")

        axes[row_index][1].imshow(mask_tensor.numpy(), cmap="gray")
        axes[row_index][1].set_title("Ground Truth")
        axes[row_index][1].axis("off")

        axes[row_index][2].imshow(prediction, cmap="gray")
        axes[row_index][2].set_title("Prediction")
        axes[row_index][2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = build_segmentation_model(use_pretrained_backbone=args.use_pretrained_backbone)
    model.to(device)

    train_dataset = HouseSegmentationDataset(args.data_dir, "train", image_size=args.image_size, augment=True)
    val_dataset = HouseSegmentationDataset(args.data_dir, "val", image_size=args.image_size, augment=False)
    test_dataset = HouseSegmentationDataset(args.data_dir, "test", image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history: list[dict[str, float]] = []
    best_val_iou = -1.0
    best_checkpoint = args.output_dir / "house_segmentation_best.pth"

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, None, device)

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "train_iou": train_metrics["iou"],
                "val_iou": val_metrics["iou"],
                "train_dice": train_metrics["dice"],
                "val_dice": val_metrics["dice"],
            }
        )

        if val_metrics["iou"] >= best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "val_iou": val_metrics["iou"],
                    "val_dice": val_metrics["dice"],
                },
                best_checkpoint,
            )

    with torch.no_grad():
        test_metrics = run_epoch(model, test_loader, None, device)

    (args.output_dir / "metrics.json").write_text(json.dumps({"test": test_metrics, "best_val_iou": best_val_iou}, indent=2))
    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    plot_history(history, args.output_dir / "training_curves.png")
    save_prediction_grid(model, test_dataset, device, args.output_dir / "sample_predictions.png")


if __name__ == "__main__":
    main()