from __future__ import annotations

import base64
import io
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from flask import Request
from torchvision import transforms
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_segmentation_model(num_classes: int = 2, use_pretrained_backbone: bool = False) -> torch.nn.Module:
    backbone_weights = ResNet50_Weights.DEFAULT if use_pretrained_backbone else None
    return deeplabv3_resnet50(weights=None, weights_backbone=backbone_weights, num_classes=num_classes)


def _prepare_image(image: Image.Image, image_size: int) -> torch.Tensor:
    pipeline = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return pipeline(image.convert("RGB"))


def _maybe_load_state_dict(checkpoint_path: str, device: torch.device) -> dict[str, Any] | None:
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return None

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    return None


class SegmentationPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        image_size: int = 256,
        device: str = "cpu",
        use_pretrained_backbone: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.image_size = image_size
        self.device = torch.device(device)
        self.model = build_segmentation_model(use_pretrained_backbone=use_pretrained_backbone)
        state_dict = _maybe_load_state_dict(checkpoint_path, self.device)
        if state_dict is not None:
            self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> np.ndarray:
        original_size = image.size
        tensor = _prepare_image(image, self.image_size).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)["out"]
            predicted = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy().astype(np.uint8)

        mask = Image.fromarray(predicted * 255).resize(original_size, resample=Image.NEAREST)
        return (np.array(mask) > 0).astype(np.uint8)


def decode_image_payload(request: Request) -> Image.Image | None:
    if request.files.get("image") is not None:
        return Image.open(request.files["image"].stream).convert("RGB")

    payload = request.get_json(silent=True) or {}
    image_base64 = payload.get("image_base64")
    if image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image_path = payload.get("image_path")
    if image_path:
        return Image.open(image_path).convert("RGB")

    return None


def load_mask_image(image: Image.Image) -> np.ndarray:
    return (np.array(image.convert("L")) > 0).astype(np.uint8)


def compute_iou_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    union = np.logical_or(pred, true).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(pred, true).sum()
    return float(intersection / union)


def compute_dice_score(pred_mask: np.ndarray, true_mask: np.ndarray) -> float:
    pred = pred_mask.astype(bool)
    true = true_mask.astype(bool)
    denominator = pred.sum() + true.sum()
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(pred, true).sum()
    return float((2.0 * intersection) / denominator)


def encode_mask_png(mask: np.ndarray) -> str:
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")