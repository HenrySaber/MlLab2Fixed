from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as tv_functional


def build_binary_mask(image_size: tuple[int, int], polygons: list[list[list[float]]]) -> Image.Image:
    mask = Image.new("L", image_size, 0)
    drawer = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) >= 3:
            drawer.polygon([tuple(point) for point in polygon], outline=1, fill=1)
    return mask


def bbox_to_mask(image_size: tuple[int, int], bbox: Sequence[float]) -> np.ndarray:
    image_width, image_height = image_size
    x_min, y_min, bbox_width, bbox_height = map(int, bbox)

    x_start = max(0, x_min)
    y_start = max(0, y_min)
    x_end = min(image_width, x_start + max(0, bbox_width))
    y_end = min(image_height, y_start + max(0, bbox_height))

    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    if x_end > x_start and y_end > y_start:
        mask[y_start:y_end, x_start:x_end] = 1
    return mask


def bboxes_to_mask(image_size: tuple[int, int], bboxes: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
    for bbox in bboxes:
        mask = np.maximum(mask, bbox_to_mask(image_size, bbox))
    return mask


def binary_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a_bool = mask_a.astype(bool)
    mask_b_bool = mask_b.astype(bool)
    union = np.logical_or(mask_a_bool, mask_b_bool).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    return float(intersection / union)


def binary_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    mask_a_bool = mask_a.astype(bool)
    mask_b_bool = mask_b.astype(bool)
    denominator = mask_a_bool.sum() + mask_b_bool.sum()
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(mask_a_bool, mask_b_bool).sum()
    return float((2.0 * intersection) / denominator)


def select_matching_sam_masks(
    sam_masks: list[dict],
    label_mask: np.ndarray,
    iou_threshold: float = 0.3,
) -> tuple[np.ndarray, list[dict]]:
    selected_mask = np.zeros_like(label_mask, dtype=np.uint8)
    matches: list[dict] = []

    for sam_mask in sam_masks:
        segmentation = sam_mask["segmentation"].astype(np.uint8)
        iou = binary_iou(segmentation, label_mask)
        if iou >= iou_threshold:
            selected_mask = np.maximum(selected_mask, segmentation)
            matches.append({
                "bbox": sam_mask.get("bbox"),
                "area": sam_mask.get("area"),
                "iou": iou,
            })

    if not matches:
        selected_mask = label_mask.astype(np.uint8)

    return selected_mask.astype(np.uint8), matches


def _group_geojson_features(annotation_path: Path) -> dict[str, list[list[list[float]]]]:
    raw = json.loads(annotation_path.read_text())
    grouped: dict[str, list[list[list[float]]]] = {}

    if isinstance(raw, dict) and "features" in raw:
        for feature in raw["features"]:
            properties = feature.get("properties", {})
            image_name = properties.get("filename") or properties.get("image_id") or properties.get("image")
            if not image_name:
                continue
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            if geometry.get("type") == "Polygon":
                polygons = [coordinates[0]] if coordinates else []
            elif geometry.get("type") == "MultiPolygon":
                polygons = [polygon[0] for polygon in coordinates if polygon]
            else:
                polygons = []
            grouped.setdefault(image_name, []).extend(polygons)
        return grouped

    if isinstance(raw, dict):
        for image_name, polygons in raw.items():
            grouped[image_name] = polygons
    return grouped


def generate_masks_from_annotations(images_dir: Path, annotations_path: Path, masks_dir: Path) -> None:
    annotations = _group_geojson_features(annotations_path)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue

        image = Image.open(image_path).convert("RGB")
        polygons = annotations.get(image_path.name) or annotations.get(image_path.stem)
        if not polygons:
            continue

        mask = build_binary_mask(image.size, polygons)
        mask.save(masks_dir / f"{image_path.stem}.png")


def copy_image_mask_pairs(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    paired: list[tuple[Path, Path]] = []

    for image_path in sorted(images_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue

        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.exists():
            continue

        paired.append((image_path, mask_path))

    return paired


def split_pairs(pairs: list[tuple[Path, Path]], train_ratio: float, val_ratio: float, test_ratio: float, seed: int) -> dict[str, list[tuple[Path, Path]]]:
    if round(train_ratio + val_ratio + test_ratio, 5) != 1.0:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    shuffled = list(pairs)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_split_dataset(pairs_by_split: dict[str, list[tuple[Path, Path]]], output_dir: Path) -> None:
    for split_name, pairs in pairs_by_split.items():
        image_target = output_dir / split_name / "images"
        mask_target = output_dir / split_name / "masks"
        image_target.mkdir(parents=True, exist_ok=True)
        mask_target.mkdir(parents=True, exist_ok=True)

        for image_path, mask_path in pairs:
            shutil.copy2(image_path, image_target / image_path.name)
            shutil.copy2(mask_path, mask_target / mask_path.name)


class HouseSegmentationDataset(Dataset):
    def __init__(self, root_dir: Path, split: str, image_size: int = 256, augment: bool = False) -> None:
        self.image_paths = sorted((root_dir / split / "images").glob("*"))
        self.mask_paths = [root_dir / split / "masks" / f"{path.stem}.png" for path in self.image_paths]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        mask = Image.open(self.mask_paths[index]).convert("L")

        if self.augment and random.random() < 0.5:
            image = tv_functional.hflip(image)
            mask = tv_functional.hflip(mask)

        image = transforms.Resize((self.image_size, self.image_size))(image)
        mask = transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.NEAREST)(mask)

        image_tensor = transforms.ToTensor()(image)
        mask_tensor = torch.from_numpy((np.array(mask) > 0).astype(np.int64))

        return image_tensor, mask_tensor