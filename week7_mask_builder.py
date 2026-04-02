from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from PIL import Image

from dataset_tools import bbox_to_mask, bboxes_to_mask, select_matching_sam_masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build building masks from the Week 7 satellite dataset using SAM agreement.")
    parser.add_argument("--dataset-name", default="keremberke/satellite-building-segmentation")
    parser.add_argument("--dataset-config", default="full")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--sam-model-type", default="vit_h")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap on how many samples to process.")
    parser.add_argument("--save-overlays", action="store_true")
    parser.add_argument("--points-per-side", type=int, default=32, help="SAM grid density. Lower is faster, higher is slower.")
    parser.add_argument("--crop-n-layers", type=int, default=0, help="Number of image crop layers SAM should use.")
    parser.add_argument("--min-mask-region-area", type=int, default=0, help="Filter tiny mask regions.")
    parser.add_argument("--fast-preset", action="store_true", help="Use a faster SAM config for CPU experiments.")
    return parser.parse_args()


def load_sam_mask_generator(
    checkpoint_path: Path,
    model_type: str,
    device: str,
    points_per_side: int,
    crop_n_layers: int,
    min_mask_region_area: int,
):
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.to(device=device)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        crop_n_layers=crop_n_layers,
        min_mask_region_area=min_mask_region_area,
    )


def load_week7_dataset(dataset_name: str, dataset_config: str):
    try:
        return load_dataset(dataset_name, name=dataset_config)
    except RuntimeError as exc:
        message = str(exc)
        if "Dataset scripts are no longer supported" in message:
            raise SystemExit(
                "Your current 'datasets' package is too new for this Week 7 dataset script.\n"
                "Fix it with:\n"
                "  ./venv/bin/pip install 'datasets<3'\n"
                "Then rerun this command."
            ) from exc
        raise


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.fast_preset:
        args.points_per_side = 16
        args.crop_n_layers = 0
        args.min_mask_region_area = 100

    dataset = load_week7_dataset(args.dataset_name, args.dataset_config)
    split = dataset[args.split]
    mask_generator = load_sam_mask_generator(
        args.sam_checkpoint,
        args.sam_model_type,
        args.device,
        args.points_per_side,
        args.crop_n_layers,
        args.min_mask_region_area,
    )

    image_dir = args.output_dir / "images"
    mask_dir = args.output_dir / "masks"
    overlay_dir = args.output_dir / "overlays"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    if args.save_overlays:
        overlay_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    sample_count = len(split) if args.max_samples <= 0 else min(len(split), args.max_samples)

    for index in range(sample_count):
        example = split[index]
        image = example["image"].convert("RGB")
        image_path = image_dir / f"{index:05d}.png"
        image.save(image_path)

        label_mask = bboxes_to_mask((image.width, image.height), example["objects"]["bbox"])
        sam_masks = mask_generator.generate(np.array(image))
        selected_mask, matches = select_matching_sam_masks(sam_masks, label_mask, iou_threshold=args.iou_threshold)

        mask_path = mask_dir / f"{index:05d}.png"
        Image.fromarray((selected_mask * 255).astype(np.uint8), mode="L").save(mask_path)

        entry = {
            "index": index,
            "image": image_path.name,
            "mask": mask_path.name,
            "label_boxes": len(example["objects"]["bbox"]),
            "sam_matches": len(matches),
            "best_iou": max((match["iou"] for match in matches), default=0.0),
        }
        manifest.append(entry)

        if args.save_overlays:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            axes[0].imshow(image)
            axes[0].set_title("Image")
            axes[0].axis("off")

            axes[1].imshow(label_mask, cmap="gray")
            axes[1].set_title("Label Mask")
            axes[1].axis("off")

            axes[2].imshow(selected_mask, cmap="gray")
            axes[2].set_title("SAM Agreement Mask")
            axes[2].axis("off")

            fig.tight_layout()
            fig.savefig(overlay_dir / f"{index:05d}.png", dpi=150)
            plt.close(fig)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()