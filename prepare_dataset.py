from __future__ import annotations

import argparse
from pathlib import Path

from dataset_tools import copy_image_mask_pairs, generate_masks_from_annotations, split_pairs, write_split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a house segmentation dataset from aerial imagery.")
    parser.add_argument("--images-dir", type=Path, required=True, help="Directory containing aerial images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where the train/val/test splits will be written.")
    parser.add_argument("--masks-dir", type=Path, help="Directory containing precomputed binary masks.")
    parser.add_argument("--annotations-file", type=Path, help="Polygon annotations used to generate pixel masks.")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.masks_dir is None and args.annotations_file is None:
        raise SystemExit("Provide either --masks-dir or --annotations-file.")

    working_masks_dir = args.masks_dir
    if working_masks_dir is None:
        working_masks_dir = args.output_dir / "generated_masks"
        generate_masks_from_annotations(args.images_dir, args.annotations_file, working_masks_dir)

    pairs = copy_image_mask_pairs(args.images_dir, working_masks_dir)
    if not pairs:
        raise SystemExit("No matched image/mask pairs were found.")

    split_dataset = split_pairs(pairs, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    write_split_dataset(split_dataset, args.output_dir)


if __name__ == "__main__":
    main()