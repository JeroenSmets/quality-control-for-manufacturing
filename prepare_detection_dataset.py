import random
import shutil
from argparse import ArgumentParser
from pathlib import Path

from project_config import (
    IMAGE_EXTENSIONS,
    OBJECT_DETECTION_DATASET_ROOT,
    OBJECT_DETECTION_DATA_YAML,
    RANDOM_SEED,
)


def find_images(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Source directory does not exist: {folder}")

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No supported image files found in: {folder}")
    return images


def ensure_label_file(image_path: Path):
    label_path = image_path.with_suffix(".txt")
    if not label_path.exists():
        label_path.write_text("", encoding="utf-8")
    return label_path


def split_images(images, train_ratio, val_ratio, seed):
    total = train_ratio + val_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_ratio and val_ratio must sum to 1.0")

    random.seed(seed)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:],
    }


def build_folders(root: Path):
    for subfolder in [
        root / "images" / "train",
        root / "images" / "val",
        root / "labels" / "train",
        root / "labels" / "val",
    ]:
        subfolder.mkdir(parents=True, exist_ok=True)


def copy_files(image_paths, split_name, source_dir: Path, target_root: Path, move_files: bool):
    target_image_dir = target_root / "images" / split_name
    target_label_dir = target_root / "labels" / split_name

    rows = []
    for image_path in image_paths:
        label_path = ensure_label_file(image_path)

        target_image_path = target_image_dir / image_path.name
        target_label_path = target_label_dir / label_path.name

        if move_files:
            shutil.move(str(image_path), str(target_image_path))
            shutil.move(str(label_path), str(target_label_path))
        else:
            shutil.copy2(str(image_path), str(target_image_path))
            shutil.copy2(str(label_path), str(target_label_path))

        rows.append((target_image_path, target_label_path))

    return rows


def write_data_yaml(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "path: object_detection_dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: inspected_object\n",
        encoding="utf-8",
    )


def parse_args():
    parser = ArgumentParser(
        description="Prepare YOLO detection dataset from labeled sample images."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("detector_labeling_pool"),
        help="Directory containing labeled sample images and .txt labels.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=OBJECT_DETECTION_DATASET_ROOT,
        help="Root of the object detection dataset to create.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of labeled images to use for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of labeled images to use for validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files into the dataset instead of copying them.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing object_detection_dataset before preparing a new one.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_root.exists():
        if args.overwrite:
            shutil.rmtree(args.dataset_root)
        else:
            raise RuntimeError(
                f"Dataset root already exists: {args.dataset_root}. Use --overwrite to recreate it."
            )

    images = find_images(args.source_dir)
    print(f"Found {len(images)} sample images in {args.source_dir}")

    build_folders(args.dataset_root)
    splits = split_images(images, args.train_ratio, args.val_ratio, args.seed)

    total_copied = 0
    for split_name, paths in splits.items():
        copied = copy_files(paths, split_name, args.source_dir, args.dataset_root, args.move)
        print(f"{split_name}: {len(copied)} images")
        total_copied += len(copied)

    write_data_yaml(args.dataset_root / "data.yaml")

    print(f"Prepared object detection dataset with {total_copied} images.")
    print(f"Dataset root: {args.dataset_root}")
    print(f"YAML configuration written to: {args.dataset_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
