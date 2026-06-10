import csv
import random
from argparse import ArgumentParser
from pathlib import Path

from project_config import (
    IMAGE_EXTENSIONS,
    RANDOM_SEED,
    RAW_BAD_DIR,
    RAW_GOOD_DIR,
)


def get_image_paths(folder: Path):
    if not folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {folder}")

    image_paths = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {folder}")

    return image_paths


def make_unique_target_path(target_dir: Path, prefix: str, source_path: Path):
    target_name = f"{prefix}__{source_path.name}"
    target_path = target_dir / target_name
    suffix_index = 1

    while target_path.exists():
        target_name = f"{prefix}__{source_path.stem}_{suffix_index}{source_path.suffix}"
        target_path = target_dir / target_name
        suffix_index += 1

    return target_path


def sample_images(image_paths, count, seed):
    if count <= 0:
        return []

    if count > len(image_paths):
        raise ValueError(
            f"Requested {count} images, but only {len(image_paths)} available. "
            "Reduce the sample count or add more images."
        )

    random.seed(seed)
    return random.sample(image_paths, count)


def write_manifest(manifest_path: Path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["original_path", "sampled_path", "quality_class"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = ArgumentParser(
        description=(
            "Sample raw good/bad images for object detector labeling and "
            "create a labeled pool with a manifest."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("detector_labeling_pool"),
        help="Output folder for sampled label images.",
    )
    parser.add_argument(
        "--good-count",
        type=int,
        default=10,
        help="Number of good images to sample.",
    )
    parser.add_argument(
        "--bad-count",
        type=int,
        default=10,
        help="Number of bad images to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic sampling.",
    )
    args = parser.parse_args()

    good_images = get_image_paths(RAW_GOOD_DIR)
    bad_images = get_image_paths(RAW_BAD_DIR)

    print(f"Found {len(good_images)} good images in: {RAW_GOOD_DIR}")
    print(f"Found {len(bad_images)} bad images in: {RAW_BAD_DIR}")

    sampled_good = sample_images(good_images, args.good_count, args.seed)
    sampled_bad = sample_images(bad_images, args.bad_count, args.seed)

    target_dir = args.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    for source_path in sampled_good:
        target_path = make_unique_target_path(target_dir, "good", source_path)
        target_path.write_bytes(source_path.read_bytes())
        manifest_rows.append(
            {
                "original_path": str(source_path.resolve()),
                "sampled_path": str(target_path.resolve()),
                "quality_class": "good",
            }
        )

    for source_path in sampled_bad:
        target_path = make_unique_target_path(target_dir, "bad", source_path)
        target_path.write_bytes(source_path.read_bytes())
        manifest_rows.append(
            {
                "original_path": str(source_path.resolve()),
                "sampled_path": str(target_path.resolve()),
                "quality_class": "bad",
            }
        )

    manifest_path = target_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)

    print(f"Wrote {len(manifest_rows)} sampled images to: {target_dir}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
