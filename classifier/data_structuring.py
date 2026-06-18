import cv2
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.project_config import (
    IMAGE_EXTENSIONS,
    OUTPUT_DATASET_DIR,
    RANDOM_SEED,
    RAW_BAD_DIR,
    RAW_GOOD_DIR,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
)


# =========================
# Helper functions
# =========================

def get_image_paths(folder: Path):
    """Return all supported image files in a folder."""
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    image_paths = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_paths = sorted(image_paths)

    if not image_paths:
        raise RuntimeError(f"No images found in: {folder}")

    return image_paths


def create_dataset_folders(output_dir: Path):
    """Create dataset/train|val|test/good|bad folder structure."""
    for split in ["train", "val", "test"]:
        for category in ["good", "bad"]:
            folder = output_dir / split / category
            folder.mkdir(parents=True, exist_ok=True)


def select_roi_from_first_image(image_path: Path):
    """Open first image and let the user select one ROI."""
    image = cv2.imread(str(image_path))

    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    roi = cv2.selectROI(
        "Select ROI, then press ENTER or SPACE",
        image,
        fromCenter=False,
        showCrosshair=True,
    )

    cv2.destroyAllWindows()

    x, y, w, h = map(int, roi)

    if w == 0 or h == 0:
        raise RuntimeError("No ROI selected.")

    print(f"Selected ROI: x={x}, y={y}, w={w}, h={h}")

    return x, y, w, h


def split_images(image_paths, train_ratio, val_ratio, test_ratio, random_seed):
    """Randomly split image paths into train, val, and test lists."""
    total = train_ratio + val_ratio + test_ratio

    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    image_paths = list(image_paths)

    random.seed(random_seed)
    random.shuffle(image_paths)

    n_total = len(image_paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_paths = image_paths[:n_train]
    val_paths = image_paths[n_train:n_train + n_val]
    test_paths = image_paths[n_train + n_val:]

    return {
        "train": train_paths,
        "val": val_paths,
        "test": test_paths,
    }


def crop_and_save_images(split_dict, category, roi, output_dir):
    """Crop images using the selected ROI and save into dataset structure."""
    x, y, w, h = roi

    for split_name, image_paths in split_dict.items():
        target_folder = output_dir / split_name / category
        target_folder.mkdir(parents=True, exist_ok=True)

        for image_path in image_paths:
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            img_h, img_w = image.shape[:2]

            if x + w > img_w or y + h > img_h:
                print(f"Skipping {image_path.name}: ROI outside image bounds")
                continue

            crop = image[y:y + h, x:x + w]

            output_path = target_folder / image_path.name

            success = cv2.imwrite(str(output_path), crop)

            if not success:
                print(f"Failed to save: {output_path}")


def save_roi(output_dir, roi):
    """Save ROI coordinates for reproducibility."""
    x, y, w, h = roi

    roi_file = output_dir / "roi.txt"
    with open(roi_file, "w", encoding="utf-8") as f:
        f.write(f"x={x}\n")
        f.write(f"y={y}\n")
        f.write(f"w={w}\n")
        f.write(f"h={h}\n")

    print(f"Saved ROI to: {roi_file}")


def clear_existing_dataset(output_dir: Path):
    """Optional: remove old dataset folder before creating a new one."""
    if output_dir.exists():
        answer = input(
            f"Output folder '{output_dir}' already exists. "
            "Delete and recreate it? [y/N]: "
        ).strip().lower()

        if answer == "y":
            shutil.rmtree(output_dir)
            print(f"Deleted existing folder: {output_dir}")
        else:
            print("Keeping existing folder. New files may be added or overwritten.")


# =========================
# Main script
# =========================

def main():
    clear_existing_dataset(OUTPUT_DATASET_DIR)
    create_dataset_folders(OUTPUT_DATASET_DIR)

    bad_images = get_image_paths(RAW_BAD_DIR)
    good_images = get_image_paths(RAW_GOOD_DIR)

    print(f"Found {len(bad_images)} bad images in: {RAW_BAD_DIR}")
    print(f"Found {len(good_images)} good images in: {RAW_GOOD_DIR}")

    # Select ROI once using the first available image.
    # You can change this to good_images[0] if preferred.
    first_image_for_roi = bad_images[0]
    print(f"Selecting ROI from first image: {first_image_for_roi}")

    roi = select_roi_from_first_image(first_image_for_roi)
    save_roi(OUTPUT_DATASET_DIR, roi)

    bad_split = split_images(
        bad_images,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED,
    )

    good_split = split_images(
        good_images,
        TRAIN_RATIO,
        VAL_RATIO,
        TEST_RATIO,
        RANDOM_SEED,
    )

    crop_and_save_images(
        split_dict=bad_split,
        category="bad",
        roi=roi,
        output_dir=OUTPUT_DATASET_DIR,
    )

    crop_and_save_images(
        split_dict=good_split,
        category="good",
        roi=roi,
        output_dir=OUTPUT_DATASET_DIR,
    )

    print("\nDataset created successfully.")
    print_dataset_summary(OUTPUT_DATASET_DIR)


def print_dataset_summary(output_dir):
    print("\nDataset summary:")

    for split in ["train", "val", "test"]:
        for category in ["good", "bad"]:
            folder = output_dir / split / category
            count = len([
                p for p in folder.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            ])
            print(f"{split}/{category}: {count} images")


if __name__ == "__main__":
    main()
