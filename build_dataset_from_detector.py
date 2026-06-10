import csv
import random
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from project_config import (
    DETECTOR_WEIGHTS,
    IMAGE_EXTENSIONS,
    NUM_WORKERS,
    OUTPUT_DATASET_DIR,
    RANDOM_SEED,
    RAW_BAD_DIR,
    RAW_GOOD_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)


class DatasetBuilderError(Exception):
    pass


def import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DatasetBuilderError(
            "Ultralytics is not installed. Install with 'pip install ultralytics'."
        ) from exc

    return YOLO


def get_image_paths(folder: Path):
    if not folder.exists():
        raise DatasetBuilderError(f"Source folder does not exist: {folder}")

    image_paths = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_paths:
        raise DatasetBuilderError(f"No supported images found in: {folder}")

    return image_paths


def split_paths(image_paths, train_ratio, val_ratio, test_ratio, seed):
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise DatasetBuilderError("Train, val, and test ratios must sum to 1.0")

    paths = list(image_paths)
    random.seed(seed)
    random.shuffle(paths)

    n_total = len(paths)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    return {
        "train": paths[:n_train],
        "val": paths[n_train:n_train + n_val],
        "test": paths[n_train + n_val:],
    }


def build_dataset_dirs(dataset_root: Path):
    for split in ["train", "val", "test"]:
        for quality in ["good", "bad"]:
            (dataset_root / split / quality).mkdir(parents=True, exist_ok=True)


def build_reject_dirs(reject_root: Path):
    for reason in ["no_detection", "read_error"]:
        for quality in ["good", "bad"]:
            (reject_root / reason / quality).mkdir(parents=True, exist_ok=True)


def load_model(weights, device):
    YOLO = import_ultralytics()
    try:
        return YOLO(str(weights))
    except Exception as exc:
        raise DatasetBuilderError(
            f"Failed to load YOLO weights '{weights}': {exc}"
        ) from exc


def parse_result_boxes(result):
    if not hasattr(result, "boxes") or result.boxes is None:
        return []

    boxes = []
    if hasattr(result.boxes, "data"):
        try:
            data = result.boxes.data
            data = data.cpu().numpy() if hasattr(data, "cpu") else np.array(data)
        except Exception:
            data = np.array([])
    else:
        data = np.array([])

    if data.size == 0:
        return []

    for row in data:
        if len(row) < 6:
            continue

        x1, y1, x2, y2, conf, cls = row[:6]
        boxes.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "conf": float(conf),
            "class_id": int(cls),
            "area": float(max(0.0, x2 - x1) * max(0.0, y2 - y1)),
        })

    return boxes


def choose_best_box(boxes):
    if not boxes:
        return None

    return max(boxes, key=lambda box: (box["conf"], box["area"]))


def add_padding(box, pad_ratio, image_width, image_height):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    width = x2 - x1
    height = y2 - y1

    pad_x = width * pad_ratio
    pad_y = height * pad_ratio

    padded_x1 = max(0, x1 - pad_x)
    padded_y1 = max(0, y1 - pad_y)
    padded_x2 = min(image_width, x2 + pad_x)
    padded_y2 = min(image_height, y2 + pad_y)

    return {
        "x1": float(padded_x1),
        "y1": float(padded_y1),
        "x2": float(padded_x2),
        "y2": float(padded_y2),
    }


def crop_image(image, bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])]
    if x2 <= x1 or y2 <= y1:
        raise DatasetBuilderError(
            f"Invalid padded crop coordinates: {(x1, y1, x2, y2)}"
        )
    return image[y1:y2, x1:x2]


def save_manifest(manifest_path: Path, rows):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source_path",
                "quality_class",
                "split",
                "output_path",
                "confidence",
                "selected_bbox",
                "padded_bbox",
                "status",
                "message",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def format_bbox(bbox):
    return f"{bbox['x1']:.2f},{bbox['y1']:.2f},{bbox['x2']:.2f},{bbox['y2']:.2f}"


def process_images(
    model,
    image_paths,
    quality_class,
    split_name,
    output_dir,
    reject_root,
    pad_ratio,
    confidence,
    imgsz,
    device,
    allow_full_image_fallback,
):
    rows = []
    for source_path in image_paths:
        target_path = output_dir / split_name / quality_class / source_path.name
        status = "ok"
        message = ""
        selected_bbox = ""
        padded_bbox = ""
        confidence_value = ""

        image = cv2.imread(str(source_path))
        if image is None:
            status = "read_error"
            message = "Unreadable image file"
            reject_folder = reject_root / "read_error" / quality_class
            reject_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source_path), str(reject_folder / source_path.name))
            rows.append({
                "source_path": str(source_path.resolve()),
                "quality_class": quality_class,
                "split": split_name,
                "output_path": "",
                "confidence": "",
                "selected_bbox": "",
                "padded_bbox": "",
                "status": status,
                "message": message,
            })
            continue

        height, width = image.shape[:2]
        results = model.predict(
            source=str(source_path),
            imgsz=imgsz,
            device=device,
            classes=[0],
            conf=confidence,
            workers=NUM_WORKERS,
            verbose=False,
        )

        if not results or len(results) == 0:
            best_box = None
        else:
            best_box = choose_best_box(parse_result_boxes(results[0]))

        if best_box is None:
            if allow_full_image_fallback:
                selected_bbox = f"0.00,0.00,{width:.2f},{height:.2f}"
                padded_bbox = selected_bbox
                confidence_value = ""
                cv2.imwrite(str(target_path), image)
                status = "fallback_full_image"
                message = "No detection found; full image fallback used"
            else:
                status = "no_detection"
                message = "No inspected_object detection"
                reject_folder = reject_root / "no_detection" / quality_class
                reject_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source_path), str(reject_folder / source_path.name))
                rows.append({
                    "source_path": str(source_path.resolve()),
                    "quality_class": quality_class,
                    "split": split_name,
                    "output_path": "",
                    "confidence": "",
                    "selected_bbox": "",
                    "padded_bbox": "",
                    "status": status,
                    "message": message,
                })
                continue
        else:
            padded_box = add_padding(best_box, pad_ratio, width, height)
            crop = crop_image(image, padded_box)

            if crop.size == 0:
                status = "invalid_crop"
                message = "Cropped area is empty"
                reject_folder = reject_root / "read_error" / quality_class
                reject_folder.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(source_path), str(reject_folder / source_path.name))
                rows.append({
                    "source_path": str(source_path.resolve()),
                    "quality_class": quality_class,
                    "split": split_name,
                    "output_path": "",
                    "confidence": "",
                    "selected_bbox": "",
                    "padded_bbox": "",
                    "status": status,
                    "message": message,
                })
                continue

            cv2.imwrite(str(target_path), crop)
            selected_bbox = format_bbox(best_box)
            padded_bbox = format_bbox(padded_box)
            confidence_value = f"{best_box['conf']:.4f}"

        rows.append({
            "source_path": str(source_path.resolve()),
            "quality_class": quality_class,
            "split": split_name,
            "output_path": str(target_path.resolve()),
            "confidence": confidence_value,
            "selected_bbox": selected_bbox,
            "padded_bbox": padded_bbox,
            "status": status,
            "message": message,
        })

    return rows


def parse_args():
    parser = ArgumentParser(
        description="Build dataset crops from a trained YOLO detector."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(DETECTOR_WEIGHTS),
        help="Trained YOLO weights file or model name.",
    )
    parser.add_argument(
        "--raw-good-dir",
        type=Path,
        default=RAW_GOOD_DIR,
        help="Raw good image folder.",
    )
    parser.add_argument(
        "--raw-bad-dir",
        type=Path,
        default=RAW_BAD_DIR,
        help="Raw bad image folder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DATASET_DIR,
        help="Output dataset root folder.",
    )
    parser.add_argument(
        "--reject-root",
        type=Path,
        default=Path("dataset_rejects"),
        help="Folder to save rejected source images.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=TEST_RATIO,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding ratio around selected bounding box.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device, e.g. cpu or 0.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of data loader workers for inference.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest.csv"),
        help="CSV manifest path.",
    )
    parser.add_argument(
        "--allow-full-image-fallback",
        action="store_true",
        help="Allow fallback to full image crop when no detection is found.",
    )
    return parser.parse_args()


def main():
    global args
    args = parse_args()

    try:
        good_images = get_image_paths(args.raw_good_dir)
        bad_images = get_image_paths(args.raw_bad_dir)
        print(f"Found {len(good_images)} good images and {len(bad_images)} bad images.")

        build_dataset_dirs(args.output_dir)
        build_reject_dirs(args.reject_root)

        model = load_model(args.weights, args.device)

        good_splits = split_paths(
            good_images,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )
        bad_splits = split_paths(
            bad_images,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed,
        )

        manifest_rows = []
        for split_name, paths in good_splits.items():
            manifest_rows.extend(
                process_images(
                    model=model,
                    image_paths=paths,
                    quality_class="good",
                    split_name=split_name,
                    output_dir=args.output_dir,
                    reject_root=args.reject_root,
                    pad_ratio=args.padding,
                    confidence=args.confidence,
                    imgsz=args.imgsz,
                    device=args.device,
                    allow_full_image_fallback=args.allow_full_image_fallback,
                )
            )

        for split_name, paths in bad_splits.items():
            manifest_rows.extend(
                process_images(
                    model=model,
                    image_paths=paths,
                    quality_class="bad",
                    split_name=split_name,
                    output_dir=args.output_dir,
                    reject_root=args.reject_root,
                    pad_ratio=args.padding,
                    confidence=args.confidence,
                    imgsz=args.imgsz,
                    device=args.device,
                    allow_full_image_fallback=args.allow_full_image_fallback,
                )
            )

        save_manifest(args.manifest, manifest_rows)

        print(f"Wrote manifest with {len(manifest_rows)} entries to {args.manifest}")
        print("Dataset build complete.")
        return 0

    except DatasetBuilderError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
