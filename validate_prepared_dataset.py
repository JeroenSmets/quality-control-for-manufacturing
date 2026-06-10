import csv
import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

from project_config import DATASET_ROOT, IMAGE_EXTENSIONS, NUM_WORKERS, OUTPUT_DATASET_DIR


class DatasetValidationError(Exception):
    pass


def get_image_paths(folder: Path):
    if not folder.exists():
        raise DatasetValidationError(f"Missing folder: {folder}")

    paths = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not paths:
        raise DatasetValidationError(f"No supported images found in: {folder}")

    return paths


def validate_structure(dataset_root: Path):
    splits = ["train", "val", "test"]
    qualities = ["good", "bad"]
    missing = []
    stats = {}

    for split in splits:
        for quality in qualities:
            folder = dataset_root / split / quality
            if not folder.exists():
                missing.append(str(folder))
                continue

            paths = get_image_paths(folder)
            stats[f"{split}/{quality}"] = len(paths)

    if missing:
        raise DatasetValidationError(
            "Dataset is missing required folders:\n" + "\n".join(missing)
        )

    return stats


def validate_images(dataset_root: Path):
    errors = []
    valid_paths = []

    for split in ["train", "val", "test"]:
        for quality in ["good", "bad"]:
            folder = dataset_root / split / quality
            image_paths = get_image_paths(folder)

            for image_path in image_paths:
                image = None
                try:
                    import cv2
                    image = cv2.imread(str(image_path))
                except Exception as exc:
                    errors.append(
                        {
                            "path": str(image_path),
                            "error": f"OpenCV read failed: {exc}",
                        }
                    )
                    continue

                if image is None:
                    errors.append(
                        {
                            "path": str(image_path),
                            "error": "Unreadable image or unsupported format",
                        }
                    )
                    continue

                if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                    errors.append(
                        {
                            "path": str(image_path),
                            "error": "Image has zero pixels",
                        }
                    )
                    continue

                valid_paths.append(str(image_path.resolve()))

    return valid_paths, errors


def load_manifest(manifest_path: Path):
    if not manifest_path.exists():
        raise DatasetValidationError(f"Manifest file not found: {manifest_path}")

    rows = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_fields = {
            "source_path",
            "quality_class",
            "split",
            "output_path",
            "status",
        }
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise DatasetValidationError(
                f"Manifest is missing required fields: {sorted(missing)}"
            )

        for row in reader:
            rows.append(row)

    if not rows:
        raise DatasetValidationError(f"Manifest is empty: {manifest_path}")

    return rows


def validate_manifest_paths(manifest_rows, dataset_root: Path):
    errors = []
    output_paths = set()
    source_to_splits = {}

    for row in manifest_rows:
        output_path = Path(row["output_path"])
        source_path = Path(row["source_path"])
        split = row["split"]
        status = row.get("status", "").strip().lower()

        if status != "ok" and status != "fallback_full_image":
            continue

        if not output_path.exists():
            errors.append(
                {
                    "path": str(output_path),
                    "error": "Manifest output_path does not exist",
                }
            )

        output_paths.add(str(output_path.resolve()))

        source_resolved = str(source_path.resolve())
        source_to_splits.setdefault(source_resolved, set()).add(split)

    duplicate_sources = {
        source: splits
        for source, splits in source_to_splits.items()
        if len(splits) > 1
    }

    return errors, duplicate_sources, output_paths


def create_preview(valid_paths, report_dir: Path, preview_name: str, max_images: int = 16):
    try:
        import cv2
    except ImportError:
        return None

    if not valid_paths:
        return None

    sample_paths = random.sample(valid_paths, min(max_images, len(valid_paths)))
    thumbnails = []

    for image_path in sample_paths:
        image = cv2.imread(str(image_path))
        if image is None or image.size == 0:
            continue

        height, width = image.shape[:2]
        scale = 256 / max(height, width)
        thumb = cv2.resize(image, (int(width * scale), int(height * scale)))
        thumbnails.append(thumb)

    if not thumbnails:
        return None

    cols = min(4, len(thumbnails))
    rows = []
    for row_start in range(0, len(thumbnails), cols):
        row_images = thumbnails[row_start:row_start + cols]
        widths = [img.shape[1] for img in row_images]
        max_height = max(img.shape[0] for img in row_images)
        normalized_row = [
            cv2.copyMakeBorder(img, 0, max_height - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            for img in row_images
        ]
        rows.append(cv2.hconcat(normalized_row))

    grid = cv2.vconcat(rows)
    preview_path = report_dir / preview_name
    report_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), grid)
    return preview_path


def write_report(report_dir: Path, report_data: dict):
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "dataset_validation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)
    return report_path


def parse_args():
    parser = ArgumentParser(
        description="Validate the prepared classification/anomaly dataset structure and write a report."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root of the prepared dataset folder.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_manifest.csv"),
        help="Path to the dataset manifest CSV.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("reports"),
        help="Directory where JSON report and preview image are written.",
    )
    parser.add_argument(
        "--preview-name",
        type=str,
        default="dataset_validation_preview.jpg",
        help="Filename for the optional preview grid image.",
    )
    parser.add_argument(
        "--preview-count",
        type=int,
        default=16,
        help="Number of random crops to include in the preview grid.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        stats = validate_structure(args.dataset_root)
        valid_paths, image_errors = validate_images(args.dataset_root)
        manifest_rows = load_manifest(args.manifest)
        manifest_errors, duplicate_sources, output_paths = validate_manifest_paths(
            manifest_rows, args.dataset_root
        )

        missing_outputs = []
        for output_path in output_paths:
            if not Path(output_path).exists():
                missing_outputs.append(output_path)

        report_data = {
            "dataset_root": str(args.dataset_root.resolve()),
            "manifest_path": str(args.manifest.resolve()),
            "stats": stats,
            "image_errors": image_errors,
            "manifest_errors": manifest_errors,
            "duplicate_source_files": {
                source: sorted(list(splits))
                for source, splits in duplicate_sources.items()
            },
            "missing_manifest_outputs": missing_outputs,
            "preview_image": None,
        }

        preview_path = create_preview(valid_paths, args.reports_dir, args.preview_name, args.preview_count)
        if preview_path is not None:
            report_data["preview_image"] = str(preview_path.resolve())

        report_path = write_report(args.reports_dir, report_data)

        print(f"Validation completed. Report written to: {report_path}")
        if report_data["preview_image"]:
            print(f"Preview image written to: {report_data['preview_image']}")

        if image_errors or manifest_errors or duplicate_sources or missing_outputs:
            print("Validation failed with issues. Check the report for details.")
            return 1

        print("Dataset validation passed with no issues.")
        return 0

    except DatasetValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
