import csv
import random
import shutil
import sys
from argparse import ArgumentParser
from pathlib import Path

from project_config import IMAGE_EXTENSIONS, RANDOM_SEED, RAW_BAD_DIR, RAW_GOOD_DIR


DEFAULT_BEST_WEIGHTS = (
    Path("runs")
    / "detect"
    / "runs"
    / "detect"
    / "inspection_object_detector"
    / "weights"
    / "best.pt"
)


def import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with 'pip install ultralytics'."
        ) from exc

    return YOLO


def parse_args():
    parser = ArgumentParser(
        description="Run the trained object detector on raw images excluded from the labeling/training pool."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_BEST_WEIGHTS,
        help="Path to trained YOLO detector weights.",
    )
    parser.add_argument(
        "--bad-dir",
        type=Path,
        default=RAW_BAD_DIR,
        help="Raw defect image directory.",
    )
    parser.add_argument(
        "--good-dir",
        type=Path,
        default=RAW_GOOD_DIR,
        help="Raw ok image directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("detector_labeling_pool") / "manifest.csv",
        help="Manifest created by sample_detection_label_images.py; used to exclude seen images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs") / "detect" / "unseen_object_detector_test",
        help="Directory for annotated predictions, labels, and summary.csv.",
    )
    parser.add_argument(
        "--count-per-class",
        type=int,
        default=10,
        help="Number of unseen images to sample from each raw class.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Inference device, e.g. 0 for CUDA GPU 0 or cpu.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible unseen sampling.",
    )
    return parser.parse_args()


def validate_dir(path: Path, description: str):
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def image_paths(folder: Path):
    return [
        path
        for path in sorted(folder.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def seen_image_names(manifest_path: Path):
    if not manifest_path.exists():
        return set()

    names = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            original_path = row.get("original_path", "")
            sampled_path = row.get("sampled_path", "")
            if original_path:
                names.add(Path(original_path).name)
            if sampled_path:
                sampled_name = Path(sampled_path).name
                for prefix in ("good__", "bad__"):
                    if sampled_name.startswith(prefix):
                        sampled_name = sampled_name[len(prefix):]
                        break
                names.add(sampled_name)

    return names


def sample_unseen(paths, seen_names, count, seed):
    unseen = [path for path in paths if path.name not in seen_names]
    if count <= 0:
        return []
    if count > len(unseen):
        raise ValueError(
            f"Requested {count} unseen images, but only {len(unseen)} are available."
        )

    rng = random.Random(seed)
    return rng.sample(unseen, count)


def reset_output_dir(output_dir: Path):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def prepare_prediction_sources(output_dir: Path, selected_images, labels_by_path):
    source_dir = output_dir / "source_images"
    source_dir.mkdir(parents=True, exist_ok=True)

    metadata_by_path = {}
    metadata_by_name = {}

    for source_path in selected_images:
        target_path = source_dir / source_path.name
        shutil.copy2(source_path, target_path)

        metadata = {
            "raw_class": labels_by_path[source_path.resolve()],
            "original_path": source_path,
            "annotated_path": output_dir / source_path.name,
        }
        metadata_by_path[target_path.resolve()] = metadata
        metadata_by_name[target_path.name] = metadata

    return source_dir, metadata_by_path, metadata_by_name


def result_metadata(result, metadata_by_path, metadata_by_name):
    result_path = Path(result.path)
    metadata = metadata_by_path.get(result_path.resolve())
    if metadata is None:
        metadata = metadata_by_name.get(result_path.name)
    if metadata is None:
        raise KeyError(f"Could not match prediction result to source image: {result.path}")
    return metadata


def best_confidence(result):
    if result.boxes is None or len(result.boxes) == 0:
        return ""
    conf = result.boxes.conf
    if conf is None or len(conf) == 0:
        return ""
    return f"{float(conf.max().item()):.6f}"


def box_summary(result):
    if result.boxes is None or len(result.boxes) == 0:
        return ""

    boxes = []
    for xyxy, conf in zip(result.boxes.xyxy, result.boxes.conf):
        x1, y1, x2, y2 = [float(value) for value in xyxy.tolist()]
        boxes.append(
            f"{x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f} {float(conf.item()):.4f}"
        )

    return " | ".join(boxes)


def write_summary(output_dir: Path, results, metadata_by_path, metadata_by_name):
    summary_path = output_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "raw_class",
                "image_path",
                "annotated_image",
                "detections",
                "best_confidence",
                "boxes_xyxy_conf",
            ],
        )
        writer.writeheader()
        for result in results:
            metadata = result_metadata(result, metadata_by_path, metadata_by_name)
            detection_count = 0 if result.boxes is None else len(result.boxes)
            writer.writerow(
                {
                    "raw_class": metadata["raw_class"],
                    "image_path": str(metadata["original_path"]),
                    "annotated_image": str(metadata["annotated_path"]),
                    "detections": detection_count,
                    "best_confidence": best_confidence(result),
                    "boxes_xyxy_conf": box_summary(result),
                }
            )

    return summary_path


def print_summary(results, metadata_by_path, metadata_by_name):
    totals = {"bad": [0, 0], "good": [0, 0]}

    for result in results:
        metadata = result_metadata(result, metadata_by_path, metadata_by_name)
        source_path = metadata["original_path"]
        raw_class = metadata["raw_class"]
        detection_count = 0 if result.boxes is None else len(result.boxes)
        totals[raw_class][0] += 1
        totals[raw_class][1] += 1 if detection_count > 0 else 0
        print(
            f"{raw_class:4} | detections={detection_count} "
            f"| best_conf={best_confidence(result) or 'none':>8} "
            f"| {source_path.name}"
        )

    print("\nDetected at least one object:")
    for raw_class, (total, detected) in totals.items():
        print(f"- {raw_class}: {detected}/{total}")


def main():
    args = parse_args()

    if not args.weights.exists():
        raise FileNotFoundError(f"Detector weights not found: {args.weights}")

    validate_dir(args.bad_dir, "Bad raw image directory")
    validate_dir(args.good_dir, "Good raw image directory")

    bad_images = image_paths(args.bad_dir)
    good_images = image_paths(args.good_dir)
    seen_names = seen_image_names(args.manifest)

    sampled_bad = sample_unseen(
        bad_images, seen_names, args.count_per_class, args.seed
    )
    sampled_good = sample_unseen(
        good_images, seen_names, args.count_per_class, args.seed + 1
    )

    selected_images = sampled_bad + sampled_good
    labels_by_path = {
        **{path.resolve(): "bad" for path in sampled_bad},
        **{path.resolve(): "good" for path in sampled_good},
    }

    reset_output_dir(args.output_dir)
    prediction_source_dir, metadata_by_path, metadata_by_name = prepare_prediction_sources(
        args.output_dir, selected_images, labels_by_path
    )

    YOLO = import_ultralytics()
    model = YOLO(str(args.weights))

    print(f"Using weights: {args.weights}")
    print(f"Saving predictions to: {args.output_dir}")
    print(f"Unseen images: {len(selected_images)}")
    print(f"Device: {args.device}")
    print()

    results = model.predict(
        source=str(prediction_source_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.output_dir.parent.resolve()),
        name=args.output_dir.name,
        exist_ok=True,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    )

    summary_path = write_summary(
        args.output_dir, results, metadata_by_path, metadata_by_name
    )
    print_summary(results, metadata_by_path, metadata_by_name)
    print(f"\nSummary CSV: {summary_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
