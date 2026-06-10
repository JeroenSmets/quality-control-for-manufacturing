import sys
from argparse import ArgumentParser
from pathlib import Path

from project_config import (
    DEFAULT_DEVICE,
    DETECTOR_BATCH_SIZE,
    DETECTOR_EPOCHS,
    DETECTOR_IMG_SIZE,
    DETECTOR_PATIENCE,
    DETECTOR_WEIGHTS,
    NUM_WORKERS,
    OBJECT_DETECTION_DATA_YAML,
)


def import_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Install it with 'pip install ultralytics'."
        ) from exc

    return YOLO


def validate_data_yaml(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find object detection data.yaml: {path}\n"
            "Create the dataset first or update the path to your dataset YAML."
        )

    if not path.is_file():
        raise FileNotFoundError(f"data.yaml is not a file: {path}")


def parse_args():
    parser = ArgumentParser(
        description="Train an Ultralytics YOLO object detector on object_detection_dataset."
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=OBJECT_DETECTION_DATA_YAML,
        help="Path to the YOLO data.yaml file.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(DETECTOR_WEIGHTS),
        help="Pretrained YOLO weights to fine-tune from.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DETECTOR_IMG_SIZE,
        help="Training image size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DETECTOR_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DETECTOR_BATCH_SIZE,
        help="Training batch size.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DETECTOR_PATIENCE,
        help="Early stopping patience in epochs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Device to use for training, e.g. cpu, 0, or 0,1.",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("runs/detect"),
        help="Ultralytics project folder for training results.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="inspection_object_detector",
        help="Run name inside the project folder.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help="Number of data loader workers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        validate_data_yaml(args.data_yaml)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        YOLO = import_ultralytics()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Using pretrained weights: {args.weights}")
    print(f"Using dataset YAML: {args.data_yaml}")
    print(f"Training device: {args.device}")
    print(f"Image size: {args.imgsz}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Workers: {args.workers}")
    print(f"Project: {args.project}")
    print(f"Run name: {args.name}")

    model = YOLO(str(args.weights))

    print("Starting YOLO training...")
    results = model.train(
        data=str(args.data_yaml),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        patience=args.patience,
    )

    best_weights = args.project / args.name / "weights" / "best.pt"
    if not best_weights.exists():
        print(
            "WARNING: best.pt was not found after training. "
            "Check the Ultralytics output for the actual weights path."
        )
    else:
        print(f"Best model saved at: {best_weights}")

    print("Training complete. Running validation...")
    model.val(data=str(args.data_yaml), imgsz=args.imgsz, device=args.device, workers=args.workers)

    print("Validation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
