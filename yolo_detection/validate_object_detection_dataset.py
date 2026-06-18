import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.path_utils import repo_relative_or_absolute
from shared.project_config import IMAGE_EXTENSIONS, OBJECT_DETECTION_DATASET_ROOT, OBJECT_DETECTION_DATA_YAML


class ValidationError(Exception):
    pass


def load_yaml(path: Path):
    try:
        import yaml
    except ImportError as exc:
        raise ValidationError(
            "PyYAML is required to validate object detection dataset. "
            "Install it with 'pip install pyyaml'."
        ) from exc

    if not path.exists():
        raise ValidationError(f"Missing data.yaml: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValidationError(f"data.yaml must contain a mapping at the top level: {path}")

    return data


def validate_data_yaml(data, data_yaml_path: Path, dataset_root: Path):
    required = ["path", "train", "val", "names"]
    for key in required:
        if key not in data:
            raise ValidationError(f"data.yaml is missing required key '{key}' in {data_yaml_path}")

    data_path_value = str(data["path"])
    data_path = Path(data_path_value)
    if data_path.is_absolute():
        path_matches = data_path.resolve() == dataset_root.resolve()
    else:
        expected_values = {
            dataset_root.name,
            str(dataset_root),
            dataset_root.as_posix(),
            repo_relative_or_absolute(dataset_root),
        }
        normalized_expected = {value.replace("\\", "/") for value in expected_values}
        path_matches = data_path_value.replace("\\", "/") in normalized_expected

    if not path_matches:
        raise ValidationError(
            f"data.yaml path must point to '{repo_relative_or_absolute(dataset_root)}', got '{data['path']}'"
        )

    if not isinstance(data["names"], dict) or not data["names"]:
        raise ValidationError("data.yaml 'names' must be a non-empty mapping of class ids to names")

    class_ids = set()
    for key in data["names"]:
        try:
            class_ids.add(int(key))
        except (TypeError, ValueError):
            raise ValidationError(
                f"data.yaml names keys must be integers, invalid key: {key}"
            )

    if not class_ids:
        raise ValidationError("data.yaml contains no class IDs in 'names'")

    return {"train": data["train"], "val": data["val"], "names": data["names"], "class_ids": class_ids}


def get_samples(folder: Path):
    if not folder.exists():
        raise ValidationError(f"Missing folder: {folder}")

    image_files = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if not image_files:
        raise ValidationError(f"No supported image files found in: {folder}")

    return image_files


def validate_label_file(label_path: Path, class_ids):
    if not label_path.exists():
        raise ValidationError(f"Missing label file for image: {label_path.with_suffix('.jpg')} or {label_path.with_suffix('.png')} -> {label_path}")

    if not label_path.is_file():
        raise ValidationError(f"Label path is not a file: {label_path}")

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return 0

    line_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            raise ValidationError(
                f"Invalid label format in {label_path} line {line_number}: "
                f"expected 5 values, got {len(parts)}"
            )

        class_id_text, x_text, y_text, w_text, h_text = parts
        try:
            class_id = int(class_id_text)
        except ValueError:
            raise ValidationError(
                f"Invalid class id in {label_path} line {line_number}: {class_id_text}"
            )

        if class_id not in class_ids:
            raise ValidationError(
                f"Unknown class id {class_id} in {label_path} line {line_number}. "
                f"Valid ids: {sorted(class_ids)}"
            )

        try:
            x = float(x_text)
            y = float(y_text)
            w = float(w_text)
            h = float(h_text)
        except ValueError:
            raise ValidationError(
                f"Invalid coordinate value in {label_path} line {line_number}: {raw_line}"
            )

        for coord_name, coord_value in ("x_center", x), ("y_center", y), ("width", w), ("height", h):
            if coord_value < 0.0 or coord_value > 1.0:
                raise ValidationError(
                    f"Coordinate '{coord_name}' out of range [0,1] in {label_path} line {line_number}: {coord_value}"
                )

        if w <= 0.0 or h <= 0.0:
            raise ValidationError(
                f"Width and height must be positive in {label_path} line {line_number}: width={w}, height={h}"
            )

        line_count += 1

    return line_count


def validate_split(dataset_root: Path, split_dir: str, names: dict, class_ids: set):
    if split_dir.startswith("images/") or split_dir.startswith("images\\"):
        images_dir = dataset_root / split_dir
        labels_dir = dataset_root / "labels" / Path(split_dir).name
    else:
        images_dir = dataset_root / "images" / split_dir
        labels_dir = dataset_root / "labels" / split_dir

    if not images_dir.exists():
        raise ValidationError(f"Missing images folder: {images_dir}")
    if not labels_dir.exists():
        raise ValidationError(f"Missing labels folder: {labels_dir}")

    image_files = get_samples(images_dir)
    total_labels = 0

    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        total_labels += validate_label_file(label_path, class_ids)

    return len(image_files), total_labels


def main():
    parser = ArgumentParser(
        description="Validate an Ultralytics YOLO object detection dataset structure."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=OBJECT_DETECTION_DATASET_ROOT,
        help="Root folder of the object detection dataset.",
    )
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=OBJECT_DETECTION_DATA_YAML,
        help="Path to the dataset data.yaml file.",
    )
    args = parser.parse_args()

    try:
        data = load_yaml(args.data_yaml)
        config = validate_data_yaml(data, args.data_yaml, args.dataset_root)

        print("data.yaml validation passed.")
        print(f"Detected class names: {config['names']}")

        train_count, train_labels = validate_split(
            args.dataset_root,
            config["train"],
            config["names"],
            config["class_ids"],
        )
        print(f"Train split: {train_count} images, {train_labels} total label lines.")

        val_count, val_labels = validate_split(
            args.dataset_root,
            config["val"],
            config["names"],
            config["class_ids"],
        )
        print(f"Val split: {val_count} images, {val_labels} total label lines.")

        total_images = train_count + val_count
        total_label_files = train_count + val_count

        print("\nValidation summary:")
        print(f"  dataset root: {args.dataset_root}")
        print(f"  data.yaml: {args.data_yaml}")
        print(f"  splits checked: train, val")
        print(f"  total images: {total_images}")
        print(f"  total label files required: {total_label_files}")
        print("Validation completed successfully.")
        return 0

    except ValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
