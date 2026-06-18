from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
import csv
import json
import shutil
import sys
import torch
from sklearn.metrics import confusion_matrix

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.project_config import (
    ANOMALIB_DATASET_ROOT,
    ANOMALY_RUNS_DIR,
    DATASET_ROOT,
    NUM_WORKERS,
)


# =========================
# Configuration
# =========================

RUNS_DIR = ANOMALY_RUNS_DIR

IMAGE_SIZE = (224, 224)

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

DEFAULT_DEVICE = "cpu"

# PatchCore settings
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.1
NUM_NEIGHBORS = 9


# =========================
# Arguments
# =========================

def parse_args():
    parser = ArgumentParser(
        description="Train a PatchCore anomaly detector on detector-cropped images."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATASET_ROOT,
        help="Root of the prepared classifier dataset.",
    )
    parser.add_argument(
        "--anomalib-root",
        type=Path,
        default=ANOMALIB_DATASET_ROOT,
        help="Folder where the Anomalib-compatible dataset is created.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Training device: cpu, cuda, cuda:0, or 0.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory where timestamped anomaly run results are saved.",
    )
    return parser.parse_args()


def get_accelerator(device_arg: str):
    normalized = device_arg.strip().lower()

    if normalized == "cpu":
        return "cpu", 1, None

    if normalized == "cuda":
        normalized = "cuda:0"

    if normalized.isdigit():
        normalized = f"cuda:{normalized}"

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

        parts = normalized.split(":", maxsplit=1)
        device_index = int(parts[1]) if len(parts) == 2 and parts[1] else 0
        return "gpu", [device_index], device_index

    raise ValueError(f"Unsupported device: {device_arg}")


# =========================
# Utility functions
# =========================

def create_run_dir(runs_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{timestamp}_patchcore"
    suffix = 1

    while run_dir.exists():
        run_dir = runs_dir / f"{timestamp}_patchcore_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    return value


def write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(make_json_safe(data), handle, indent=2)


def flatten_dict(data, prefix=""):
    rows = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            rows.update(flatten_dict(value, name))
        else:
            rows[name] = make_json_safe(value)
    return rows


def save_dataset_manifest(run_dir: Path, manifest_rows):
    manifest_path = run_dir / "anomalib_dataset_manifest.csv"
    fieldnames = [
        "source_path",
        "source_split",
        "quality_class",
        "target_path",
        "target_split",
        "anomalib_role",
    ]

    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return manifest_path


def save_test_results(run_dir: Path, test_results):
    write_json(run_dir / "test_results.json", test_results)
    (run_dir / "test_results.txt").write_text(str(test_results), encoding="utf-8")

    result_rows = test_results if isinstance(test_results, list) else [test_results]
    csv_rows = []
    for row in result_rows:
        if isinstance(row, dict):
            csv_rows.append(flatten_dict(row))
        else:
            csv_rows.append({"value": make_json_safe(row)})

    if not csv_rows:
        csv_rows.append({"value": ""})

    csv_path = run_dir / "test_results.csv"
    fieldnames = sorted({key for row in csv_rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    return csv_path


def tensor_to_list(value):
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return [value.item()]
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def clean_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def get_anomaly_threshold(model):
    post_processor = getattr(model, "post_processor", None)
    if post_processor is None:
        return None

    threshold = getattr(post_processor, "normalized_image_threshold", None)
    if threshold is None:
        threshold = getattr(post_processor, "image_threshold", None)

    threshold = clean_scalar(threshold)
    if threshold is None:
        return None

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        return None

    if threshold != threshold:
        return None

    return threshold


def build_manifest_lookup(manifest_rows):
    lookup = {}
    for row in manifest_rows:
        lookup[str(Path(row["target_path"]).resolve())] = row
    return lookup


def collect_prediction_rows(prediction_batches, manifest_rows, threshold):
    manifest_lookup = build_manifest_lookup(manifest_rows)
    rows = []

    def label_to_quality(label):
        if label == 0:
            return "good"
        if label == 1:
            return "bad"
        return "unknown"

    for batch in prediction_batches or []:
        if isinstance(batch, list):
            rows.extend(collect_prediction_rows(batch, manifest_rows, threshold))
            continue

        image_paths = tensor_to_list(getattr(batch, "image_path", None))
        gt_labels = tensor_to_list(getattr(batch, "gt_label", None))
        pred_labels = tensor_to_list(getattr(batch, "pred_label", None))
        pred_scores = tensor_to_list(getattr(batch, "pred_score", None))

        for index, image_path in enumerate(image_paths):
            image_path = str(image_path)
            gt_label = int(gt_labels[index]) if index < len(gt_labels) else -1
            pred_label = int(pred_labels[index]) if index < len(pred_labels) else -1
            pred_score = float(pred_scores[index]) if index < len(pred_scores) else float("nan")

            manifest_row = manifest_lookup.get(str(Path(image_path).resolve()), {})
            true_quality = label_to_quality(gt_label)
            predicted_quality = label_to_quality(pred_label)

            rows.append(
                {
                    "image_path": image_path,
                    "source_path": manifest_row.get("source_path", ""),
                    "true_label": gt_label,
                    "true_quality": true_quality,
                    "predicted_label": pred_label,
                    "predicted_quality": predicted_quality,
                    "anomaly_score": pred_score,
                    "threshold": threshold if threshold is not None else "",
                    "correct": gt_label == pred_label,
                }
            )

    return rows


def save_anomaly_predictions(run_dir: Path, prediction_rows):
    csv_path = run_dir / "anomaly_predictions.csv"
    json_path = run_dir / "anomaly_predictions.json"

    fieldnames = [
        "image_path",
        "source_path",
        "true_label",
        "true_quality",
        "predicted_label",
        "predicted_quality",
        "anomaly_score",
        "threshold",
        "correct",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_rows)

    write_json(json_path, prediction_rows)
    return csv_path, json_path


def save_anomaly_confusion_matrix(run_dir: Path, prediction_rows):
    labels = [0, 1]
    class_names = ["good", "bad"]
    y_true = [row["true_label"] for row in prediction_rows]
    y_pred = [row["predicted_label"] for row in prediction_rows]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    csv_path = run_dir / "anomaly_confusion_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["true_label", "pred_good", "pred_bad"])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name] + [int(value) for value in row])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping anomaly_confusion_matrix.png")
        return matrix, csv_path, None

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Anomaly Confusion Matrix")

    threshold_value = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            color = "white" if value > threshold_value else "black"
            ax.text(col_index, row_index, value, ha="center", va="center", color=color)

    fig.tight_layout()
    plot_path = run_dir / "anomaly_confusion_matrix.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return matrix, csv_path, plot_path


def save_anomaly_score_plots(run_dir: Path, prediction_rows, threshold):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping anomaly score plots")
        return []

    good_scores = [row["anomaly_score"] for row in prediction_rows if row["true_label"] == 0]
    bad_scores = [row["anomaly_score"] for row in prediction_rows if row["true_label"] == 1]

    paths = []

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = 30
    ax.hist(good_scores, bins=bins, alpha=0.65, label="true good", color="#2f7ed8")
    ax.hist(bad_scores, bins=bins, alpha=0.65, label="true bad", color="#d94f45")
    if threshold is not None:
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold {threshold:.3f}")
    ax.set_title("Anomaly Score Distribution")
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Image count")
    ax.legend()
    fig.tight_layout()
    distribution_path = run_dir / "anomaly_score_distribution.png"
    fig.savefig(distribution_path, dpi=160)
    plt.close(fig)
    paths.append(distribution_path)

    ranked_rows = sorted(prediction_rows, key=lambda row: row["anomaly_score"])
    fig, ax = plt.subplots(figsize=(11, 5))
    for index, row in enumerate(ranked_rows):
        if not row["correct"]:
            color = "#f2b701"
            marker = "x"
            size = 70
        elif row["true_label"] == 1:
            color = "#d94f45"
            marker = "o"
            size = 24
        else:
            color = "#2f7ed8"
            marker = "o"
            size = 24

        ax.scatter(index, row["anomaly_score"], c=color, marker=marker, s=size)

    if threshold is not None:
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.5, label=f"threshold {threshold:.3f}")

    wrong_count = sum(1 for row in prediction_rows if not row["correct"])
    ax.set_title(f"Ranked Anomaly Scores ({wrong_count} wrong predictions highlighted)")
    ax.set_xlabel("Images sorted by anomaly score")
    ax.set_ylabel("Anomaly score")
    ax.scatter([], [], c="#2f7ed8", marker="o", label="true good correct")
    ax.scatter([], [], c="#d94f45", marker="o", label="true bad correct")
    ax.scatter([], [], c="#f2b701", marker="x", s=70, label="wrong prediction")
    ax.legend(loc="best")
    fig.tight_layout()
    ranking_path = run_dir / "anomaly_score_ranking.png"
    fig.savefig(ranking_path, dpi=160)
    plt.close(fig)
    paths.append(ranking_path)

    return paths


def copy_wrong_prediction_images(run_dir: Path, prediction_rows):
    wrong_dir = run_dir / "wrong_predictions"
    wrong_rows = [row for row in prediction_rows if not row["correct"]]

    if not wrong_rows:
        wrong_dir.mkdir(parents=True, exist_ok=True)
        (wrong_dir / "README.txt").write_text("No wrong predictions found.\n", encoding="utf-8")
        return wrong_dir

    wrong_dir.mkdir(parents=True, exist_ok=True)
    for row in wrong_rows:
        image_path = Path(row["image_path"])
        source_path = Path(row["source_path"]) if row["source_path"] else image_path
        source = source_path if source_path.exists() else image_path
        target_name = (
            f"true-{row['true_quality']}__pred-{row['predicted_quality']}__"
            f"score-{row['anomaly_score']:.4f}__{source.name}"
        )
        shutil.copy2(source, wrong_dir / target_name)

    return wrong_dir


def save_prediction_visuals(run_dir: Path, prediction_rows, model):
    threshold = get_anomaly_threshold(model)
    save_anomaly_predictions(run_dir, prediction_rows)
    matrix, _, _ = save_anomaly_confusion_matrix(run_dir, prediction_rows)
    save_anomaly_score_plots(run_dir, prediction_rows, threshold)
    wrong_dir = copy_wrong_prediction_images(run_dir, prediction_rows)

    wrong_count = sum(1 for row in prediction_rows if not row["correct"])
    summary = {
        "total_images": len(prediction_rows),
        "wrong_predictions": wrong_count,
        "correct_predictions": len(prediction_rows) - wrong_count,
        "threshold": threshold,
        "confusion_matrix": matrix.tolist(),
        "wrong_predictions_dir": wrong_dir,
    }
    write_json(run_dir / "prediction_summary.json", summary)
    return summary


def check_folder_exists(folder: Path, name: str):
    if not folder.exists():
        raise FileNotFoundError(f"{name} folder not found: {folder}")

    image_files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    ]

    if len(image_files) == 0:
        raise RuntimeError(f"{name} folder contains no images: {folder}")

    print(f"{name}: {len(image_files)} images")


def copy_images(src_dir: Path, dst_dir: Path, source_split: str, target_split: str, quality_class: str, anomalib_role: str):
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    count = 0
    manifest_rows = []

    for src_path in sorted(src_dir.iterdir()):
        if not src_path.is_file():
            continue

        if src_path.suffix.lower() not in image_extensions:
            continue

        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        count += 1
        manifest_rows.append(
            {
                "source_path": str(src_path.resolve()),
                "source_split": source_split,
                "quality_class": quality_class,
                "target_path": str(dst_path.resolve()),
                "target_split": target_split,
                "anomalib_role": anomalib_role,
            }
        )

    return count, manifest_rows


def prepare_anomalib_folder_dataset(dataset_root: Path, anomalib_dataset_root: Path):
    """
    Convert the existing classification dataset into an Anomalib-friendly layout.

    Source:
        dataset/train/good
        dataset/val/good
        dataset/val/bad
        dataset/test/good
        dataset/test/bad

    Target:
        anomalib_dataset/train/good
        anomalib_dataset/test/good
        anomalib_dataset/test/bad

    PatchCore trains only on normal images. For evaluation, we combine the
    existing val and test splits into one test set:
        good = val/good + test/good
        bad  = val/bad  + test/bad
    """

    print("\nChecking source dataset...")

    train_good_dir = dataset_root / "train" / "good"
    val_good_dir = dataset_root / "val" / "good"
    val_bad_dir = dataset_root / "val" / "bad"
    test_good_dir = dataset_root / "test" / "good"
    test_bad_dir = dataset_root / "test" / "bad"

    check_folder_exists(train_good_dir, "Train good")
    check_folder_exists(val_good_dir, "Validation good")
    check_folder_exists(val_bad_dir, "Validation bad")
    check_folder_exists(test_good_dir, "Test good")
    check_folder_exists(test_bad_dir, "Test bad")

    if anomalib_dataset_root.exists():
        print(f"\nRemoving existing folder: {anomalib_dataset_root}")
        shutil.rmtree(anomalib_dataset_root)

    train_good_target = anomalib_dataset_root / "train" / "good"
    test_good_target = anomalib_dataset_root / "test" / "good"
    test_bad_target = anomalib_dataset_root / "test" / "bad"

    print("\nPreparing Anomalib folder dataset...")

    manifest_rows = []

    train_good_count, rows = copy_images(
        train_good_dir,
        train_good_target,
        source_split="train",
        target_split="train",
        quality_class="good",
        anomalib_role="normal_train",
    )
    manifest_rows.extend(rows)

    val_good_count, rows = copy_images(
        val_good_dir,
        test_good_target,
        source_split="val",
        target_split="test",
        quality_class="good",
        anomalib_role="normal_test",
    )
    manifest_rows.extend(rows)
    test_good_count, rows = copy_images(
        test_good_dir,
        test_good_target,
        source_split="test",
        target_split="test",
        quality_class="good",
        anomalib_role="normal_test",
    )
    manifest_rows.extend(rows)

    val_bad_count, rows = copy_images(
        val_bad_dir,
        test_bad_target,
        source_split="val",
        target_split="test",
        quality_class="bad",
        anomalib_role="abnormal_test",
    )
    manifest_rows.extend(rows)
    test_bad_count, rows = copy_images(
        test_bad_dir,
        test_bad_target,
        source_split="test",
        target_split="test",
        quality_class="bad",
        anomalib_role="abnormal_test",
    )
    manifest_rows.extend(rows)

    print("\nCreated Anomalib dataset:")
    print(f"  {train_good_target}: {train_good_count} images")
    print(f"  {test_good_target}:  {val_good_count + test_good_count} images")
    print(f"  {test_bad_target}:   {val_bad_count + test_bad_count} images")

    dataset_summary = {
        "anomalib_dataset_root": str(anomalib_dataset_root),
        "train_good": train_good_count,
        "test_good_from_val": val_good_count,
        "test_good_from_test": test_good_count,
        "test_good_total": val_good_count + test_good_count,
        "test_bad_from_val": val_bad_count,
        "test_bad_from_test": test_bad_count,
        "test_bad_total": val_bad_count + test_bad_count,
    }

    return anomalib_dataset_root, dataset_summary, manifest_rows


# =========================
# Main
# =========================

def main():
    args = parse_args()
    run_dir = create_run_dir(args.runs_dir)
    anomalib_engine_dir = run_dir / "anomalib_engine"

    print("Run results:", run_dir)

    anomalib_root, dataset_summary, manifest_rows = prepare_anomalib_folder_dataset(
        args.dataset_root,
        args.anomalib_root,
    )
    save_dataset_manifest(run_dir, manifest_rows)
    write_json(run_dir / "dataset_summary.json", dataset_summary)

    accelerator, devices, cuda_index = get_accelerator(args.device)

    print("\nTraining configuration:")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices:     {devices}")
    if cuda_index is not None:
        print(f"  GPU:         {torch.cuda.get_device_name(cuda_index)}")
    print(f"  Backbone:    {BACKBONE}")
    print(f"  Layers:      {LAYERS}")
    print(f"  Image size:  {IMAGE_SIZE}")

    write_json(
        run_dir / "config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device_arg": args.device,
            "accelerator": accelerator,
            "devices": devices,
            "gpu": torch.cuda.get_device_name(cuda_index) if cuda_index is not None else None,
            "dataset_root": args.dataset_root,
            "anomalib_dataset_root": anomalib_root,
            "anomalib_engine_dir": anomalib_engine_dir,
            "image_size": IMAGE_SIZE,
            "train_batch_size": TRAIN_BATCH_SIZE,
            "eval_batch_size": EVAL_BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "model_name": "Patchcore",
            "backbone": BACKBONE,
            "layers": LAYERS,
            "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
            "num_neighbors": NUM_NEIGHBORS,
            "dataset_summary": dataset_summary,
            "run_dir": run_dir,
        },
    )

    # PatchCore preprocessing.
    # The current Anomalib PatchCore docs show this helper for setting image size.
    pre_processor = Patchcore.configure_pre_processor(
        image_size=IMAGE_SIZE,
    )

    # Folder expects:
    #   normal_dir       -> normal training images
    #   abnormal_dir     -> anomalous test images
    #   normal_test_dir  -> normal test images
    #
    # In your case:
    #   train/good -> normal training
    #   test/bad   -> anomalous test
    #   test/good  -> normal test
    #
    # The latest Anomalib docs show Folder supports normal_dir, abnormal_dir,
    # and normal_test_dir for custom folder datasets.
    datamodule = Folder(
        name="casting_quality_anomaly",
        root=anomalib_root,
        normal_dir="train/good",
        abnormal_dir="test/bad",
        normal_test_dir="test/good",
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        pre_trained=True,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=NUM_NEIGHBORS,
        pre_processor=pre_processor,
    )

    engine = Engine(
        accelerator=accelerator,
        devices=devices,
        default_root_dir=anomalib_engine_dir,
    )

    print("\nFitting PatchCore model...")
    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print("\nTesting PatchCore model on normal + anomalous test images...")
    test_results = engine.test(
        model=model,
        datamodule=datamodule,
    )

    print("\nTest results:")
    print(test_results)
    save_test_results(run_dir, test_results)

    print("\nSaving per-image anomaly predictions and plots...")
    prediction_batches = engine.predict(
        model=model,
        dataloaders=datamodule.test_dataloader(),
        return_predictions=True,
    )
    prediction_rows = collect_prediction_rows(
        prediction_batches=prediction_batches,
        manifest_rows=manifest_rows,
        threshold=get_anomaly_threshold(model),
    )
    prediction_summary = save_prediction_visuals(run_dir, prediction_rows, model)
    print(
        "Wrong predictions: "
        f"{prediction_summary['wrong_predictions']}/{prediction_summary['total_images']}"
    )

    write_json(
        run_dir / "metrics.json",
        {
            "test_results": test_results,
            "prediction_summary": prediction_summary,
            "dataset_summary": dataset_summary,
            "anomalib_engine_dir": anomalib_engine_dir,
        },
    )

    print("\nSaving trained PatchCore model checkpoint...")
    checkpoint_path = run_dir / "patchcore_metadata.ckpt"

    # Anomalib/Lightning stores checkpoints automatically under results,
    # but this also saves a simple explicit checkpoint for reference.
    torch.save(
        {
            "model_name": "Patchcore",
            "backbone": BACKBONE,
            "layers": LAYERS,
            "coreset_sampling_ratio": CORESET_SAMPLING_RATIO,
            "num_neighbors": NUM_NEIGHBORS,
            "image_size": IMAGE_SIZE,
        },
        checkpoint_path,
    )

    print(f"Saved metadata checkpoint to: {checkpoint_path}")
    print(f"Saved run results to: {run_dir}")
    print(f"Anomalib engine outputs are under: {anomalib_engine_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
