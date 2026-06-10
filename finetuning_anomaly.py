from pathlib import Path
import shutil
import torch

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore

from project_config import DATASET_ROOT, NUM_WORKERS


# =========================
# Configuration
# =========================

TRAIN_GOOD_DIR = DATASET_ROOT / "train" / "good"
VAL_GOOD_DIR = DATASET_ROOT / "val" / "good"
VAL_BAD_DIR = DATASET_ROOT / "val" / "bad"
TEST_GOOD_DIR = DATASET_ROOT / "test" / "good"
TEST_BAD_DIR = DATASET_ROOT / "test" / "bad"

ANOMALIB_DATASET_ROOT = Path("anomalib_dataset")
RESULTS_DIR = Path("anomalib_results")

IMAGE_SIZE = (224, 224)

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

# Recommended for your current AMD Radeon 740M setup, because your earlier
# PyTorch run crashed in ROCm/rocBLAS.
FORCE_CPU = True

# PatchCore settings
BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.1
NUM_NEIGHBORS = 9


# =========================
# Utility functions
# =========================

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


def copy_images(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    count = 0

    for src_path in sorted(src_dir.iterdir()):
        if not src_path.is_file():
            continue

        if src_path.suffix.lower() not in image_extensions:
            continue

        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        count += 1

    return count


def prepare_anomalib_folder_dataset():
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

    check_folder_exists(TRAIN_GOOD_DIR, "Train good")
    check_folder_exists(VAL_GOOD_DIR, "Validation good")
    check_folder_exists(VAL_BAD_DIR, "Validation bad")
    check_folder_exists(TEST_GOOD_DIR, "Test good")
    check_folder_exists(TEST_BAD_DIR, "Test bad")

    if ANOMALIB_DATASET_ROOT.exists():
        print(f"\nRemoving existing folder: {ANOMALIB_DATASET_ROOT}")
        shutil.rmtree(ANOMALIB_DATASET_ROOT)

    train_good_target = ANOMALIB_DATASET_ROOT / "train" / "good"
    test_good_target = ANOMALIB_DATASET_ROOT / "test" / "good"
    test_bad_target = ANOMALIB_DATASET_ROOT / "test" / "bad"

    print("\nPreparing Anomalib folder dataset...")

    train_good_count = copy_images(TRAIN_GOOD_DIR, train_good_target)

    val_good_count = copy_images(VAL_GOOD_DIR, test_good_target)
    test_good_count = copy_images(TEST_GOOD_DIR, test_good_target)

    val_bad_count = copy_images(VAL_BAD_DIR, test_bad_target)
    test_bad_count = copy_images(TEST_BAD_DIR, test_bad_target)

    print("\nCreated Anomalib dataset:")
    print(f"  {train_good_target}: {train_good_count} images")
    print(f"  {test_good_target}:  {val_good_count + test_good_count} images")
    print(f"  {test_bad_target}:   {val_bad_count + test_bad_count} images")

    return ANOMALIB_DATASET_ROOT


def get_accelerator():
    if FORCE_CPU:
        return "cpu", 1

    if torch.cuda.is_available():
        return "gpu", 1

    return "cpu", 1


# =========================
# Main
# =========================

def main():
    anomalib_root = prepare_anomalib_folder_dataset()

    accelerator, devices = get_accelerator()

    print("\nTraining configuration:")
    print(f"  Accelerator: {accelerator}")
    print(f"  Devices:     {devices}")
    print(f"  Backbone:    {BACKBONE}")
    print(f"  Layers:      {LAYERS}")
    print(f"  Image size:  {IMAGE_SIZE}")

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
        default_root_dir=RESULTS_DIR,
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

    print("\nSaving trained PatchCore model checkpoint...")
    checkpoint_path = RESULTS_DIR / "patchcore_trained.ckpt"

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

    print("\nDone.")


if __name__ == "__main__":
    main()