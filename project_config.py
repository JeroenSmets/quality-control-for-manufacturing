from pathlib import Path

# Raw image source folders.
RAW_BAD_DIR = Path(r"casting_512x512\def_front")
RAW_GOOD_DIR = Path(r"casting_512x512\ok_front")

# Output dataset folder consumed by classifier and anomaly scripts.
OUTPUT_DATASET_DIR = Path("dataset")
DATASET_ROOT = OUTPUT_DATASET_DIR

# Train/validation/test split ratios.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Supported image file extensions.
IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
}

# Deterministic random seed for reproducible splits.
RANDOM_SEED = 42

# Windows-safe defaults.
DEFAULT_DEVICE = "cpu"
NUM_WORKERS = 0

# Object detector defaults.
DETECTOR_WEIGHTS = "yolo26n.pt"
DETECTOR_FALLBACK_WEIGHTS = ["yolo11n.pt", "yolov8n.pt"]
DETECTOR_IMG_SIZE = 640
DETECTOR_BATCH_SIZE = 4
DETECTOR_EPOCHS = 100
DETECTOR_PATIENCE = 50

OBJECT_DETECTION_DATASET_ROOT = Path("object_detection_dataset")
OBJECT_DETECTION_DATA_YAML = OBJECT_DETECTION_DATASET_ROOT / "data.yaml"
OBJECT_DETECTION_NAMES = {
    0: "inspected_object",
}
