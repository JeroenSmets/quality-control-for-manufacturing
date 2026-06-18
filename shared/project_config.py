from .path_utils import repo_path

# Raw image source folders.
RAW_DATA_ROOT = repo_path("raw_data", "casting_512x512")
RAW_BAD_DIR = RAW_DATA_ROOT / "def_front"
RAW_GOOD_DIR = RAW_DATA_ROOT / "ok_front"

# SAM-assisted labeling defaults.
SAM_LABELING_ROOT = repo_path("sam_labeling")
DETECTOR_LABELING_POOL_DIR = SAM_LABELING_ROOT / "detector_labeling_pool"
SAM_CHECKPOINTS_DIR = SAM_LABELING_ROOT / "checkpoints"
SAM_VIT_B_CHECKPOINT = SAM_CHECKPOINTS_DIR / "sam_vit_b_01ec64.pth"

# YOLO object detection defaults.
YOLO_DETECTION_ROOT = repo_path("yolo_detection")
YOLO_CHECKPOINTS_DIR = YOLO_DETECTION_ROOT / "checkpoints"
YOLO_RUNS_DIR = YOLO_DETECTION_ROOT / "runs"
OBJECT_DETECTION_DATASET_ROOT = YOLO_DETECTION_ROOT / "object_detection_dataset"
OBJECT_DETECTION_DATA_YAML = OBJECT_DETECTION_DATASET_ROOT / "data.yaml"
DETECTOR_BEST_WEIGHTS = (
    YOLO_RUNS_DIR
    / "detect"
    / "inspection_object_detector"
    / "weights"
    / "best.pt"
)
UNSEEN_DETECTOR_TEST_DIR = YOLO_RUNS_DIR / "unseen_object_detector_test"

# Output dataset folder consumed by classifier and anomaly scripts.
CLASSIFIER_ROOT = repo_path("classifier")
OUTPUT_DATASET_DIR = CLASSIFIER_ROOT / "dataset"
DATASET_ROOT = OUTPUT_DATASET_DIR
DATASET_REJECTS_DIR = CLASSIFIER_ROOT / "dataset_rejects"
DATASET_MANIFEST = CLASSIFIER_ROOT / "dataset_manifest.csv"
CLASSIFIER_REPORTS_DIR = CLASSIFIER_ROOT / "reports"
CLASSIFIER_RUNS_DIR = CLASSIFIER_ROOT / "runs"
CLASSIFIER_MODEL_PATH = CLASSIFIER_RUNS_DIR / "qc_classifier.pt"

# Anomaly detection defaults.
ANOMALY_DETECTION_ROOT = repo_path("anomaly_detection")
ANOMALIB_DATASET_ROOT = ANOMALY_DETECTION_ROOT / "anomalib_dataset"
ANOMALY_RUNS_DIR = ANOMALY_DETECTION_ROOT / "runs"

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
DETECTOR_MODEL_NAME = "yolo26n.pt"
DETECTOR_WEIGHTS = YOLO_CHECKPOINTS_DIR / DETECTOR_MODEL_NAME
DETECTOR_FALLBACK_WEIGHTS = ["yolo11n.pt", "yolov8n.pt"]
DETECTOR_IMG_SIZE = 640
DETECTOR_BATCH_SIZE = 4
DETECTOR_EPOCHS = 100
DETECTOR_PATIENCE = 50

OBJECT_DETECTION_NAMES = {
    0: "inspected_object",
}
