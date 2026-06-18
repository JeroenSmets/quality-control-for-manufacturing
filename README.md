# Quality Control for Manufacturing

This repository contains a manufacturing quality-control pipeline split into four stages:

1. SAM-assisted labeling for inspected-object boxes
2. YOLO object detection for locating the inspected object
3. Classifier training for good/bad quality classification
4. Anomaly detection with PatchCore/Anomalib

All commands below are run from the repository root.

## Quick Start

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux:

```bash
source .venv/bin/activate
```

Install base dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-base.txt
```

Install PyTorch for your hardware:

```bash
# NVIDIA CUDA
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ROCm
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.8
```

Or use the installer scripts:

```powershell
.\install_windows.ps1
```

```bash
./install_linux.sh
```

## Repository Structure

```text
quality-control-for-manufacturing/
  README.md
  requirements-base.txt
  install_windows.ps1
  install_linux.sh
  raw_data/
    casting_512x512/
      ok_front/
      def_front/
  sam_labeling/
    sample_detection_label_images.py
    label_detection_images.py
    detector_labeling_pool/
    checkpoints/
  yolo_detection/
    prepare_detection_dataset.py
    validate_object_detection_dataset.py
    train_object_detector.py
    test_object_detector_on_unseen.py
    build_dataset_from_detector.py
    object_detection_dataset/
    checkpoints/
    runs/
  classifier/
    validate_prepared_dataset.py
    finetuning_classify.py
    dataset/
    dataset_rejects/
    reports/
    runs/
  anomaly_detection/
    finetuning_anomaly.py
    anomalib_dataset/
    runs/
  shared/
    path_utils.py
    project_config.py
```

Generated datasets, checkpoints, reports, and run outputs are ignored by Git.

## Important Labeling Note

Detector labels must only mark the inspected object:

```text
0 inspected_object
```

Do not label defects as separate detector classes. The good/bad quality label comes from the raw source folder path.

## Full Pipeline Commands

### 1. Sample images for detector labeling

```bash
python sam_labeling/sample_detection_label_images.py --good-count 50 --bad-count 50
```

Defaults:

```text
raw_data/casting_512x512/ok_front
raw_data/casting_512x512/def_front
sam_labeling/detector_labeling_pool/
```

### 2. Manually label object boxes

Without SAM:

```bash
python sam_labeling/label_detection_images.py --image-dir sam_labeling/detector_labeling_pool --labels-dir sam_labeling/detector_labeling_pool
```

With SAM:

```bash
python sam_labeling/label_detection_images.py --image-dir sam_labeling/detector_labeling_pool --labels-dir sam_labeling/detector_labeling_pool --sam-checkpoint sam_labeling/checkpoints/sam_vit_b_01ec64.pth --sam-model vit_b --device cuda
```

Controls:

- Left click adds foreground points on the inspected object.
- Right click adds background points.
- Press `g` to compute the mask and derive a bounding box.
- Press `s` to save the YOLO label file.
- Press `n` and `p` to move through images.

### 3. Prepare the YOLO detection dataset

```bash
python yolo_detection/prepare_detection_dataset.py --source-dir sam_labeling/detector_labeling_pool
```

Default output:

```text
yolo_detection/object_detection_dataset/
```

Overwrite an existing detection dataset:

```bash
python yolo_detection/prepare_detection_dataset.py --source-dir sam_labeling/detector_labeling_pool --overwrite
```

### 4. Validate the YOLO detection dataset

```bash
python yolo_detection/validate_object_detection_dataset.py
```

Default dataset:

```text
yolo_detection/object_detection_dataset/
```

### 5. Train the YOLO object detector

CPU default:

```bash
python yolo_detection/train_object_detector.py
```

GPU example:

```bash
python yolo_detection/train_object_detector.py --device 0 --epochs 100 --batch-size 4
```

Defaults:

```text
yolo_detection/object_detection_dataset/data.yaml
yolo_detection/checkpoints/yolo26n.pt
yolo_detection/runs/
```

If `yolo_detection/checkpoints/yolo26n.pt` is not present, the script falls back to the model name `yolo26n.pt`, matching the previous Ultralytics behavior.

### 6. Test the object detector on unseen raw images

```bash
python yolo_detection/test_object_detector_on_unseen.py --device 0 --count-per-class 10
```

Defaults:

```text
raw_data/casting_512x512/
sam_labeling/detector_labeling_pool/manifest.csv
yolo_detection/runs/unseen_object_detector_test/
```

Lower confidence example:

```bash
python yolo_detection/test_object_detector_on_unseen.py --device 0 --count-per-class 10 --conf 0.10 --output-dir yolo_detection/runs/unseen_object_detector_test_conf010
```

### 7. Build the detector-cropped classifier dataset

```bash
python yolo_detection/build_dataset_from_detector.py --weights yolo_detection/runs/detect/inspection_object_detector/weights/best.pt --device 0
```

Defaults:

```text
raw_data/casting_512x512/
classifier/dataset/
classifier/dataset_rejects/
classifier/dataset_manifest.csv
```

### 8. Validate the prepared classifier/anomaly dataset

```bash
python classifier/validate_prepared_dataset.py
```

Defaults:

```text
classifier/dataset/
classifier/reports/
```

### 9. Run classifier training

```bash
python classifier/finetuning_classify.py --device 0
```

Defaults:

```text
classifier/dataset/
classifier/runs/
```

Each run writes training metrics, plots, predictions, and a per-run `qc_classifier.pt` under `classifier/runs/`.

### 10. Run anomaly detector training

```bash
python anomaly_detection/finetuning_anomaly.py --device 0
```

Defaults:

```text
classifier/dataset/
anomaly_detection/anomalib_dataset/
anomaly_detection/runs/
```

## Folder Path Guide

- Raw source data lives under `raw_data/`.
- SAM labeling images and manifests live under `sam_labeling/`.
- YOLO datasets, checkpoints, and runs live under `yolo_detection/`.
- Classifier datasets, rejects, reports, and runs live under `classifier/`.
- Anomaly datasets and runs live under `anomaly_detection/`.
- Shared path/config helpers live under `shared/`.

## Troubleshooting

If `ultralytics` is missing:

```bash
python -m pip install ultralytics
```

If OpenCV is missing:

```bash
python -m pip install opencv-python
```

If SAM is missing:

```bash
python -m pip install segment-anything
```

If a script cannot find data, check that the command is being run from the repository root and that the input folder matches the new stage path.
