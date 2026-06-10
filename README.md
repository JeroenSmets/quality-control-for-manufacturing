# Quality Control for Manufacturing

This repository contains a manufacturing quality-control pipeline for classifier, anomaly, and object-detection training. The object-detection stage locates the inspected part in raw images, crops it, and generates YOLO-format data for the existing classification/anomaly pipeline.

## Quick Start

1. Create and activate a Python virtual environment.
2. Install base dependencies with `python -m pip install -r requirements-base.txt`.
3. Install a PyTorch wheel for your hardware:
   - NVIDIA CUDA: `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`
   - CPU-only: `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
   - ROCm: `python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.8`
4. On Windows, run `.\install_windows.ps1` to install dependencies and choose the correct PyTorch target.
5. On Linux, run `./install_linux.sh`.

## Repository Setup

This repository is ready to initialize and push to GitHub. The included `.gitignore` excludes local dataset folders, generated outputs, and large model weights so that the published repo stays small and portable.

Example commands:

```powershell
git init
git add .
git commit -m "Initial quality control pipeline"
git branch -M main
git remote add origin https://github.com/<your-username>/quality-control-for-manufacturing.git
git push -u origin main
```

## Pipeline Overview

1. Sample raw images for detector labeling
2. Manually label object bounding boxes in the sampled images
3. Create a YOLO-format detection dataset and validate it
4. Train an Ultralytics YOLO object detector
5. Build a detector-cropped `dataset` for classifier and anomaly training
6. Validate the prepared classifier/anomaly dataset
7. Run classifier training
8. Run anomaly detector training

## Expected Folder Structure

### Raw source folders

```text
casting_512x512/
  ok_front/
  def_front/
```

### Detection dataset

```text
object_detection_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

### Classifier / Anomaly dataset output

```text
dataset/
  train/
    good/
    bad/
  val/
    good/
    bad/
  test/
    good/
    bad/
```

### Rejects and reports

```text
dataset_rejects/
  no_detection/
    good/
    bad/
  read_error/
    good/
    bad/
reports/
  dataset_validation_report.json
  dataset_validation_preview.jpg
```

## Important Labeling Note

Detector labels must only mark the inspected object:

```text
0 inspected_object
```

Do not label defects as separate detector classes. The `good` / `bad` quality label is derived from the raw source folder path, not from object detection labels.

## Commands

### 1. Sample images for detector labeling

```bash
python sample_detection_label_images.py
```

This copies a small representative sample from `casting_512x512/ok_front` and `casting_512x512/def_front` into `detector_labeling_pool/` and creates `detector_labeling_pool/manifest.csv`.

### 2. Manually label object boxes

Use the built-in interactive point-based labeler to generate a mask and bounding box automatically:

```bash
python label_detection_images.py --image-dir detector_labeling_pool --labels-dir detector_labeling_pool
```

If you have a Segment Anything checkpoint, enable point-guided SAM segmentation with:

```bash
python label_detection_images.py --image-dir detector_labeling_pool --labels-dir detector_labeling_pool --sam-checkpoint C:\Python\Projects\PytorchTest\sam_vit_b_01ec64.pth --sam-model vit_b
```

- Left click to add foreground points on the inspected object.
- Right click to add background points on the surrounding area.
- Press `g` to compute the mask and derive a tight bounding box.
- Press `s` to save the generated YOLO label file.
- Label only the object to inspect, not the defect.
- After labeling, run the preparation helper to split data and generate `data.yaml`:

```bash
python prepare_detection_dataset.py --source-dir detector_labeling_pool
```

This creates the expected layout under `object_detection_dataset/`:

```text
object_detection_dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  data.yaml
```

If you want to overwrite an existing dataset root, add `--overwrite`.

### 3. Validate the YOLO detection dataset

```bash
python validate_object_detection_dataset.py
```

This checks:
- `object_detection_dataset` folder layout
- `data.yaml` contents
- every image has a matching `.txt` label file or an empty label file
- normalized coordinates are between 0 and 1
- class IDs exist in `names`

### 4. Train the object detector

```bash
python train_object_detector.py
```

Defaults:
- `device=cpu`
- `workers=0`
- pretrained weights: `yolo26n.pt`

If you have a GPU, you can override:

```bash
python train_object_detector.py --device 0 --epochs 100 --batch-size 4
```

### 5. Build the detector-cropped dataset

```bash
python build_dataset_from_detector.py
```

This processes raw images, crops detected inspected objects, splits by `70/15/15`, and writes the output into:

```text
dataset/train/good
 dataset/train/bad
 dataset/val/good
 dataset/val/bad
 dataset/test/good
 dataset/test/bad
```

Rejected source images are written to `dataset_rejects/no_detection/` or `dataset_rejects/read_error/`.

### 6. Validate the prepared dataset

```bash
python validate_prepared_dataset.py
```

This validates the final dataset structure, reads images, ensures no zero-sized crops, checks manifest consistency, and writes a JSON report and optional preview image under `reports/`.

### 7. Run classifier training

```bash
python finetuning_classify.py
```

### 8. Run anomaly detector training

```bash
python finetuning_anomaly.py
```

## Notes

- The object detector is only responsible for finding the inspected object in the scene.
- Good/bad class labels remain based on the raw folder source.
- The goal is to preserve the existing classifier and anomaly dataset contract exactly.

## Troubleshooting

- If `ultralytics` is not installed, install it with:

```bash
pip install ultralytics
```

- If OpenCV is not installed and the preview generation fails, install:

```bash
pip install opencv-python
```
