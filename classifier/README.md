# Classifier

This stage validates the detector-cropped dataset and trains the good/bad image classifier.

Scripts:

- `validate_prepared_dataset.py`
- `finetuning_classify.py`
- `data_structuring.py` legacy manual ROI utility
- `testing.py` legacy classifier evaluation utility

Inputs:

- `classifier/dataset/train/good/`
- `classifier/dataset/train/bad/`
- `classifier/dataset/val/good/`
- `classifier/dataset/val/bad/`
- `classifier/dataset/test/good/`
- `classifier/dataset/test/bad/`

Outputs:

- `classifier/reports/`
- `classifier/runs/`
- `classifier/runs/qc_classifier.pt`

Run from the repository root:

```bash
python classifier/validate_prepared_dataset.py
python classifier/finetuning_classify.py --device 0
```

The dataset is normally created by `yolo_detection/build_dataset_from_detector.py`.
