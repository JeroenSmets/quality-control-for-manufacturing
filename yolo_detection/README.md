# YOLO Object Detection

This stage builds and validates the YOLO dataset, trains the inspected-object detector, tests it on unseen raw images, and uses it to crop the classifier dataset.

Scripts:

- `prepare_detection_dataset.py`
- `validate_object_detection_dataset.py`
- `train_object_detector.py`
- `test_object_detector_on_unseen.py`
- `build_dataset_from_detector.py`

Inputs:

- `sam_labeling/detector_labeling_pool/`
- `raw_data/casting_512x512/`
- Optional pretrained checkpoint in `yolo_detection/checkpoints/`

Outputs:

- `yolo_detection/object_detection_dataset/`
- `yolo_detection/runs/`
- `classifier/dataset/`
- `classifier/dataset_rejects/`

Run from the repository root:

```bash
python yolo_detection/prepare_detection_dataset.py --source-dir sam_labeling/detector_labeling_pool
python yolo_detection/validate_object_detection_dataset.py
python yolo_detection/train_object_detector.py --device 0 --epochs 100 --batch-size 4
python yolo_detection/test_object_detector_on_unseen.py --device 0 --count-per-class 10
python yolo_detection/build_dataset_from_detector.py --weights yolo_detection/runs/detect/inspection_object_detector/weights/best.pt --device 0
```

Use `--overwrite` with `prepare_detection_dataset.py` to recreate `yolo_detection/object_detection_dataset/`.
