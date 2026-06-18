# Anomaly Detection

This stage converts the detector-cropped classifier dataset into an Anomalib folder dataset and trains a PatchCore anomaly detector.

Script:

- `finetuning_anomaly.py`

Inputs:

- `classifier/dataset/`

Outputs:

- `anomaly_detection/anomalib_dataset/`
- `anomaly_detection/runs/`

Run from the repository root:

```bash
python anomaly_detection/finetuning_anomaly.py --device 0
```

PatchCore trains on normal `good` images and evaluates on both `good` and `bad` crops.
