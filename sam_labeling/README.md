# SAM-Assisted Labeling

This stage samples raw good/bad casting images and creates YOLO label files for the inspected object.

Scripts:

- `sample_detection_label_images.py`
- `label_detection_images.py`

Inputs:

- `raw_data/casting_512x512/ok_front/`
- `raw_data/casting_512x512/def_front/`
- Optional SAM checkpoint in `sam_labeling/checkpoints/`

Outputs:

- `sam_labeling/detector_labeling_pool/`
- `sam_labeling/detector_labeling_pool/manifest.csv`
- One `.txt` YOLO label file per sampled image

Run from the repository root:

```bash
python sam_labeling/sample_detection_label_images.py --good-count 50 --bad-count 50
python sam_labeling/label_detection_images.py --image-dir sam_labeling/detector_labeling_pool --labels-dir sam_labeling/detector_labeling_pool
python sam_labeling/label_detection_images.py --image-dir sam_labeling/detector_labeling_pool --labels-dir sam_labeling/detector_labeling_pool --sam-checkpoint sam_labeling/checkpoints/sam_vit_b_01ec64.pth --sam-model vit_b --device cuda
```

Label only the inspected object as class `0 inspected_object`; do not label defects as detector classes.
