import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from segment_anything import SamPredictor, sam_model_registry
    SAM_AVAILABLE = True
except ImportError:
    SamPredictor = None
    sam_model_registry = None
    SAM_AVAILABLE = False

from project_config import IMAGE_EXTENSIONS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Interactive point-based segmentation labeler for detector sample images."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("detector_labeling_pool"),
        help="Directory containing images to label.",
    )
    parser.add_argument(
        "--labels-dir",
        type=Path,
        default=Path("detector_labeling_pool"),
        help="Directory to write YOLO label .txt files.",
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        default=None,
        help="Optional directory to save binary masks for review.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional Segment Anything checkpoint file for point-guided mask prediction. "
            "If provided, the script will use SAM instead of GrabCut."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run SAM on, e.g. 'cpu' or 'cuda'. If omitted, auto-detects.",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Disable converting the SAM model to fp16 (avoid if ROCm causes crashes).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision (autocast) during SAM prediction.",
    )
    parser.add_argument(
        "--sam-scale",
        type=float,
        default=1.0,
        help="Scale factor for SAM inference (0 < scale <= 1). Smaller values speed up inference.",
    )
    parser.add_argument(
        "--sam-model",
        type=str,
        default="vit_b",
        choices=["vit_b", "vit_l", "vit_h"],
        help="SAM backbone model type when using --sam-checkpoint.",
    )
    parser.add_argument(
        "--class-id",
        type=int,
        default=0,
        help="YOLO class id for the inspected object.",
    )
    return parser.parse_args()


def find_images(image_dir: Path):
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_paths:
        raise RuntimeError(f"No supported images found in: {image_dir}")

    return image_paths


def save_label(label_path: Path, class_id: int, box, image_shape):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    if box is None:
        label_path.write_text("", encoding="utf-8")
        return

    x1, y1, x2, y2 = box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_shape[1], x2), min(image_shape[0], y2)
    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:
        raise ValueError("Invalid bounding box coordinates for saving.")

    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    label_path.write_text(
        f"{class_id} {x_center / image_shape[1]:.6f} {y_center / image_shape[0]:.6f} {width / image_shape[1]:.6f} {height / image_shape[0]:.6f}\n",
        encoding="utf-8",
    )


def load_sam_predictor(checkpoint_path: Path, model_type: str = "vit_b", device: str | torch.device = None, no_half: bool = False):
    if not SAM_AVAILABLE:
        raise RuntimeError(
            "segment-anything is not installed. Install it with pip install segment-anything"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    model.to(device)
    if device.type == "cuda" and not no_half:
        try:
            model.half()
        except Exception:
            pass

    predictor = SamPredictor(model)
    return predictor


def build_grabcut_mask(image, fg_points, bg_points):
    mask = np.full(image.shape[:2], cv2.GC_PR_BGD, dtype=np.uint8)

    if fg_points or bg_points:
        points = fg_points + bg_points
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        min_x, max_x = max(min(xs) - 20, 0), min(max(xs) + 20, image.shape[1] - 1)
        min_y, max_y = max(min(ys) - 20, 0), min(max(ys) + 20, image.shape[0] - 1)
        rect = (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
    else:
        rect = (1, 1, image.shape[1] - 2, image.shape[0] - 2)

    for x, y in fg_points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            mask[y, x] = cv2.GC_FGD
    for x, y in bg_points:
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            mask[y, x] = cv2.GC_BGD

    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    if np.any(fg_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return fg_mask


def build_sam_mask(image, fg_points, bg_points, predictor, use_amp: bool = True, scale: float = 1.0):
    if predictor is None:
        raise RuntimeError("SAM predictor is not loaded.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]

    if scale <= 0 or scale > 1.0:
        raise ValueError("--sam-scale must be in (0, 1].")

    if scale != 1.0:
        small_w = max(1, int(orig_w * scale))
        small_h = max(1, int(orig_h * scale))
        image_for_sam = cv2.resize(image_rgb, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    else:
        image_for_sam = image_rgb

    predictor.set_image(image_for_sam)

    point_coords = []
    point_labels = []
    for x, y in fg_points:
        point_coords.append([x, y])
        point_labels.append(1)
    for x, y in bg_points:
        point_coords.append([x, y])
        point_labels.append(0)

    if len(point_coords) == 0:
        return None

    point_coords = np.array(point_coords, dtype=np.float32)
    point_labels = np.array(point_labels, dtype=np.int32)

    # scale point coordinates for the resized image
    if scale != 1.0:
        point_coords = point_coords * scale

    # Use autocast on CUDA to accelerate fp16 models
    try:
        model_device = next(iter(predictor.model.parameters())).device
    except Exception:
        model_device = None

    if model_device is not None and model_device.type == "cuda" and use_amp:
        with torch.cuda.amp.autocast():
            masks, scores, logits = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )

    if masks is None or len(masks) == 0:
        return None

    fg_mask = (masks[0].astype(np.uint8) * 255)
    # if mask was produced on a smaller image, scale it back to original size
    if scale != 1.0:
        fg_mask = cv2.resize(fg_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    if np.any(fg_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    return fg_mask


def mask_to_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def draw_points(image, points, color):
    for x, y in points:
        cv2.circle(image, (x, y), 4, color, -1)


def overlay_mask(image, mask):
    colored = np.zeros_like(image)
    colored[:, :, 1] = mask
    return cv2.addWeighted(image, 0.75, colored, 0.25, 0)


def main():
    args = parse_args()
    image_paths = find_images(args.image_dir)
    args.labels_dir.mkdir(parents=True, exist_ok=True)

    sam_predictor = None
    if args.sam_checkpoint is not None:
        # allow user to override device via --device; otherwise autodetect
        device_arg = args.device if args.device is not None else None
        sam_predictor = load_sam_predictor(args.sam_checkpoint, args.sam_model, device_arg, no_half=args.no_half)
        print(f"Loaded SAM {args.sam_model} checkpoint: {args.sam_checkpoint}")

    index = 0
    fg_points = []
    bg_points = []
    current_mask = None
    current_bbox = None

    def on_mouse(event, x, y, flags, param):
        nonlocal fg_points, bg_points
        if event == cv2.EVENT_LBUTTONDOWN:
            fg_points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            bg_points.append((x, y))

    window_name = "Smart Mask Labeler"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    while True:
        image_path = image_paths[index]
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to load image: {image_path}")

        label_path = args.labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            saved_text = label_path.read_text(encoding="utf-8").strip()
            current_bbox = None
            current_mask = None
            if saved_text:
                parts = saved_text.split()
                if len(parts) == 5:
                    _, x_center, y_center, width, height = map(float, parts)
                    x1 = int((x_center - width / 2.0) * image.shape[1])
                    y1 = int((y_center - height / 2.0) * image.shape[0])
                    x2 = int((x_center + width / 2.0) * image.shape[1])
                    y2 = int((y_center + height / 2.0) * image.shape[0])
                    current_bbox = (x1, y1, x2, y2)
        else:
            current_bbox = None
            current_mask = None

        while True:
            frame = image.copy()
            if current_mask is not None:
                frame = overlay_mask(frame, current_mask)
            draw_points(frame, fg_points, (0, 255, 0))
            draw_points(frame, bg_points, (0, 0, 255))
            if current_bbox is not None:
                x1, y1, x2, y2 = current_bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            instructions = [
                "Left click: add foreground point",
                "Right click: add background point",
                "g: compute mask (SAM if configured else GrabCut)",
                "s: save bounding box",
                "c: clear points/mask",
                "n: next image",
                "p: previous image",
                "q: quit",
            ]
            for i, text in enumerate(instructions):
                cv2.putText(
                    frame,
                    text,
                    (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"Image {index + 1}/{len(image_paths)}: {image_path.name}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord("q"):
                cv2.destroyAllWindows()
                return
            elif key == ord("n"):
                break
            elif key == ord("p"):
                index = max(0, index - 1)
                fg_points = []
                bg_points = []
                current_mask = None
                current_bbox = None
                break
            elif key == ord("c"):
                fg_points = []
                bg_points = []
                current_mask = None
                current_bbox = None
            elif key == ord("g"):
                if fg_points or bg_points:
                    if sam_predictor is not None:
                        try:
                            current_mask = build_sam_mask(image, fg_points, bg_points, sam_predictor, use_amp=(not args.no_amp))
                        except Exception as exc:
                            print(f"SAM segmentation failed: {exc}. Falling back to GrabCut.")
                            current_mask = build_grabcut_mask(image, fg_points, bg_points)
                    else:
                        current_mask = build_grabcut_mask(image, fg_points, bg_points)

                    current_bbox = mask_to_bbox(current_mask)
                    if current_bbox is None:
                        print("Mask did not produce a valid object region. Add more points.")
                else:
                    print("Add at least one foreground or background point before computing the mask.")
            elif key == ord("s"):
                if current_bbox is None:
                    print("No mask bounding box available. Press g to compute the mask first.")
                else:
                    save_label(label_path, args.class_id, current_bbox, image.shape)
                    if args.mask_dir is not None and current_mask is not None:
                        args.mask_dir.mkdir(parents=True, exist_ok=True)
                        mask_path = args.mask_dir / f"{image_path.stem}_mask.png"
                        cv2.imwrite(str(mask_path), current_mask)
                        print(f"Saved mask: {mask_path}")
                    print(f"Saved label: {label_path}")
            elif key == 27:
                cv2.destroyAllWindows()
                return

        index = min(len(image_paths) - 1, index + 1)


if __name__ == "__main__":
    main()
