from pathlib import Path
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.project_config import CLASSIFIER_MODEL_PATH, CLASSIFIER_REPORTS_DIR, DATASET_ROOT


# =========================
# Configuration
# =========================

DATA_DIR = DATASET_ROOT
TEST_DIR = DATA_DIR / "test"

MODEL_PATH = CLASSIFIER_MODEL_PATH

BATCH_SIZE = 16
NUM_WORKERS = 0

# Set to True if your AMD ROCm GPU gives rocBLAS errors.
# This is recommended for your current setup.
FORCE_CPU = True


# =========================
# Helper functions
# =========================

def load_checkpoint(model_path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint


def build_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def build_model(checkpoint, device):
    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=len(class_names),
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    return model


def evaluate_model(model, test_loader, device, class_names):
    y_true = []
    y_pred = []
    y_prob_bad = []
    image_paths = []

    bad_class_index = class_names.index("bad") if "bad" in class_names else 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(predictions.cpu().tolist())
            y_prob_bad.extend(probabilities[:, bad_class_index].cpu().tolist())

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names,
        )
    )

    return y_true, y_pred, y_prob_bad


def print_wrong_predictions(test_dataset, y_true, y_pred, y_prob_bad, class_names):
    print("\nWrong predictions:")
    print("------------------")

    wrong_count = 0

    for idx, (true_idx, pred_idx, prob_bad) in enumerate(zip(y_true, y_pred, y_prob_bad)):
        if true_idx != pred_idx:
            image_path = test_dataset.samples[idx][0]
            true_label = class_names[true_idx]
            predicted_label = class_names[pred_idx]

            print(
                f"{image_path} | "
                f"true={true_label} | "
                f"predicted={predicted_label} | "
                f"P(bad)={prob_bad:.4f}"
            )

            wrong_count += 1

    if wrong_count == 0:
        print("No wrong predictions found.")

    print(f"\nTotal wrong predictions: {wrong_count}")


def save_predictions_csv(test_dataset, y_true, y_pred, y_prob_bad, class_names):
    output_path = CLASSIFIER_REPORTS_DIR / "test_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("image_path,true_label,predicted_label,prob_bad,correct\n")

        for idx, (true_idx, pred_idx, prob_bad) in enumerate(zip(y_true, y_pred, y_prob_bad)):
            image_path = test_dataset.samples[idx][0]
            true_label = class_names[true_idx]
            predicted_label = class_names[pred_idx]
            correct = true_idx == pred_idx

            f.write(
                f'"{image_path}",'
                f"{true_label},"
                f"{predicted_label},"
                f"{prob_bad:.6f},"
                f"{correct}\n"
            )

    print(f"\nSaved detailed predictions to: {output_path}")


# =========================
# Main
# =========================

def main():
    if FORCE_CPU:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    checkpoint = load_checkpoint(MODEL_PATH)

    model_name = checkpoint["model_name"]
    class_names = checkpoint["class_names"]
    img_size = checkpoint["img_size"]

    print("Model:", model_name)
    print("Classes from checkpoint:", class_names)
    print("Image size:", img_size)

    eval_transform = build_eval_transform(img_size)

    test_dataset = datasets.ImageFolder(
        TEST_DIR,
        transform=eval_transform,
    )

    print("Classes from test folder:", test_dataset.classes)
    print("Test images:", len(test_dataset))

    if test_dataset.classes != class_names:
        raise RuntimeError(
            "Class mismatch between checkpoint and test folder.\n"
            f"Checkpoint classes: {class_names}\n"
            f"Test folder classes: {test_dataset.classes}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model(checkpoint, device)

    y_true, y_pred, y_prob_bad = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    print_wrong_predictions(
        test_dataset=test_dataset,
        y_true=y_true,
        y_pred=y_pred,
        y_prob_bad=y_prob_bad,
        class_names=class_names,
    )

    save_predictions_csv(
        test_dataset=test_dataset,
        y_true=y_true,
        y_pred=y_pred,
        y_prob_bad=y_prob_bad,
        class_names=class_names,
    )


if __name__ == "__main__":
    main()
