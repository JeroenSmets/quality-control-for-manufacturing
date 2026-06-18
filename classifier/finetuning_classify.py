from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime
import csv
import copy
import json
import sys
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.project_config import CLASSIFIER_MODEL_PATH, CLASSIFIER_RUNS_DIR, DATASET_ROOT, NUM_WORKERS


# =========================
# Configuration
# =========================

DATA_DIR = DATASET_ROOT

MODEL_NAME = "efficientnet_b0"
# Other options to try:
# MODEL_NAME = "resnet18"
# MODEL_NAME = "convnext_tiny"
# MODEL_NAME = "tf_efficientnetv2_s"

IMG_SIZE = 224
BATCH_SIZE = 8

EPOCHS_HEAD = 50
EPOCHS_FINETUNE = 50

LR_HEAD = 1e-3
LR_FINETUNE = 1e-5

PATIENCE = 20

OUTPUT_MODEL_PATH = CLASSIFIER_MODEL_PATH
RUN_MODEL_FILENAME = "qc_classifier.pt"
RUNS_DIR = CLASSIFIER_RUNS_DIR


# =========================
# Arguments
# =========================

def parse_args():
    parser = ArgumentParser(
        description="Train the good/bad classifier on detector-cropped images."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DATA_DIR,
        help="Root of the prepared classifier dataset.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device: cpu, cuda, cuda:0, or 0.",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory where timestamped classifier run results are saved.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str):
    normalized = device_arg.strip().lower()

    if normalized == "cpu":
        return torch.device("cpu")

    if normalized.isdigit():
        normalized = f"cuda:{normalized}"

    if normalized == "cuda":
        normalized = "cuda:0"

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
        return torch.device(normalized)

    raise ValueError(f"Unsupported device: {device_arg}")


# =========================
# Transforms
# =========================

def build_transforms():
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_tfms, eval_tfms


# =========================
# Model helpers
# =========================

def freeze_backbone(model):
    """
    Freeze all model parameters, then unfreeze only the classifier head.
    """
    for param in model.parameters():
        param.requires_grad = False

    classifier = model.get_classifier()

    if isinstance(classifier, nn.Module):
        for param in classifier.parameters():
            param.requires_grad = True
    else:
        raise RuntimeError(
            "Could not access classifier head. "
            "Try a different timm model or inspect model.get_classifier()."
        )


def unfreeze_all_layers(model):
    """
    Unfreeze the full model for gentle fine-tuning.
    """
    for param in model.parameters():
        param.requires_grad = True


def count_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# =========================
# Result saving helpers
# =========================

def create_run_dir(runs_dir: Path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_dir / f"{timestamp}_{MODEL_NAME}"
    suffix = 1

    while run_dir.exists():
        run_dir = runs_dir / f"{timestamp}_{MODEL_NAME}_{suffix}"
        suffix += 1

    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, data):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_training_history_csv(run_dir: Path, history):
    history_path = run_dir / "training_history.csv"
    fieldnames = [
        "global_epoch",
        "stage",
        "stage_epoch",
        "train_loss",
        "train_acc",
        "val_loss",
        "val_acc",
        "lr",
    ]

    with history_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return history_path


def save_training_curves(run_dir: Path, history):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping training_curves.png")
        return None

    if not history:
        return None

    epochs = [row["global_epoch"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, [row["train_loss"] for row in history], label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [row["train_acc"] for row in history], label="train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend()

    previous_stage = history[0]["stage"]
    for index, row in enumerate(history[1:], start=1):
        if row["stage"] != previous_stage:
            for axis in axes:
                axis.axvline(row["global_epoch"] - 0.5, color="0.6", linestyle="--", linewidth=1)
            previous_stage = row["stage"]

    fig.tight_layout()
    plot_path = run_dir / "training_curves.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def save_confusion_matrix_csv(run_dir: Path, matrix, class_names):
    matrix_path = run_dir / "confusion_matrix.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["true_label"] + [f"pred_{name}" for name in class_names])
        for label, row in zip(class_names, matrix):
            writer.writerow([label] + [int(value) for value in row])

    return matrix_path


def save_confusion_matrix_plot(run_dir: Path, matrix, class_names):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping confusion_matrix.png")
        return None

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    fig.colorbar(image, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = int(matrix[row_index, col_index])
            color = "white" if value > threshold else "black"
            ax.text(col_index, row_index, value, ha="center", va="center", color=color)

    fig.tight_layout()
    plot_path = run_dir / "confusion_matrix.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
    return plot_path


def save_test_predictions_csv(run_dir: Path, test_dataset, evaluation, class_names):
    predictions_path = run_dir / "test_predictions.csv"
    probability_fields = [f"prob_{name}" for name in class_names]
    fieldnames = [
        "image_path",
        "true_label",
        "predicted_label",
        "correct",
        *probability_fields,
    ]

    with predictions_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for index, (true_index, pred_index, probabilities) in enumerate(
            zip(evaluation["y_true"], evaluation["y_pred"], evaluation["probabilities"])
        ):
            row = {
                "image_path": test_dataset.samples[index][0],
                "true_label": class_names[true_index],
                "predicted_label": class_names[pred_index],
                "correct": true_index == pred_index,
            }
            row.update({
                probability_fields[class_index]: f"{probability:.6f}"
                for class_index, probability in enumerate(probabilities)
            })
            writer.writerow(row)

    return predictions_path


def save_run_results(
    run_dir: Path,
    history,
    evaluation,
    test_dataset,
    class_names,
    checkpoint,
    best_stage_metrics,
):
    save_training_history_csv(run_dir, history)
    save_training_curves(run_dir, history)

    matrix = evaluation["confusion_matrix"]
    save_confusion_matrix_csv(run_dir, matrix, class_names)
    save_confusion_matrix_plot(run_dir, matrix, class_names)

    report_text_path = run_dir / "classification_report.txt"
    report_text_path.write_text(evaluation["classification_report_text"], encoding="utf-8")
    write_json(run_dir / "classification_report.json", evaluation["classification_report_dict"])

    save_test_predictions_csv(run_dir, test_dataset, evaluation, class_names)

    metrics = {
        "best_stage_metrics": best_stage_metrics,
        "confusion_matrix": matrix.tolist(),
        "classification_report": evaluation["classification_report_dict"],
    }
    write_json(run_dir / "metrics.json", metrics)

    run_model_path = run_dir / RUN_MODEL_FILENAME
    torch.save(checkpoint, run_model_path)
    return run_model_path


# =========================
# Training helpers
# =========================

def run_epoch(model, loader, criterion, device, optimizer=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    avg_acc = correct / total

    return avg_loss, avg_acc


def train_stage(
    model,
    train_loader,
    val_loader,
    criterion,
    device,
    stage_name,
    epochs,
    lr,
    patience,
    global_epoch_start,
):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    bad_epochs = 0
    stage_history = []

    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
        )

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        stage_history.append(
            {
                "global_epoch": global_epoch_start + epoch + 1,
                "stage": stage_name,
                "stage_epoch": epoch + 1,
                "train_loss": float(train_loss),
                "train_acc": float(train_acc),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(lr),
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model)
    print(f"Best validation accuracy in this stage: {best_val_acc:.3f}")

    return model, stage_history, best_val_acc


def evaluate_on_test_set(model, test_loader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []
    probabilities = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            batch_probabilities = torch.softmax(outputs, dim=1)
            preds = batch_probabilities.argmax(dim=1).cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(labels.tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())

    matrix = confusion_matrix(y_true, y_pred)
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(report_text)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "probabilities": probabilities,
        "confusion_matrix": matrix,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
    }


# =========================
# Main
# =========================

def main():
    args = parse_args()
    device = resolve_device(args.device)
    run_dir = create_run_dir(args.runs_dir)
    data_dir = args.dataset_root

    print("Using:", device)
    print("Run results:", run_dir)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device.index or 0))

    train_tfms, eval_tfms = build_transforms()

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tfms)

    class_names = train_ds.classes

    print("Classes:", class_names)
    print(f"Train images: {len(train_ds)}")
    print(f"Val images:   {len(val_ds)}")
    print(f"Test images:  {len(test_ds)}")

    if len(class_names) != 2:
        raise RuntimeError(
            f"Expected exactly 2 classes, but found {len(class_names)}: {class_names}"
        )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    # Handle class imbalance
    class_counts = torch.bincount(torch.tensor(train_ds.targets))

    class_counts_dict = {
        class_names[i]: int(class_counts[i])
        for i in range(len(class_names))
    }
    print("Class counts:", class_counts_dict)

    if torch.any(class_counts == 0):
        raise RuntimeError(
            "At least one class has zero training images. "
            "Check your dataset/train/good and dataset/train/bad folders."
        )

    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    class_weights_dict = {
        class_names[i]: float(class_weights[i].cpu())
        for i in range(len(class_names))
    }
    print("Class weights:", class_weights_dict)

    write_json(
        run_dir / "config.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device.index or 0) if device.type == "cuda" else None,
            "data_dir": str(data_dir),
            "model_name": MODEL_NAME,
            "image_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs_head": EPOCHS_HEAD,
            "epochs_finetune": EPOCHS_FINETUNE,
            "lr_head": LR_HEAD,
            "lr_finetune": LR_FINETUNE,
            "patience": PATIENCE,
            "num_workers": NUM_WORKERS,
            "class_names": class_names,
            "class_counts": class_counts_dict,
            "class_weights": class_weights_dict,
            "train_images": len(train_ds),
            "val_images": len(val_ds),
            "test_images": len(test_ds),
            "output_model_path": str(OUTPUT_MODEL_PATH),
            "run_dir": str(run_dir),
        },
    )

    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=len(class_names),
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Stage 1: train only the classifier head
    print("\nStage 1: training classifier head only")

    freeze_backbone(model)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    history = []
    best_stage_metrics = {}

    model, stage_history, best_val_acc = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        stage_name="head",
        epochs=EPOCHS_HEAD,
        lr=LR_HEAD,
        patience=PATIENCE,
        global_epoch_start=len(history),
    )
    history.extend(stage_history)
    best_stage_metrics["head"] = {
        "best_val_acc": float(best_val_acc),
        "epochs_run": len(stage_history),
    }

    # Stage 2: fine-tune the full model gently
    print("\nStage 2: fine-tuning full model")

    unfreeze_all_layers(model)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    model, stage_history, best_val_acc = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        stage_name="finetune",
        epochs=EPOCHS_FINETUNE,
        lr=LR_FINETUNE,
        patience=PATIENCE,
        global_epoch_start=len(history),
    )
    history.extend(stage_history)
    best_stage_metrics["finetune"] = {
        "best_val_acc": float(best_val_acc),
        "epochs_run": len(stage_history),
    }

    # Save model
    checkpoint = {
        "model_name": MODEL_NAME,
        "state_dict": model.state_dict(),
        "class_names": class_names,
        "img_size": IMG_SIZE,
    }
    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, OUTPUT_MODEL_PATH)

    print(f"\nSaved model to: {OUTPUT_MODEL_PATH}")

    # Test evaluation
    evaluation = evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )

    run_model_path = save_run_results(
        run_dir=run_dir,
        history=history,
        evaluation=evaluation,
        test_dataset=test_ds,
        class_names=class_names,
        checkpoint=checkpoint,
        best_stage_metrics=best_stage_metrics,
    )
    print(f"\nSaved run results to: {run_dir}")
    print(f"Saved run model to: {run_model_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
