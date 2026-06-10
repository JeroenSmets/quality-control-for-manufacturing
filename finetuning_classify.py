from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import timm

from project_config import DATASET_ROOT, NUM_WORKERS


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

EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 20

LR_HEAD = 1e-3
LR_FINETUNE = 1e-5

PATIENCE = 5

OUTPUT_MODEL_PATH = "qc_classifier.pt"


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
    epochs,
    lr,
    patience,
):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

    best_model = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    bad_epochs = 0

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

    return model


def evaluate_on_test_set(model, test_loader, device, class_names):
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=True)

            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(labels.tolist())

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


# =========================
# Main
# =========================

def main():
    device = torch.device("cpu")

    print("Using:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_tfms, eval_tfms = build_transforms()

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tfms)

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

    print("Class counts:", {
        class_names[i]: int(class_counts[i])
        for i in range(len(class_names))
    })

    if torch.any(class_counts == 0):
        raise RuntimeError(
            "At least one class has zero training images. "
            "Check your dataset/train/good and dataset/train/bad folders."
        )

    class_weights = 1.0 / class_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = class_weights.to(device)

    print("Class weights:", {
        class_names[i]: float(class_weights[i].cpu())
        for i in range(len(class_names))
    })

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

    model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=EPOCHS_HEAD,
        lr=LR_HEAD,
        patience=PATIENCE,
    )

    # Stage 2: fine-tune the full model gently
    print("\nStage 2: fine-tuning full model")

    unfreeze_all_layers(model)

    trainable, total = count_trainable_parameters(model)
    print(f"Trainable parameters: {trainable:,} / {total:,}")

    model = train_stage(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        epochs=EPOCHS_FINETUNE,
        lr=LR_FINETUNE,
        patience=PATIENCE,
    )

    # Save model
    torch.save(
        {
            "model_name": MODEL_NAME,
            "state_dict": model.state_dict(),
            "class_names": class_names,
            "img_size": IMG_SIZE,
        },
        OUTPUT_MODEL_PATH,
    )

    print(f"\nSaved model to: {OUTPUT_MODEL_PATH}")

    # Test evaluation
    evaluate_on_test_set(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names,
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()