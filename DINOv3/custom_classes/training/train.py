import argparse
import os
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from PIL import Image

def filter_dataset_inplace(dataset, threshold=8):
    """
    In-place deduplication of dataset using dHash (difference hash).
    Filters out perceptually identical images to prevent data leakage and overfitting.
    """
    print(f"[*] Starting dHash deduplication (Hamming distance threshold: {threshold})")
    hashes = {}
    valid_samples = []
    duplicates_count = 0
    
    for path, target in tqdm(dataset.samples, desc="Filtering"):
        img = cv2.imread(path)
        if img is None:
            continue
        
        # Compute 64-bit dHash
        resized = cv2.resize(img, (9, 8))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        diff = gray[:, 1:] > gray[:, :-1]
        h = sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
        
        # Check against existing hashes
        is_duplicate = False
        for existing_path, existing_hash in hashes.items():
            dist = bin(h ^ existing_hash).count('1')
            if dist <= threshold:
                is_duplicate = True
                break
                
        if is_duplicate:
            duplicates_count += 1
        else:
            hashes[path] = h
            valid_samples.append((path, target))
            
    # Update dataset state
    dataset.samples = valid_samples
    dataset.targets = [s[1] for s in valid_samples]
    
    print(f"[-] Removed {duplicates_count} redundant samples. Remaining: {len(valid_samples)}")


class CustomImageFolder(Dataset):
    """
    Custom Dataset to load images from a structure like:
    dataset/
      class_A/
        train/
        valid/
      class_B/
        train/
        valid/
    """
    def __init__(self, root, split='train', transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        
        # Automatically detect classes from folder names
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for cls_name in self.classes:
            split_dir = self.root / cls_name / split
            if split_dir.exists():
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    for img_path in split_dir.glob(ext):
                        self.samples.append((str(img_path), self.class_to_idx[cls_name]))
        
        self.targets = [s[1] for s in self.samples]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            with open(path, "rb") as f:
                img = Image.open(f).convert("RGB")
        except Exception:
            # Fallback for corrupt images
            img = Image.new('RGB', (224, 224), color=0)
            
        if self.transform is not None:
            img = self.transform(img)
            
        return img, target


class LightweightDINO(nn.Module):
    """
    Lightweight classification head attached to a frozen DINOv3 backbone.
    """
    def __init__(self, model_name, num_classes):
        super(LightweightDINO, self).__init__()
        
        print(f"[*] Initializing frozen backbone: {model_name}")
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        return self.classifier(features)


def prepare_data(data_dir, model_name, batch_size, num_workers, threshold=8):
    model_cfg = timm.data.resolve_model_data_config(timm.create_model(model_name))
    train_transform = timm.data.create_transform(**model_cfg, is_training=True)
    val_transform = timm.data.create_transform(**model_cfg, is_training=False)

    train_dataset = CustomImageFolder(data_dir, split='train', transform=train_transform)
    val_dataset = CustomImageFolder(data_dir, split='valid', transform=val_transform)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise FileNotFoundError(f"Missing images in train or valid splits inside {data_dir}")

    print("\n[+] Processing Training Split")
    filter_dataset_inplace(train_dataset, threshold=threshold)
    
    print("\n[+] Processing Validation Split")
    filter_dataset_inplace(val_dataset, threshold=threshold)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"\n[*] Dynamic classes detected: {train_dataset.classes}")
    
    train_count = len(train_dataset)
    val_count = len(val_dataset)
    total = train_count + val_count
    
    if total > 0:
        train_ratio = (train_count / total) * 100
        val_ratio = (val_count / total) * 100
        print(f"[*] Final split ratio: {train_ratio:.1f}% Train | {val_ratio:.1f}% Valid ({total} total samples)")

    return train_loader, val_loader, train_dataset.classes


def train_model(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[*] Compute device: {device}")
    
    if not os.path.exists(args.data_dir):
        print(f"[!] Error: Dataset directory '{args.data_dir}' not found.")
        return

    train_loader, val_loader, class_names = prepare_data(
        args.data_dir, args.model_name, args.batch_size, args.workers, args.threshold
    )

    if len(train_loader.dataset) == 0:
        print("[!] Error: Training dataset is empty after deduplication.")
        return

    num_classes = len(class_names)
    print(f"[*] Total classes for model output: {num_classes}")

    model = LightweightDINO(args.model_name, num_classes=num_classes)
    
    if torch.cuda.device_count() > 1:
        print(f"[*] Enabling DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
        
    model = model.to(device)
    
    # Compute dynamic class weights to handle any remaining imbalances
    targets = train_loader.dataset.targets
    class_counts = np.bincount(targets)
    total_samples = len(targets)
    class_weights = total_samples / (len(class_counts) * class_counts)
        
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print(f"[*] Computed class weights: {class_weights_tensor.cpu().numpy()}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    classifier_params = model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()
    optimizer = optim.Adam(classifier_params, lr=args.lr)

    best_roc = 0.0
    history = {'train_loss': [], 'val_acc': [], 'val_roc': []}
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = output_dir / "dinov3_custom_classifier.pth"
    plot_path = output_dir / "training_report.png"
    csv_path = output_dir / "training_stats.csv"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_ROC_AUC'])

    print("\n[*] Commencing Fine-Tuning Sequence...")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        all_targets = []
        all_probs = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                probs = torch.softmax(outputs, dim=1)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        try:
            if num_classes == 2:
                epoch_roc = roc_auc_score(all_targets, np.array(all_probs)[:, 1])
            else:
                epoch_roc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
        except ValueError:
            epoch_roc = 0.0
        
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        history['val_roc'].append(epoch_roc)

        print(f"[*] Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}% | Val ROC AUC: {epoch_roc:.4f}")
        
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{epoch_loss:.4f}", f"{epoch_acc:.2f}", f"{epoch_roc:.4f}"])

        if epoch_roc >= best_roc:
            best_roc = epoch_roc
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': model_state,
                'class_names': class_names,
                'backbone_name': args.model_name
            }, save_path)
            print(f"[+] Optimal weights captured (ROC AUC: {best_roc:.4f})")

    print(f"\n[+] Training lifecycle complete. Artifacts persisted to: {output_dir}")
    
    print("\n[*] Running final evaluation on validation set...")
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    all_preds = np.argmax(all_probs, axis=1)
        
    report = classification_report(all_targets, all_preds, target_names=class_names)
    report_path = output_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write("Classification Report\n")
        f.write("=====================\n")
        f.write(report)
    print(f"[*] Classification report saved: {report_path.name}")
    print(report)
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    cm_path = output_dir / "confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"[*] Confusion matrix saved: {cm_path.name}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot([roc * 100 for roc in history['val_roc']], label='Val ROC AUC (%)')
    plt.xlabel('Epochs')
    plt.ylabel('Score (%)')
    plt.legend()
    
    plt.savefig(plot_path)
    plt.close()
    print(f"[*] Performance curve saved: {plot_path.name}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = Path(current_dir).resolve().parent # custom_classes folder
    default_data_dir = os.path.join(project_root, "dataset")
    default_output_dir = os.path.join(project_root, "output_train")

    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 with Dynamic Classes")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Root directory containing dynamic class folders (each with train/valid)")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_dinov3.lvd1689m", help="Timm registry identifier for DINOv3 backbone")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Subprocess count for data loading")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Destination path for checkpoints and telemetry")
    parser.add_argument("--threshold", type=int, default=8, help="Hamming distance threshold for dHash deduplication")
    
    args = parser.parse_args()
    
    import platform
    if platform.system() == "Darwin" and args.workers > 0:
        print("[!] macOS detected: Automatically setting DataLoader workers to 0 to ensure stability.")
        args.workers = 0
        
    train_model(args)
