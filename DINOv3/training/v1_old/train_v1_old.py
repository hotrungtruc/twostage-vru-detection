import argparse
import os
import csv
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. MODEL DEFINITION (BACKBONE + LINEAR HEAD)
# ==========================================
class LightweightDINO(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super(LightweightDINO, self).__init__()
        # Load DINOv3 Backbone via timm
        print(f"Loading backbone: {model_name}...")
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # Freeze Backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Linear Head (DINOv3 ViT-S features = 384)
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.classifier(features)
        return logits

# ==========================================
# 2. DATA PREPARATION
# ==========================================
def prepare_data(data_dir, model_name, batch_size, num_workers):
    model_cfg = timm.data.resolve_model_data_config(timm.create_model(model_name))
    train_transform = timm.data.create_transform(**model_cfg, is_training=True)
    val_transform = timm.data.create_transform(**model_cfg, is_training=False)

    train_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    indices = list(range(len(train_dataset)))
    train_size = int(0.8 * len(train_dataset))

    np.random.seed(42)
    np.random.shuffle(indices)

    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_set = torch.utils.data.Subset(train_dataset, train_idx)
    val_set = torch.utils.data.Subset(val_dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print(f"Dataset classes: {train_dataset.classes}")
    print(f"Train size: {len(train_set)}, Val size: {len(val_set)}")

    return train_loader, val_loader, train_dataset.classes

# ==========================================
# 3. TRAINING ENGINE
# ==========================================
def train_model(args):
    # Auto select device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Available GPUs: {torch.cuda.device_count()}")

    # Prepare data
    if not os.path.exists(args.data_dir):
        print(f"[ERROR] Data directory '{args.data_dir}' not found. Please provide a valid dataset.")
        return

    train_loader, val_loader, class_names = prepare_data(
        args.data_dir, args.model_name, args.batch_size, args.workers
    )

    # Initialize model
    model = LightweightDINO(args.model_name, num_classes=len(class_names))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for parallel training!")
        model = nn.DataParallel(model)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    classifier_params = model.module.classifier.parameters() if isinstance(model, nn.DataParallel) else model.classifier.parameters()
    optimizer = optim.Adam(classifier_params, lr=args.lr)

    best_acc = 0.0
    history = {'train_loss': [], 'val_acc': []}
    
    # Prepare output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "dinov3_lightweight_classifier_v1_old.pth"
    plot_path = output_dir / "training_report_v1_old.png"
    csv_path = output_dir / "training_stats_v1_old.csv"

    # Initialize CSV header
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy'])

    print("\n[STARTING] Fine-Tuning...")
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

        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)

        print(f"Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}%")
        
        # Append stats to CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{epoch_loss:.4f}", f"{epoch_acc:.2f}"])

        # Save best model
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            model_state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save({
                'model_state_dict': model_state,
                'class_names': class_names,
                'backbone_name': args.model_name
            }, save_path)
            print(f"[BEST] Model saved with Accuracy: {best_acc:.2f}%")

    print(f"\n[DONE] Training Complete! Model saved at: {save_path}")

    # ---------------------------------------------------------
    # Final Evaluation: Classification Report & Confusion Matrix
    # ---------------------------------------------------------
    print("\n[*] Running final evaluation on validation set...")
    checkpoint = torch.load(save_path, map_location=device, weights_only=True)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.numpy())
            
    # Generate Classification Report
    report = classification_report(all_targets, all_preds, target_names=class_names)
    report_path = output_dir / "classification_report_v1.txt"
    with open(report_path, "w") as f:
        f.write("Classification Report (v1_old)\n")
        f.write("==============================\n")
        f.write(report)
    print(f"[*] Classification report saved: {report_path.name}")
    print(report)
    
    # Generate Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (v1_old)')
    cm_path = output_dir / "confusion_matrix_v1.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"[*] Confusion matrix saved: {cm_path.name}")

    # Plot results
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.savefig(plot_path)
    plt.close()
    print(f"[REPORT] Training report saved at: {plot_path}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(current_dir, "labeled_data_v1_old")
    default_output_dir = os.path.join(current_dir, "output")

    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 for Cyclist State Classification")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Path to labeled dataset")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_dinov3.lvd1689m", help="DINOv3 backbone name")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers for CPU parallelization")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Directory to save the trained model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (overridden by auto-detect)")
    
    args = parser.parse_args()
    
    # Autonomously disable multiprocessing on macOS to prevent native fork/spawn crashes
    import platform
    if platform.system() == "Darwin" and args.workers > 0:
        print("[!] macOS detected: Automatically setting DataLoader workers to 0 to ensure stability.")
        args.workers = 0

    train_model(args)
