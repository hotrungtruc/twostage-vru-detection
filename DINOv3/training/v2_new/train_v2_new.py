import argparse
import os
import sys
import csv
import random

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_recall_curve

def set_seed(seed=42):
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def filter_dataset_inplace(dataset, threshold=8):
    """
    In-place deduplication of ImageFolder dataset using dHash (difference hash).
    Filters out perceptually identical images to prevent data leakage and overfitting.
    """
    print(f"[*] Starting dHash deduplication (Hamming distance threshold: {threshold})")
    hashes = {}
    valid_samples = []
    duplicates_count = 0
    
    for path, target in tqdm(dataset.samples, desc="Filtering dHash"):
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

class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss as described in Section 3.6.2:
        L_focal = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Default: gamma = 2.0, alpha = None (alpha_t = 1 for all classes).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class LightweightDINO(nn.Module):
    """
    Lightweight classification head attached to DINOv3 ViT-Small backbone.
    Architecture and unfreezing scheme strictly aligned with Thesis Section 3.5 & 3.6.1:
      - Backbone: vit_small_patch16_dinov3.lvd1689m (d=384)
      - Blocks 0..9: Frozen
      - Blocks 10..11 + Output LayerNorm: Unfrozen
      - Pooling: CLS, Patch-Mean, or CLS+Patch
      - Head: LayerNorm(384) -> Linear(384, 256) -> GELU -> Dropout(0.3) -> Linear(256, num_classes)
    """
    def __init__(self, model_name="vit_small_patch16_dinov3.lvd1689m", num_classes=5, pooling="cls_patch"):
        super(LightweightDINO, self).__init__()
        self.pooling = pooling.lower()
        self.num_classes = num_classes

        print(f"[*] Initializing DINOv3 backbone: {model_name}")
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)

        # 1. Freeze entire backbone first
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 2. Unfreeze the last 2 Transformer blocks (blocks 10 and 11)
        if hasattr(self.backbone, 'blocks') and len(self.backbone.blocks) >= 2:
            print("[*] Unfreezing last 2 Transformer blocks (blocks 10 & 11)")
            for block in self.backbone.blocks[-2:]:
                for param in block.parameters():
                    param.requires_grad = True

        # 3. Unfreeze output LayerNorm
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            print("[*] Unfreezing output LayerNorm layer")
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
        if hasattr(self.backbone, 'fc_norm') and self.backbone.fc_norm is not None:
            for param in self.backbone.fc_norm.parameters():
                param.requires_grad = True

        # 4. Classifier MLP Head: LayerNorm(384) -> Linear(384, 256) -> GELU -> Dropout(0.3) -> Linear(256, num_classes)
        self.classifier = nn.Sequential(
            nn.LayerNorm(384),
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x):
        """
        Extract token features from backbone and apply selected pooling scheme:
          - 'cls': g = z_CLS
          - 'patch_mean': g = (1/N) * sum(z_i)
          - 'cls_patch': g = z_CLS + (1/N) * sum(z_i)
        """
        feat = self.backbone.forward_features(x)  # shape: [B, num_tokens, 384]
        
        # Token 0 is [CLS] token
        cls_tok = feat[:, 0]
        
        # Tokens from num_prefix_tokens onwards are patch tokens
        num_prefix = getattr(self.backbone, 'num_prefix_tokens', 1)
        patch_tokens = feat[:, num_prefix:]
        patch_mean = patch_tokens.mean(dim=1)

        if self.pooling == "cls":
            g = cls_tok
        elif self.pooling == "patch_mean":
            g = patch_mean
        elif self.pooling in ["cls_patch", "cls+patch"]:
            g = cls_tok + patch_mean
        else:
            raise ValueError(f"Unsupported pooling mode: '{self.pooling}'. Choose from ['cls', 'patch_mean', 'cls_patch'].")
            
        return g

    def forward(self, x):
        g = self.extract_features(x)
        logits = self.classifier(g)
        return logits

def get_data_transforms(model_name):
    """
    Construct training and validation transforms matching Thesis Section 3.3.3:
      - Training: RandomResizedCrop(224, scale=(0.8, 1.0)), RandomHorizontalFlip(0.5),
                  ColorJitter(0.2, 0.2, 0.2, 0.1), ToTensor(), Normalize, RandomErasing(0.2, (0.02, 0.2))
      - Validation: Resize(224, 224), ToTensor(), Normalize
    """
    model_cfg = timm.data.resolve_model_data_config(timm.create_model(model_name))
    mean = model_cfg.get('mean', (0.485, 0.456, 0.406))
    std = model_cfg.get('std', (0.229, 0.224, 0.225))

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), value='random')
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform

def prepare_data(data_dir, model_name, batch_size, num_workers, threshold=8, use_dedup=True):
    """
    Loads dataset from data_dir/train and data_dir/valid, applies deduplication (optional),
    and creates WeightedRandomSampler on train split as per Section 3.3.4.
    """
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Missing 'train' or 'valid' directory inside {data_dir}")

    train_transform, val_transform = get_data_transforms(model_name)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if use_dedup and threshold > 0:
        print("\n[+] Processing Training Split Deduplication")
        filter_dataset_inplace(train_dataset, threshold=threshold)
        
        print("\n[+] Processing Validation Split Deduplication")
        filter_dataset_inplace(val_dataset, threshold=threshold)

    # Section 3.3.4: WeightedRandomSampler with w_c = 1 / n_c
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets)
    print(f"[*] Training Class Counts: {dict(zip(train_dataset.classes, class_counts))}")
    
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = torch.tensor([class_weights[t] for t in targets], dtype=torch.double)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"\n[*] Class mapping ({len(train_dataset.classes)} classes): {train_dataset.classes}")
    train_count = len(train_dataset)
    val_count = len(val_dataset)
    total = train_count + val_count
    
    if total > 0:
        train_ratio = (train_count / total) * 100
        val_ratio = (val_count / total) * 100
        print(f"[*] Final split ratio: {train_ratio:.1f}% Train ({train_count}) | {val_ratio:.1f}% Valid ({val_count}) (Total: {total})")

    return train_loader, val_loader, train_dataset.classes

def train_model(args):
    """
    Main training and validation loop implementing:
      - DINOv3 ViT-Small with partial unfreezing
      - CLS+Patch pooling
      - Focal Loss (gamma=2.0)
      - AdamW with discriminative learning rates (Head: 1e-3, Backbone: 5e-5)
      - CosineAnnealingLR (eta_min=1e-6)
      - Checkpoint selection by validation ROC AUC / Macro-F1
      - Optimal threshold tuning for Pushing class
      - Comprehensive classification metrics and visualization
    """
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device(args.device if "cuda" in args.device else "cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"[*] Compute device: {device}")
    if torch.cuda.is_available():
        print(f"[*] Available GPUs: {torch.cuda.device_count()} ({torch.cuda.get_device_name(0)})")

    if not os.path.exists(args.data_dir):
        print(f"[!] Error: Dataset directory '{args.data_dir}' not found.")
        return

    # Data loaders
    train_loader, val_loader, class_names = prepare_data(
        args.data_dir,
        args.model_name,
        args.batch_size,
        args.workers,
        args.threshold,
        use_dedup=args.dedup
    )

    if len(train_loader.dataset) == 0:
        print("[!] Error: Training dataset is empty.")
        return

    num_classes = len(class_names)
    print(f"[*] Initializing model for {num_classes} classes with '{args.pooling}' pooling...")
    model = LightweightDINO(
        model_name=args.model_name,
        num_classes=num_classes,
        pooling=args.pooling
    )

    if torch.cuda.device_count() > 1 and "cuda" in str(device):
        print(f"[*] Enabling DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)

    # Section 3.6.2: Loss function is Focal Loss (gamma=2.0, alpha=None)
    criterion = FocalLoss(gamma=args.focal_gamma)

    # Section 3.6.2: Optimizer AdamW with discriminative learning rates
    # Head lr = 1e-3, Unfrozen backbone lr = 5e-5, Weight decay = 1e-2
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    backbone_unfrozen = [p for p in base_model.backbone.parameters() if p.requires_grad]
    classifier_params = [p for p in base_model.classifier.parameters() if p.requires_grad]

    print(f"[*] Parameter Groups:")
    print(f"    - Unfrozen Backbone parameters: {len(backbone_unfrozen)} tensors (lr={args.lr_backbone})")
    print(f"    - Classifier Head parameters:   {len(classifier_params)} tensors (lr={args.lr_head})")

    optimizer = optim.AdamW([
        {'params': backbone_unfrozen, 'lr': args.lr_backbone, 'weight_decay': args.weight_decay},
        {'params': classifier_params, 'lr': args.lr_head, 'weight_decay': args.weight_decay}
    ])

    # Section 3.6.2 & Table 3.2: CosineAnnealingLR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.eta_min
    )

    best_metric = -1.0
    history = {'train_loss': [], 'val_acc': [], 'val_macro_f1': [], 'val_roc': []}
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pooling_clean = args.pooling.replace("+", "_").replace("-", "_")
    save_path = output_dir / f"dinov3_lightweight_classifier_v2_{pooling_clean}.pth"
    plot_path = output_dir / f"training_report_v2_{pooling_clean}.png"
    csv_path = output_dir / f"training_stats_v2_{pooling_clean}.csv"

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Val_Accuracy', 'Val_Macro_F1', 'Val_ROC_AUC', 'LR_Backbone', 'LR_Head'])

    print("\n" + "=" * 65)
    print(f" COMMENCING 5-CLASS DINOv3 TRAINING SEQUENCE ({args.epochs} EPOCHS)")
    print(f" Pooling: {args.pooling} | Loss: FocalLoss(gamma={args.focal_gamma}) | Sampler: WeightedRandomSampler")
    print("=" * 65)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{args.epochs:02d}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        correct = 0
        total = 0
        all_targets = []
        all_probs = []
        all_preds = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (preds == labels).sum().item()

                all_targets.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        epoch_acc = 100.0 * correct / max(total, 1)
        all_targets_np = np.array(all_targets)
        all_probs_np = np.array(all_probs)
        all_preds_np = np.array(all_preds)

        epoch_macro_f1 = f1_score(all_targets_np, all_preds_np, average='macro', zero_division=0)

        # Multiclass ROC AUC (OVR)
        try:
            if num_classes == 2:
                epoch_roc = roc_auc_score(all_targets_np, all_probs_np[:, 1])
            else:
                epoch_roc = roc_auc_score(all_targets_np, all_probs_np, multi_class='ovr')
        except ValueError:
            epoch_roc = 0.0

        cur_lr_backbone = optimizer.param_groups[0]['lr']
        cur_lr_head = optimizer.param_groups[1]['lr']

        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(epoch_acc)
        history['val_macro_f1'].append(epoch_macro_f1)
        history['val_roc'].append(epoch_roc)

        print(f"[*] Epoch {epoch+1:02d} -> Loss: {epoch_loss:.4f} | Val Acc: {epoch_acc:.2f}% | Macro-F1: {epoch_macro_f1:.4f} | ROC AUC: {epoch_roc:.4f}")

        with open(csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{epoch_loss:.4f}", f"{epoch_acc:.2f}",
                f"{epoch_macro_f1:.4f}", f"{epoch_roc:.4f}",
                f"{cur_lr_backbone:.6f}", f"{cur_lr_head:.6f}"
            ])

        # Checkpoint persistence based on ROC AUC (Section 3.6.3)
        if epoch_roc >= best_metric:
            best_metric = epoch_roc
            model_state = base_model.state_dict()
            torch.save({
                'model_state_dict': model_state,
                'class_names': class_names,
                'backbone_name': args.model_name,
                'pooling_mode': args.pooling,
                'num_classes': num_classes,
                'val_roc_auc': epoch_roc,
                'val_macro_f1': epoch_macro_f1,
                'epoch': epoch + 1
            }, save_path)
            print(f"[+] Optimal checkpoint captured at epoch {epoch+1:02d} (ROC AUC: {best_metric:.4f})")

    print(f"\n[+] Training complete. Best checkpoint saved to: {save_path}")

    # ---------------------------------------------------------
    # Final Evaluation & Threshold Tuning (Section 3.6.3)
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print(" FINAL EVALUATION & THRESHOLD TUNING ON VALIDATION SPLIT")
    print("=" * 65)

    if save_path.exists():
        checkpoint = torch.load(save_path, map_location=device, weights_only=True)
        base_model.load_state_dict(checkpoint['model_state_dict'])

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

    all_probs_np = np.array(all_probs)
    all_targets_np = np.array(all_targets)

    # Threshold tuning for Pushing bicycle (Section 3.6.3 & Section 4.2.3)
    pushing_idx = [i for i, name in enumerate(class_names) if 'push' in name.lower()]
    best_threshold = 0.5
    if pushing_idx:
        p_idx = pushing_idx[0]
        binary_targets = (all_targets_np == p_idx).astype(int)
        pushing_probs = all_probs_np[:, p_idx]
        
        precisions, recalls, thresholds = precision_recall_curve(binary_targets, pushing_probs)
        f1_scores = np.divide(
            2 * (precisions * recalls),
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        best_f1 = f1_scores[best_idx]
        print(f"[*] Optimal Precision-Recall Threshold for '{class_names[p_idx]}':")
        print(f"    - Threshold: {best_threshold:.4f} | Max Class F1: {best_f1:.4f}")

        # Update saved checkpoint with optimal threshold
        checkpoint_data = torch.load(save_path, weights_only=True)
        checkpoint_data['optimal_threshold'] = best_threshold
        torch.save(checkpoint_data, save_path)
        print(f"[+] Embedded optimal threshold ({best_threshold:.4f}) into checkpoint.")

    # Apply standard argmax or thresholded prediction
    final_preds = np.argmax(all_probs_np, axis=1)
    
    # Classification Report
    report_str = classification_report(all_targets_np, final_preds, target_names=class_names, digits=4)
    print("\n" + report_str)

    report_path = output_dir / f"classification_report_v2_{pooling_clean}.txt"
    with open(report_path, "w", encoding='utf-8') as f:
        f.write(f"5-Class VRU Classification Report (v2_new - {args.pooling})\n")
        f.write("=" * 65 + "\n")
        f.write(report_str)
    print(f"[*] Saved Classification Report: {report_path.name}")

    # Confusion Matrix
    cm = confusion_matrix(all_targets_np, final_preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')
    plt.title(f'Confusion Matrix (v2_new - {args.pooling})', fontweight='bold')
    plt.tight_layout()
    cm_path = output_dir / f"confusion_matrix_v2_{pooling_clean}.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[*] Saved Confusion Matrix: {cm_path.name}")

    # Training Curves Visualization
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], color='tab:red', label='Train Focal Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(history['val_acc'], color='tab:blue', label='Val Accuracy (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(history['val_macro_f1'], color='tab:green', label='Macro-F1')
    plt.plot([r * 1.0 for r in history['val_roc']], color='tab:orange', linestyle='--', label='ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Macro-F1 & ROC AUC')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[*] Saved Training Curves: {plot_path.name}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parents[2] # twostage-vru-detection

    # Default to dataset/dataset_5classes if available, else local labeled_data_v2_new
    primary_data_dir = project_root / "dataset" / "dataset_5classes"
    fallback_data_dir = current_dir / "labeled_data_v2_new"
    default_data_dir = str(primary_data_dir if primary_data_dir.exists() else fallback_data_dir)
    default_output_dir = str(current_dir / "output")

    parser = argparse.ArgumentParser(description="Fine-tune DINOv3 for 5-Class VRU State Classification (Thesis Aligned)")
    parser.add_argument("--data-dir", type=str, default=default_data_dir, help="Root dataset directory containing train/ and valid/ splits")
    parser.add_argument("--model-name", type=str, default="vit_small_patch16_dinov3.lvd1689m", help="Timm backbone model identifier")
    parser.add_argument("--pooling", type=str, default="cls_patch", choices=["cls", "patch_mean", "cls_patch"], help="Feature pooling strategy")
    parser.add_argument("--epochs", type=int, default=15, help="Total training epochs (Table 3.2: 15)")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size (Table 3.2: 32)")
    parser.add_argument("--lr-head", type=float, default=1e-3, help="Optimizer learning rate for MLP head (Table 3.2: 1e-3)")
    parser.add_argument("--lr-backbone", type=float, default=5e-5, help="Optimizer learning rate for unfrozen backbone (Table 3.2: 5e-5)")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW (Table 3.2: 1e-2)")
    parser.add_argument("--eta-min", type=float, default=1e-6, help="Minimum LR for CosineAnnealingLR (Table 3.2: 1e-6)")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Gamma parameter for Focal Loss (Table 3.2: 2.0)")
    parser.add_argument("--workers", type=int, default=4, help="Subprocess count for data loading")
    parser.add_argument("--threshold", type=int, default=8, help="Hamming distance threshold for dHash deduplication")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="Disable dHash deduplication if already deduplicated")
    parser.set_defaults(dedup=True)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (Table 3.2: 42)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Target compute hardware (cuda:0, cpu)")
    parser.add_argument("--output-dir", type=str, default=default_output_dir, help="Directory to save checkpoints, logs, and plots")
    
    args = parser.parse_args()

    # macOS DataLoader workers safety
    import platform
    if platform.system() == "Darwin" and args.workers > 0:
        print("[!] macOS detected: Setting workers to 0 for stability.")
        args.workers = 0

    train_model(args)
