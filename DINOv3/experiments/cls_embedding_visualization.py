"""
t-SNE Visualization for DINOv3 Classification Embeddings
Visualizes embeddings at 2 stages:
1. 384D features from DINOv3 backbone (before MLP)
2. 128D features from first Linear layer (after MLP projection)
"""

import torch
import torch.nn as nn
import numpy as np
import timm
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# ── Dynamic Path Setup ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[0]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.inference.inference_pipeline import LightweightDINOInference

# ==========================================
# 1. MODEL DEFINITION WITH HOOKS
# ==========================================
class LightweightDINOWithHooks(nn.Module):
    def __init__(self, model_name="vit_small_patch16_dinov3.lvd1689m", num_classes=2):
        super(LightweightDINOWithHooks, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),  # 384D -> 128D
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        # Storage for intermediate features
        self.features_384 = None
        self.features_128 = None

        # Register hook for 128D features (after Linear(384, 128))
        self.classifier[0].register_forward_hook(self._hook_linear_384_128)

    def _hook_linear_384_128(self, module, input, output):
        """Capture features after Linear(384, 128) but before ReLU"""
        self.features_128 = output.detach()

    def forward(self, x):
        # Extract 384D features from backbone
        with torch.no_grad():
            features_384 = self.backbone(x)  # [Batch, 384]

        self.features_384 = features_384.detach()

        # Pass through classifier (hook will capture 128D)
        logits = self.classifier(features_384)

        return logits, features_384, self.features_128


# ==========================================
# 2. CUSTOM DATASET FOR LABELED DATA
# ==========================================
class LabeledCyclistDataset(Dataset):
    def __init__(self, data_root, transform=None):
        self.data_root = Path(data_root)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.label_map = {}

        # Load images from subdirectories (cluster0, cluster1, etc.)
        label_idx = 0
        for class_folder in sorted(self.data_root.iterdir()):
            if class_folder.is_dir():
                self.label_map[class_folder.name] = label_idx

                for img_file in class_folder.glob("*.jpg"):
                    if img_file.exists():
                        self.samples.append(img_file)
                        self.labels.append(label_idx)

                for img_file in class_folder.glob("*.png"):
                    if img_file.exists():
                        self.samples.append(img_file)
                        self.labels.append(label_idx)

                for img_file in class_folder.glob("*.jpeg"):
                    if img_file.exists():
                        self.samples.append(img_file)
                        self.labels.append(label_idx)

                label_idx += 1

        print(f"Loaded {len(self.samples)} images from {len(self.label_map)} classes")
        print(f"Label map: {self.label_map}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy = Image.new("RGB", (224, 224))
            if self.transform:
                dummy = self.transform(dummy)
            return dummy, label


# ==========================================
# 3. FEATURE EXTRACTION
# ==========================================
def extract_embeddings(model, data_loader, device):
    """Extract 384D and 128D embeddings from all images"""
    embeddings_384 = []
    embeddings_128 = []
    labels_list = []

    model.eval()

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            if device.type == "cuda":
                images = images.half()

            logits, feat_384, feat_128 = model(images)

            embeddings_384.append(feat_384.cpu().numpy())
            embeddings_128.append(feat_128.cpu().numpy())
            labels_list.extend(labels.numpy())

    embeddings_384 = np.vstack(embeddings_384)
    embeddings_128 = np.vstack(embeddings_128)
    labels_array = np.array(labels_list)

    return embeddings_384, embeddings_128, labels_array


# ==========================================
# 4. t-SNE VISUALIZATION
# ==========================================
def visualize_tsne(embeddings, labels, title, output_path, class_names=None):
    print(f"\n🔧 Applying t-SNE to {embeddings.shape}...")

    tsne = TSNE(
        n_components=2, random_state=42, perplexity=30, max_iter=1000, verbose=1
    )
    embeddings_2d = tsne.fit_transform(embeddings)

    if class_names is None:
        unique_labels = np.unique(labels)
        class_names = [f"Class {i}" for i in unique_labels]

    plt.figure(figsize=(12, 8))

    colors = [
        "#FF6B6B",
        "#4ECDC4",
        "#45B7D1",
        "#FFA07A",
    ]
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=colors[i % len(colors)],
            label=class_names[i],
            alpha=0.6,
            s=50,
            edgecolors="black",
            linewidth=0.5,
        )

    plt.xlabel("t-SNE Component 1", fontsize=12, fontweight="bold")
    plt.ylabel("t-SNE Component 2", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")
    plt.legend(fontsize=11, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved to {output_path}")
    plt.close()


# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def main():
    labeled_data_dir = project_root / "data" / "labeled_data"
    model_weights = project_root / "models" / "weights" / "dinov3_lightweight_classifier.pth"
    output_dir = project_root / "output" / "cls_embedding"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📂 Labeled data dir: {labeled_data_dir}")
    print(f"📂 Model weights: {model_weights}")
    print(f"📂 Output dir: {output_dir}")

    if not model_weights.exists():
        print(f"❌ Model weights not found at {model_weights}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    print("\n📦 Loading model...")
    model = LightweightDINOWithHooks(num_classes=2).to(device)

    checkpoint = torch.load(model_weights, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    model.load_state_dict(state_dict, strict=False)

    if device.type == "cuda":
        model.half()

    print("✅ Model loaded")

    print("\n📊 Preparing data...")
    model_cfg = timm.data.resolve_model_data_config(
        timm.create_model("vit_small_patch16_dinov3.lvd1689m")
    )
    transform = timm.data.create_transform(**model_cfg, is_training=False)

    dataset = LabeledCyclistDataset(labeled_data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    class_names = ["Riding", "Pushing"]

    print("\n🔍 Extracting embeddings...")
    embeddings_384, embeddings_128, labels = extract_embeddings(
        model, data_loader, device
    )

    print(f"\n📈 Embeddings shape:")
    print(f"   384D: {embeddings_384.shape}")
    print(f"   128D: {embeddings_128.shape}")
    print(f"   Labels: {labels.shape}")
    print(f"   Class distribution: {np.bincount(labels)}")

    print("\n📊 Creating t-SNE visualizations...")

    output_path_384 = output_dir / "tsne_384d_before_mlp.png"
    visualize_tsne(
        embeddings_384,
        labels,
        "t-SNE Visualization: 384D Features (Before MLP)\nDINOv3 Backbone Output",
        str(output_path_384),
        class_names=class_names,
    )

    output_path_128 = output_dir / "tsne_128d_after_mlp.png"
    visualize_tsne(
        embeddings_128,
        labels,
        "t-SNE Visualization: 128D Features (After MLP Projection)\nAfter Linear(384→128) + ReLU",
        str(output_path_128),
        class_names=class_names,
    )

    print(f"\n✨ Visualization complete!")
    print(f"📸 Output images:")
    print(f"   - {output_path_384}")
    print(f"   - {output_path_128}")

    stats_path = output_dir / "embedding_statistics.txt"
    with open(stats_path, "w") as f:
        f.write("DINOv3 Classification Embedding Statistics\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Dataset size: {len(dataset)}\n")
        f.write(
            f"Class distribution: {dict(zip(class_names, np.bincount(labels)))}\n\n"
        )

        f.write("384D Features (Before MLP):\n")
        f.write(f"  Shape: {embeddings_384.shape}\n")
        f.write(f"  Mean: {embeddings_384.mean(axis=0)[:5]}...\n")
        f.write(f"  Std: {embeddings_384.std(axis=0)[:5]}...\n")
        f.write(f"  Min: {embeddings_384.min():.4f}\n")
        f.write(f"  Max: {embeddings_384.max():.4f}\n\n")

        f.write("128D Features (After MLP):\n")
        f.write(f"  Shape: {embeddings_128.shape}\n")
        f.write(f"  Mean: {embeddings_128.mean(axis=0)[:5]}...\n")
        f.write(f"  Std: {embeddings_128.std(axis=0)[:5]}...\n")
        f.write(f"  Min: {embeddings_128.min():.4f}\n")
        f.write(f"  Max: {embeddings_128.max():.4f}\n")

    print(f"📄 Statistics saved to {stats_path}")


if __name__ == "__main__":
    main()
