# DINOv3 Traffic State Classification

## Executive Summary
A robust, multi-threaded computer vision pipeline engineered for real-time detection, tracking, and complex behavioral classification (e.g., Riding vs. Pushing, or custom defined actions). This system integrates YOLOv8 for spatial detection, OC-SORT for temporal tracking, and a trained lightweight classifier on top of a frozen DINOv3 Vision Transformer for high-fidelity state classification.

## Table of Contents
- [Before You Start](#before-you-start)
- [Core Architecture](#core-architecture)
- [Project Repository Structure](#project-repository-structure)
- [Quick Start Guide (Docker Execution)](#quick-start-guide-docker-execution)
- [Manual Deployment & Local Installation](#manual-deployment--local-installation)
- [Model Training & Fine-Tuning](#model-training--fine-tuning)
- [Dynamic Custom Classification Module](#dynamic-custom-classification-module)
- [Experimentation & Diagnostics Toolkit](#experimentation--diagnostics-toolkit)

---

## Before You Start

> [!NOTE]  
> The required model weights are directly included in this repository for a seamless plug-and-play deployment.

### 1. Model Weights
The following files are pre-packaged in `models/weights/`:

| File | Purpose |
|---|---|
| `dinov3_lightweight_classifier_v2_cls_patch.pth` | Multi-token classifier checkpoint combining DINOv3 `[CLS]` token and average-pooled Patch tokens (~83MB) |
| `dinov3_lightweight_classifier_v2_cls.pth` | DINOv3 `[CLS]` token classifier checkpoint (~83MB) |
| `dinov3_lightweight_classifier_v2_patch.pth` | DINOv3 Patch tokens classifier checkpoint (~83MB) |
| `dinov3_lightweight_classifier.pth` | Baseline DINOv3 classifier checkpoint (~83MB) |
| `yolo26n.pt` / `yolo26s.pt` | YOLO detection weights |

### 2. Input Video
Place your input video (`.mp4`, `.avi`, etc.) in the `data/` directory.  
The repository provides 3 test videos for evaluation:
- `data/test01.mp4` (Traffic scene 1 - dense environment)
- `data/test02.mp4` (Traffic scene 2 - medium distance)
- `data/test03.mp4` (Traffic scene 3 - complex behaviors)
- `data/input.mp4` (Default pipeline test input)

---

## Core Architecture

- **High-Precision Tracking:** Utilizes Observation-Centric SORT (OC-SORT) to maintain robust tracking continuity through dense traffic environments and severe occlusions.
- **Vision Transformer Classification:** DINOv3 is utilized as a frozen feature extractor. The core classifier head receives a 384-dimensional DINOv3 feature vector and predicts actions using the following index mapping:
  - `class 0 = pushing`
  - `class 1 = riding`

  Architecture:  
  DINOv3 feature vector (384D)  
  → Linear(384 → 128)  
  → ReLU  
  → Dropout(0.2)  
  → Linear(128 → N classes)  
  → Softmax probabilities
- **Dynamic Threshold Tuning**: Computes the optimal class probability thresholds using a precision-recall curve to maximize F1-scores for imbalanced behaviors. The optimal threshold is stored in the `.pth` checkpoint metadata.
- **Asynchronous Processing:** Achieves high frame rates (FPS) by decoupling video I/O, object detection, and transformer inference into independent parallel worker queues.

---

## Project Repository Structure

```text
DINOv3/
├── main.py                           # Unified CLI entry point for inference
├── run_main_docker.bat               # 1-Click Inference Execution (Windows)
├── run_main_docker.sh                # 1-Click Inference Execution (Mac/Linux)
├── run_train_docker.bat              # 1-Click Training Execution (Windows)
├── run_train_docker.sh               # 1-Click Training Execution (Mac/Linux)
├── requirements.txt                  # Python dependencies
├── README.md                         # Core Documentation
├── data/                             # Input videos and labeled training data
├── models/
│   └── weights/                      # Trained .pth model files
├── output/                           # Directory for rendered videos and JSON logs
├── src/
│   ├── core/                         # Video processing threads
│   ├── inference/                    # DINOv3 Model wrapper and caching logic
│   ├── trackers/                     # OC-SORT implementation
│   └── utils/                        # Mathematical and geometric helpers
├── training/                         # Model fine-tuning logic (v1_old & v2_new)
├── custom_classes/                   # Dynamic training and inference for arbitrary user classes
└── experiments/                      # Research and diagnostic scripts
```

---

## Quick Start Guide 

Containerization via Docker is the recommended deployment strategy. It guarantees environment parity and eliminates the need for manual dependency resolution.

### 1. Prerequisites
- Docker Engine installed and running.
- Model weights are present in `models/weights/`.
- Input video in the `data/` directory.

### 2. Run Inference Pipeline
We provide unified execution scripts that automatically handle environment construction, volume mapping, and hardware acceleration routing.

**For Windows Environments (Auto-detects NVIDIA GPUs):**
```bash
run_main_docker.bat
```
To override default parameters (e.g., specifying a custom input file):
```bash
run_main_docker.bat --input data/custom_video.mp4 --output output/result.mp4
```

**For macOS / Linux Environments (Supports CPU and Apple Silicon MPS):**
```bash
./run_main_docker.sh
```
To override default parameters:
```bash
./run_main_docker.sh --input data/custom_video.mp4 --output output/result.mp4
```

---

## Manual Deployment & Local Installation

For developers intending to modify the source code or run the pipeline natively without Docker:

### 1. Environment Initialization
Requires Python 3.8 or higher.

> [!NOTE]  
> The core deep learning packages (`torch`, `torchvision`) are explicitly included in `requirements.txt` to ensure local manual deployments function correctly outside of Docker.
```bash
git clone <repository_url>
cd DINOv3

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Standard Execution
Initiate the unified CLI entry point:
```bash
python main.py --input data/input.mp4 --output output/tracking_result.mp4 --device cuda:0
```

### CLI Arguments Reference
| Argument | Description | Default |
|---|---|---|
| `--input` | Target video file path | `data/input.mp4` |
| `--output` | Destination path for the rendered video | `output/result_dinov3/tracking_result.mp4` |
| `--device` | Hardware compute target (`cuda:0`, `mps`, `cpu`) | Auto-detected |

---

## Model Training & Fine-Tuning

The repository includes a dedicated pipeline for fine-tuning the DINOv3 classification head on custom datasets. The `v2_new` module incorporates advanced data deduplication (via physical disk pruning and in-memory Difference Hashing) to prevent data leakage from highly correlated video frames, alongside automated Optimal Threshold Tuning to maximize accuracy for rare classes.

### Docker Training Execution
Similar to inference, training can be executed entirely within the automated Docker environment. Output artifacts (weights, CSV telemetry, performance curves) are automatically mapped to the respective `output/` directory of the selected version.

You can easily select the training pipeline version by appending `v1` or `v2` to the execution command. If omitted, the system defaults to the latest `v2` pipeline (dHash enabled).

- **Windows Environments:**
  ```bash
  run_train_docker.bat v2
  ```
- **macOS / Linux Environments:**
  ```bash
  ./run_train_docker.sh v2
  ```

### Manual Training Execution
To execute training natively without Docker:
```bash
cd training/v2_new
python train_v2_new.py --epochs 15 --batch-size 64 --lr 0.0005 --threshold 8
```

---

## Dynamic Custom Classification Module

If you need to rapidly prototype and add **new behavioral classes** without modifying the core system architecture, refer to the **[Custom Classes Module](custom_classes/README.md)**. 

The `custom_classes` module provides an automated sub-system to:
1. Extract crop images directly from video feeds.
2. Dynamically determine network topology (number of classes) based on dataset folder structures.
3. Automatically handle class imbalances and validation splitting.
4. Execute dynamic inference using the newly trained weights.

**Docker Integration:** The custom classes module is fully compatible with the existing `dinov3_env` Docker image. You do not need to build a separate container or install dependencies natively.

Please read `custom_classes/README.md` for detailed integration instructions and Docker execution commands.

---

## Experimentation & Diagnostics Toolkit

The `experiments/` module provides a comprehensive suite of analytical scripts to evaluate model stability and tune hyperparameters.

1. **analyze_tracking_quality.py:** Evaluates JSON tracking telemetry to detect classification oscillation (label flipping) and spatial overlaps.
2. **optimize_crops.py:** Executes parameter sweeps across bounding box padding ratios to maximize transformer confidence scores.
3. **optimize_iou_threshold.py:** Frame-by-frame analysis to determine optimal Non-Maximum Suppression (NMS) intersection-over-union thresholds.
4. **attention_heatmap.py:** Extracts and visualizes the ViT self-attention matrices to debug semantic focal points (e.g., verifying if the model correctly isolates bicycle wheels).
5. **cls_embedding_visualization.py:** Projects `[CLS]` token embeddings into 2D/3D spaces using t-SNE and PCA for clustering validation.
6. **pca_heatmap.py:** Performs unsupervised foreground segmentation via Principal Component Analysis on dense patch embeddings.
7. **test_detection_yolo.py:** A lightweight diagnostic module strictly isolating the YOLOv8 spatial detection logic.
