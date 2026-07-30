# DINOv3 Custom Classifier & Tracking Pipeline

This module provides a pipeline for detecting people in videos and classifying their specific states or behaviors (e.g., `walking`, `pushing`, `riding`, or custom user-defined classes). 

It utilizes YOLOv8 for person detection, DINOv3 for feature extraction and classification, and OC-SORT for object tracking. The module is designed to allow training and inference on custom classes without modifying the core system architecture.

---

## Directory Structure

The module is divided into data preparation, training, and inference components.

```text
custom_classes/
├── data_prep/                 # Scripts to extract person crops from videos
│   ├── extract_cyclist.py     # Extracts specific "riding" crops
│   └── extract_walking.py     # Extracts general "walking" crops
├── training/                  # Script to fine-tune the classifier
│   └── train.py               # Model training pipeline
├── inference/                 # Scripts for video processing
│   ├── infer.py               # Command-Line Interface (CLI)
│   └── pipeline.py            # Core tracking and inference logic
├── dataset/                   # Training data directory (User created)
├── output_train/              # Auto-generated: Saved weights & training metrics
└── output_inference/          # Auto-generated: Processed videos and JSON results
```

---

## Pipeline Architecture

The inference pipeline utilizes asynchronous processing to handle video frames.

1. **YOLO Detection**: A YOLOv8 model detects bounding boxes for the `person` class. Bounding boxes smaller than 30 pixels in either dimension are filtered out.
2. **Contextual Padding**: Detected bounding boxes are expanded by a default of `30%` before cropping. This is intended to capture the visual context around the person.
3. **Classification**: The cropped image is passed to a lightweight linear classifier attached to a frozen DINOv3 backbone. 
4. **Confidence Thresholding**: If the highest classification probability is lower than a configured threshold (default: `0.6`), the system assigns a neutral `"person"` label (gray color) rather than a specific behavioral class.
5. **Temporal Tracking**: OC-SORT is used to link bounding boxes across frames. The tracker helps maintain classification consistency across consecutive frames.

---

## Adding Custom Classes

The system determines the classification classes based on the subdirectories present in the `dataset` folder during training.

### Step 1: Prepare the Dataset Structure
Create a `dataset` folder inside `custom_classes`. For each class you intend to recognize, create a folder containing `train` and `valid` subdirectories.

```text
custom_classes/dataset/
├── walking/
│   ├── train/ (Images for training)
│   └── valid/ (Images for validation)
```

> **Note:** The `dataset` folder provided in this repository is a sample dataset intended for demonstration and baseline training purposes (approximately 133 MB total). It contains 3 classes with the following image counts:
> - **`pushing`**: 559 images (484 in `train`, 75 in `valid`)
> - **`riding`**: 2171 images (1370 in `train`, 801 in `valid`)
> - **`walking`**: 1190 images (940 in `train`, 250 in `valid`)
> 
> **Total:** 3,920 images. For real-world production use, you should prepare and train on an even larger and more diverse dataset.

**Data Extraction via Scripts:**
You can use the scripts in `data_prep/` to automatically generate cropped images from raw videos. 
- The scripts run YOLO detection, apply the `30%` padding, and crop the images.
- The data is split into an 80/20 chronological split (train/valid) with a frame gap applied to minimize overlap between the splits.
- *Command example:* `python data_prep/extract_walking.py --input video1.mp4 video2.mp4`

### Step 2: Train the Model
Once the data is structured within the `dataset/` directory, execute the training script.

```bash
python training/train.py --epochs 10 --batch-size 32
```

**Training Script Operations:**
- **Topology Detection:** The script counts the subdirectories in `dataset/` and configures the linear classification head's output size accordingly.
- **Data Deduplication:** It utilizes `dHash` to identify and remove near-duplicate images within the dataset prior to training.
- **Class Balancing:** Cross-Entropy Loss weights are dynamically calculated based on class frequencies to account for imbalanced datasets.
- **Output:** The trained weights and class index mappings are saved to `output_train/dinov3_custom_classifier.pth`.

### Step 3: Run Inference on a Video
Use the trained `.pth` file to process a new video. The script will automatically load the custom class names and generate distinct colors for visualization.

```bash
python inference/infer.py \
    --input ../data/my_test_video.mp4 \
    --output output_inference/my_test_result.mp4 \
    --weights output_train/dinov3_custom_classifier.pth \
    --class-thresh 0.6
```

**Command-Line Arguments:**
- `--input`: Path to the input video file.
- `--output`: Destination path for the output MP4. A corresponding `.json` file with tracking coordinates will be saved in the same directory.
- `--weights`: Path to the trained classifier weights.
- `--class-thresh`: Confidence threshold. Predictions below this score fall back to the `"person"` label (default: `0.60`).
- `--device`: Target compute hardware (e.g., `cuda:0` or `cpu`).

---

## Running via Docker

The custom classes module is compatible with the `dinov3_env` Docker image from the main project. 

> **Hardware Note:** The `--gpus all` flag requires an NVIDIA GPU. If the host machine does not have a GPU, use the "CPU Only" commands below to prevent Docker initialization errors.

**1. Training via Docker:**
From the root repository directory, execute:
```bash
# With NVIDIA GPU
docker run --rm --gpus all --shm-size=8g -v "%cd%:/workspace" dinov3_env python /workspace/custom_classes/training/train.py --epochs 10 --batch-size 32

# Without GPU (CPU Only)
docker run --rm --shm-size=8g -v "%cd%:/workspace" dinov3_env python /workspace/custom_classes/training/train.py --epochs 10 --batch-size 32
```

**2. Inference via Docker:**
From the root repository directory, execute:
```bash
# With NVIDIA GPU
docker run --rm --gpus all --shm-size=8g -v "%cd%:/workspace" dinov3_env python /workspace/custom_classes/inference/infer.py --input /workspace/data/my_test_video.mp4 --output /workspace/custom_classes/output_inference/my_test_result.mp4 --weights /workspace/custom_classes/output_train/dinov3_custom_classifier.pth --class-thresh 0.6

# Without GPU (CPU Only)
docker run --rm --shm-size=8g -v "%cd%:/workspace" dinov3_env python /workspace/custom_classes/inference/infer.py --input /workspace/data/my_test_video.mp4 --output /workspace/custom_classes/output_inference/my_test_result.mp4 --weights /workspace/custom_classes/output_train/dinov3_custom_classifier.pth --class-thresh 0.6
```
*(Note: On macOS/Linux, replace `"%cd%:/workspace"` with `"$(pwd):/workspace"`).*

---

## Outputs & Analytics

After training completes, the `output_train/` directory will contain the following files:
- `dinov3_custom_classifier.pth`: The trained model checkpoint.
- `training_report.png`: Plots showing Loss, Accuracy, and ROC AUC metrics across epochs.
- `confusion_matrix.png`: A heatmap representing the classification performance on the validation set.
- `classification_report.txt`: Text file containing Precision, Recall, and F1-Score metrics per class.
