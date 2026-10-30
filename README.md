# Two-Stage VRU Detection & Behavioral State Classification with DINOv3

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![DINOv3](https://img.shields.io/badge/Backbone-DINOv3%20ViT--S%2F16-darkgreen.svg)](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m)
[![YOLO](https://img.shields.io/badge/Detection-YOLO26s%20%7C%20YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking](https://img.shields.io/badge/Tracking-OC--SORT%20%7C%20SORT-purple.svg)](https://github.com/noahcv/ocsort)
[![Accuracy](https://img.shields.io/badge/Val%20Accuracy-95.67%25-brightgreen.svg)](#4-thuc-nghiem--hieu-nang)
[![ROC AUC](https://img.shields.io/badge/Val%20ROC%20AUC-0.9923-success.svg)](#4-thuc-nghiem--hieu-nang)

Hệ thống thị giác máy tính hai giai đoạn (Two-Stage Pipeline) phát hiện, theo dõi thời gian thực và phân loại trạng thái hành vi chi tiết của Người tham gia Giao thông Dễ bị Tổn thương (Vulnerable Road Users - VRUs) từ luồng camera giám sát đường phố, kết hợp giữa mô hình phát hiện đối tượng YOLO, bộ theo dõi đa mục tiêu OC-SORT và mô hình nền tảng thị giác DINOv3 (Vision Transformer).

---

## 1. Giới thiệu tổng quan (Overview)

Trong các hệ thống giao thông thông minh (ITS), bộ phát hiện đối tượng truyền thống thường chỉ gắn nhãn khái quát ở cấp độ thực thể như `Person` hoặc `Bicycle`. Phân loại thô này không mô tả được trạng thái tương tác động học giữa người và phương tiện — ví dụ: phân biệt người đạp xe (**Riding**) và người dắt xe đạp (**Pushing**), người đi bộ thông thường (**Pedestrian**) với người đẩy xe nôi trẻ em (**Stroller pusher**) hay người sử dụng xe lăn (**Wheelchair user**).

Dự án này triển khai một kiến trúc phân tích hai giai đoạn với mô-đun theo dõi trung gian:

```
[Camera Stream / Video Input]
              │
              ▼
   ┌─────────────────────┐
   │ Giai đoạn 1:        │
   │ YOLO26s / YOLOv8    │ ──> Phát hiện spatial Bounding Boxes (Person & Bicycle)
   └─────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Mô-đun Theo dõi:    │
   │ OC-SORT / SORT      │ ──> Duy trì định danh ID, liên kết trajectory qua thời gian
   └─────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Contextual Padding  │
   │ (+35% BBox Clamp)   │ ──> Nới rộng hộp bao 35% mỗi phía, hợp nhất BBox Person + Bicycle
   └─────────────────────┘
              │
              ▼
   ┌─────────────────────┐
   │ Giai đoạn 2:        │
   │ DINOv3 ViT-S/16     │ ──> Trích xuất đặc trưng đa token (CLS + Mean Patch)
   │ + Lightweight MLP   │ ──> Phân loại 5 trạng thái hành vi VRU với độ chính xác >95%
   └─────────────────────┘
              │
              ▼
[BBox + Tracking ID + Nhãn Trạng thái + Độ tin cậy + Ngưỡng quyết định tối ưu]
```

---

## 2. 5 Lớp Trạng thái VRU Mục tiêu (Target Classes)

| Lớp (Class ID) | Tên nhãn | Định nghĩa & Đặc trưng thị giác | Nguồn dữ liệu |
| :---: | :--- | :--- | :--- |
| **0** | **`pushing`** | Người đang đi bộ và dùng tay dắt hoặc đẩy xe đạp bên cạnh | HICO-DET, Video thực tế, `labeled_data_v2_new` |
| **1** | **`riding`** | Người đang ngồi trên yên xe đạp và điều khiển xe di chuyển | HICO-DET, Cyclist 1–2, `labeled_data_v2_new` |
| **2** | **`stroller`** | Người đang đẩy xe nôi / xe đẩy trẻ em | Roboflow (Stroller 1, 2, 3), PennFudan |
| **3** | **`walking`** | Người đi bộ tự do, không kèm phương tiện hoặc thiết bị hỗ trợ | PennFudanPed, Custom walking dataset |
| **4** | **`wheelchair`** | Người đang ngồi trên xe lăn (tự điều khiển hoặc có người đẩy) | Roboflow (Wheelchair 1, 2, 3) |

---

## 3. Kiến trúc Kỹ thuật Cốt lõi (Core Architecture)

### 3.1 Thích nghi DINOv3 ViT-Small/16 (`lvd1689m`)
- Khởi tạo từ checkpoint tự giám sát chưng cất từ ViT-7B trên tập dữ liệu 1.68 tỷ ảnh (**LVD-1689M**), biểu diễn vector đặc trưng 384 chiều ($d = 384$).
- **Chiến lược mở băng một phần (Partial Unfreezing)**: Đóng băng 10 khối Transformer đầu (`blocks[0..9]`), chỉ cập nhật 2 khối cuối (`blocks[10..11]`) và lớp `LayerNorm` đầu ra, giúp thích ứng với miền ảnh giao thông đô thị mà vẫn giữ vững biểu diễn tổng quát tiền huấn luyện.

### 3.2 Khai thác Đặc trưng Đa Token (`CLS+Patch`)
- Kết hợp thông tin toàn cảnh và chi tiết cục bộ theo công thức:
  $$\mathbf{g}_{\text{cls+patch}} = \mathbf{z}_{\text{CLS}} + \frac{1}{N} \sum_{i=1}^{N} \mathbf{z}_i$$
- Token `[CLS]` đại diện cho hình thái tổng thể của đối tượng, trong khi các token patch ($16 \times 16$) giữ lại chi tiết nhỏ đặc trưng: nan hoa bánh xe, khung sườn xe đạp, tay vịn xe nôi, tư thế chân đạp.

### 3.3 Đầu Phân loại Nhẹ (Lightweight MLP Head)
- Cấu trúc: $\text{LayerNorm}(384) \to \text{Linear}(384, 256) \to \text{GELU} \to \text{Dropout}(0.3) \to \text{Linear}(256, 5)$.
- LayerNorm ổn định phân phối đặc trưng, tầng ẩn 256 chiều mô hình hóa ranh giới phi tuyến, Dropout giảm thiểu nguy cơ quá khớp.

### 3.4 Tối ưu Hóa với Focal Loss & Weighted Sampling
- **Focal Loss** ($\gamma = 2.0$, $\alpha_t = 1$): Tập trung tổn thất vào các mẫu khó, mẫu bị che khuất hoặc có ranh giới thị giác mờ nhạt.
- **WeightedRandomSampler**: Lấy mẫu có hoàn lại với trọng số $w_c = 1 / n_c$ cân bằng phân bổ các lớp trong batch, giải quyết tình trạng mất cân bằng dữ liệu tự nhiên trong giao thông.

### 3.5 Hiệu chỉnh Ngưỡng Quyết định Đa lớp (Threshold Calibration)
- Tính toán ngưỡng xác suất tối ưu $\tau$ cho lớp thiểu số `pushing` dựa trên đường cong Precision-Recall trên tập validation, giúp cân bằng giữa độ chính xác và độ bao phủ, tự động lưu trữ và tải từ checkpoint `.pth`.

---

## 4. Thực Nghiệm & Hiệu Năng (Benchmarks)

### 4.1 Đánh giá Phân loại 5 Lớp trên Tập Validation (1,849 mẫu)

| Lớp trạng thái | Precision | Recall (chuẩn) | Recall (Calibrated $\tau$) | F1-Score (Calibrated) | Số mẫu kiểm tra |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`stroller`** | 0.9876 | 1.0000 | **1.0000** | **0.9938** | 399 |
| **`wheelchair`** | 1.0000 | 0.9665 | **0.9665** | **0.9830** | 239 |
| **`walking`** | 0.9785 | 0.9701 | **0.9493** | **0.9636** | 335 |
| **`riding`** | 0.9616 | 0.9913 | **0.9376** | **0.9494** | 801 |
| **`pushing`** | 0.5000 | 0.2667 | **0.7200** | **0.5902** | 75 |
| **Macro Average** | **0.8855** | **0.8389** | **0.9147** | **`0.8960`** | **1,849** |
| **Accuracy / ROC AUC** | — | — | **Accuracy: `94.81%` - `95.67%`** | **ROC AUC: `0.9923`** | **1,849** |

### 4.2 Hiệu năng Xử lý Video Thời gian Thực (Wall-clock Benchmark Full HD 1080p)

| Video kiểm thử | Độ phân giải & FPS gốc | Tổng số frames | Thời gian xử lý | FPS thực tế | Độ trễ trung bình / frame |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `test01.mp4` | 1920 × 1080 @ 30 FPS | 9,634 frames | 523 s | **18.42 FPS** | 54.29 ms |
| `test02.mp4` | 1920 × 1080 @ 30 FPS | 16,304 frames | 688 s | **23.70 FPS** | 42.20 ms |
| `test03.mp4` | 1920 × 1080 @ 30 FPS | 3,814 frames | 142 s | **26.86 FPS** | 37.23 ms |
| **Tổng / Trung bình gộp** | — | **29,752 frames** | **1,353 s** | **`21.99 FPS`** | **`45.48 ms`** |

---

## 5. Cấu Trúc Thư Mục Dự Án (Repository Structure)

```text
twostage-vru-detection/
├── DINOv3/                                 # Hệ thống mô hình, huấn luyện & suy luận
│   ├── data/                               # Video thử nghiệm (test01.mp4, test02.mp4, test03.mp4, input.mp4)
│   ├── models/weights/                     # Trọng số mô hình chính (.pth, .pt)
│   │   ├── dinov3_lightweight_classifier_v2_cls_patch.pth  # Trọng số 5 lớp DINOv3 tối ưu
│   │   ├── dinov3_lightweight_classifier.pth               # Trọng số baseline 2 lớp
│   │   └── yolo26n.pt / yolo26s.pt                         # Trọng số detector
│   ├── training/                           # Pipeline huấn luyện DINOv3
│   │   ├── v1_old/                         # Phiên bản baseline
│   │   └── v2_new/                         # Pipeline huấn luyện tối ưu
│   │       ├── train_v2_new.py             # Script huấn luyện (Focal Loss, CLS+Patch, CosineLR)
│   │       ├── labeled_data_v2_new/        # Dữ liệu train/valid phân tầng
│   │       └── output/                     # Checkpoints, ma trận nhầm lẫn, biểu đồ hội tụ, telemetry CSV
│   ├── custom_classes/                     # Module mở rộng cho các lớp tùy chỉnh tùy biến
│   ├── src/                                # Mã nguồn hạ tầng và pipeline
│   │   ├── core/                           # Luồng đa luồng (pipeline.py) và cấu trúc bounding box
│   │   ├── inference/                      # Lớp TrafficInferencePipeline kết nối YOLO + Tracker + DINOv3
│   │   ├── trackers/ocsort/                # Thuật toán Observation-Centric SORT
│   │   └── utils/                          # Hàm tính IoU, contextual padding, chuyển đổi tọa độ
│   ├── main.py                             # CLI suy luận đầu cuối trên video
│   └── requirements.txt                    # Danh sách thư viện phụ thuộc
├── dataset/                                # Pipeline quản lý và xử lý dữ liệu
│   ├── dataset_5classes/                   # Bộ dữ liệu 5 lớp hoàn chỉnh (7,067 ảnh crops)
│   │   ├── train/                          # 5,218 ảnh huấn luyện (pushing, riding, stroller, walking, wheelchair)
│   │   ├── valid/                          # 1,849 ảnh kiểm thử
│   │   ├── dataset_metadata.csv            # Chi tiết nguồn gốc, bbox và split của từng ảnh
│   │   └── dataset_summary.json            # Thống kê phân bố lớp
│   ├── download_code/                      # Bộ công cụ tự động tải dataset
│   │   ├── download_citypersons.py         # Tải bộ dữ liệu Người đi bộ (PennFudanPed)
│   │   ├── download_kitti.py               # Tải KITTI Object Detection
│   │   └── download_roboflow.py            # Tải Stroller & Wheelchair datasets từ Roboflow
│   ├── build_5class_dataset.py             # Script tổng hợp, chuẩn hóa BBox, dHash lọc trùng và chia 5 lớp
│   └── crop_vru_dataset.py                 # Pipeline crop ảnh với +35% padding
├── docs/                                   # Tài liệu thiết kế hệ thống
├── .gitattributes                          # Cấu hình Git LFS quản lý weights và video lớn
├── .gitignore
└── README.md                               # Tài liệu hướng dẫn dự án
```

---

## 6. Hướng Dẫn Cài Đặt & Sử Dụng (Quick Start)

### 6.1 Chuẩn bị môi trường

Yêu cầu hệ thống: **Python 3.10+**, **CUDA 12.1+** (khuyến nghị GPU NVIDIA $\ge$ 4GB VRAM).

```bash
# Clone repository
git clone https://github.com/hotrungtruc/twostage-vru-detection.git
cd twostage-vru-detection

# Cài đặt Git LFS để pull trọng số và video lớn
git lfs install
git lfs pull

# Cài đặt dependencies
pip install -r DINOv3/requirements.txt
```

### 6.2 Chuẩn bị Bộ Dữ Liệu 5 Lớp

Dự án cung cấp bộ dữ liệu 5 lớp hoàn chỉnh trong thư mục `dataset/dataset_5classes`. Nếu muốn tự động tải lại từ các nguồn nguyên bản và trích xuất:

```bash
# Tải các tập dữ liệu gốc
python dataset/download_code/download_citypersons.py
python dataset/download_code/download_roboflow.py

# Tổng hợp, khử trùng lặp (dHash) và phân chia train/valid cho 5 lớp
python dataset/build_5class_dataset.py
```

### 6.3 Huấn luyện Mô hình DINOv3 (`train_v2_new.py`)

Huấn luyện mô hình với cấu hình `CLS+Patch`, Focal Loss, `WeightedRandomSampler`, 15 epochs:

```bash
cd DINOv3/training/v2_new

# Chạy huấn luyện trên GPU (mặc định đọc từ dataset/dataset_5classes)
python train_v2_new.py --epochs 15 --batch-size 32 --device cuda:0 --no-dedup
```

**Các tham số tùy chọn:**
- `--pooling`: Cơ chế pooling đặc trưng (`cls_patch`, `patch_mean`, hoặc `cls`). Mặc định: `cls_patch`.
- `--epochs`: Số lượt huấn luyện. Mặc định: `15`.
- `--batch-size`: Kích thước mini-batch. Mặc định: `32`.
- `--lr-head`: Tốc độ học của đầu MLP. Mặc định: `1e-3`.
- `--lr-backbone`: Tốc độ học của 2 khối Transformer cuối. Mặc định: `5e-5`.
- `--focal-gamma`: Hệ số gamma của Focal Loss. Mặc định: `2.0`.
- `--no-dedup`: Bỏ qua bước lọc dHash nếu dữ liệu đã được tiền xử lý trước đó.

*Kết quả sau khi train (checkpoints, metrics report, confusion matrix, loss curves) được tự động lưu trong `DINOv3/training/v2_new/output/`.*

### 6.4 Chạy Suy luận & Theo dõi trên Video Đường Phố (Inference)

Chạy hệ thống đầu cuối (YOLO + OC-SORT + DINOv3):

```bash
cd DINOv3

# Chạy trên video test với GPU
python main.py \
    --input data/test01.mp4 \
    --output output/result_dinov3/result_test01.mp4 \
    --device cuda:0
```

---

## 7. Triển Khai Nhanh Bằng Docker (One-Click Deployment)

Dự án hỗ trợ sẵn script đóng gói Docker bảo đảm tái lập 100% môi trường:

- **Windows**:
  ```cmd
  cd DINOv3
  run_main_docker.bat        # Chạy suy luận video trong container
  run_train_docker.bat v2    # Chạy huấn luyện DINOv3 v2_new trong container
  ```
- **Linux / macOS**:
  ```bash
  cd DINOv3
  chmod +x run_main_docker.sh run_train_docker.sh
  ./run_main_docker.sh
  ./run_train_docker.sh v2
  ```

---

## 8. Bản quyền & Giấy phép (License)

Dự án này được phân phối dưới giấy phép mã nguồn mở [MIT License](LICENSE). Trọng số mô hình nền tảng DINOv3 tuân theo các điều khoản sử dụng nghiên cứu từ Meta AI Research.
