# DINOv3 VRU Detection Project

Dự án phát hiện và phân loại đối tượng giao thông người dùng yếu thế (Vulnerable Road Users - VRU) sử dụng mô hình DINOv3 kết hợp YOLOv8 và OC-SORT.

## Cấu trúc dự án

```
dinov3-vru/
├── DINOv3/              # Mã nguồn huấn luyện, xử lý và đánh giá DINOv3
│   ├── data/            # Thư mục chứa các video thử nghiệm (test01.mp4, test02.mp4, test03.mp4, input.mp4)
│   ├── models/weights/  # Thư mục chứa trọng số mô hình (.pth, .pt)
│   │   ├── dinov3_lightweight_classifier_v2_cls_patch.pth  # Mô hình kết hợp CLS + Patch tokens
│   │   └── ...
│   ├── custom_classes/  # Module huấn luyện & suy luận cho các lớp hành vi tùy chỉnh
│   └── training/        # Pipeline huấn luyện mô hình (v1_old & v2_new)
├── dataset/             # Mã nguồn tải dữ liệu và các công cụ crop/tiền xử lý VRU
│   ├── download_code/   # Các script tự động tải các tập dữ liệu Citypersons, KITTI, Roboflow
│   ├── crop_vru_dataset.py
│   └── source.txt
├── docs/                # Tài liệu tham khảo và tài liệu dự án
├── .gitattributes       # Cấu hình Git LFS quản lý video (.mp4) và trọng số (.pth, .pt)
├── .gitignore
└── README.md
```

## Mô hình & Video thử nghiệm

- **Trọng số huấn luyện (`DINOv3/models/weights/`)**:
  - `dinov3_lightweight_classifier_v2_cls_patch.pth`: Mô hình phân loại nâng cao kết hợp `[CLS]` token và Patch tokens từ DINOv3 Vision Transformer.
  - `yolo26n.pt` / `yolo26s.pt`: Mô hình YOLO phát hiện đối tượng giao thông.
- **Video test (`DINOv3/data/`)**:
  - `test01.mp4`, `test02.mp4`, `test03.mp4`: Các video dữ liệu kiểm thử thực tế.

## Bắt đầu nhanh

### 1. Cài đặt môi trường
Cài đặt các thư viện cần thiết:
```bash
pip install -r DINOv3/requirements.txt
```
*Lưu ý: Dự án sử dụng Git LFS để quản lý video và trọng số mô hình. Hãy đảm bảo đã cài đặt `git lfs`:*
```bash
git lfs install
```

### 2. Thử nghiệm suy luận (Inference)
Chạy suy luận với video test và mô hình `cls_patch`:
```bash
cd DINOv3
python main.py --input data/test01.mp4 --output output/result_test01_cls_patch.mp4 --device cuda:0
```

### 3. Huấn luyện & Đánh giá DINOv3
Xem hướng dẫn chi tiết trong [DINOv3/README.md](DINOv3/README.md) và [DINOv3/training/README.md](DINOv3/training/README.md).

