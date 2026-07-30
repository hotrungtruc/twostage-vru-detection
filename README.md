# DINOv3 VRU Detection Project

Dự án phát hiện và phân loại đối tượng giao thông người dùng yếu thế (Vulnerable Road Users - VRU) sử dụng mô hình DINOv3.

## Cấu trúc dự án

```
dinov3-vru/
├── DINOv3/              # Mã nguồn huấn luyện, xử lý và đánh giá DINOv3
├── dataset/             # Mã nguồn tải dữ liệu và các công cụ crop/tiền xử lý VRU
│   ├── download_code/   # Các script tự động tải các tập dữ liệu Citypersons, KITTI, Roboflow
│   ├── crop_vru_dataset.py
│   └── source.txt
├── docs/                # Tài liệu tham khảo và tài liệu dự án
│   ├── images/
│   └── YOLO26/
├── .gitignore
└── README.md
```

## Bắt đầu nhanh

### 1. Cài đặt môi trường
Cài đặt các thư viện cần thiết:
```bash
pip install -r DINOv3/requirements.txt
```

### 2. Tiền xử lý dữ liệu
Sử dụng các script trong thư mục `dataset/download_code/` để tải dataset và `dataset/crop_vru_dataset.py` để xử lý crop VRU.

### 3. Huấn luyện & Đánh giá DINOv3
Xem hướng dẫn chi tiết trong [DINOv3/README.md](DINOv3/README.md).
