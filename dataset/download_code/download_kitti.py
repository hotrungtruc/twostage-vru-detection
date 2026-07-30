import os
import sys
import zipfile
import urllib.request
import argparse
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Direct AWS S3 URLs cho bộ dữ liệu KITTI 2D Object Detection
KITTI_LABEL_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"
KITTI_IMAGE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path):
    filename = os.path.basename(output_path)
    print(f"\n[+] Đang tải {filename} từ: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"[+] Đang giải nén {os.path.basename(zip_path)} vào {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[✓] Giải nén hoàn tất!")

def parse_kitti_labels(label_dir):
    """
    Thống kê số lượng các lớp VRU (Pedestrian, Cyclist, Person_sitting) trong KITTI
    """
    stats = {
        "Pedestrian": 0,
        "Cyclist": 0,
        "Person_sitting": 0,
        "Car": 0,
        "DontCare": 0,
        "Other": 0
    }
    
    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    print(f"\n[+] Đang phân tích {len(label_files)} tệp nhãn KITTI...")
    
    for f_name in label_files:
        f_path = os.path.join(label_dir, f_name)
        with open(f_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_name = parts[0]
                if cls_name in stats:
                    stats[cls_name] += 1
                else:
                    stats["Other"] += 1
                    
    print("\n--------------------------------------------------")
    print(" THỐNG KÊ ĐỐI TƯỢNG TRONG KITTI DATASET:")
    print("--------------------------------------------------")
    for cat, count in stats.items():
        print(f"  • {cat:<15}: {count:>6} đối tượng")
    print("--------------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser(description="Tự động tải bộ dữ liệu KITTI Object Detection Dataset (2D VRUs)")
    parser.add_argument("--output_dir", type=str, default="./dataset/KITTI", help="Thư mục lưu dataset")
    parser.add_argument("--download_images", action="store_true", help="Bật cờ này để tải cả gói Ảnh (.zip ~12GB)")
    parser.add_argument("--download_labels_only", action="store_true", help="Chỉ tải gói Nhãn (.zip ~5MB)")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("==================================================")
    print(" KITTI OBJECT DETECTION DATASET DOWNLOADER")
    print("==================================================")
    print(f" Thư mục đích: {os.path.abspath(args.output_dir)}")
    print("==================================================")
    
    # 1. Tải gói nhãn Label_2 (~5MB)
    label_zip = os.path.join(args.output_dir, "data_object_label_2.zip")
    if not os.path.exists(label_zip):
        download_file(KITTI_LABEL_URL, label_zip)
    else:
        print("[!] File nhãn data_object_label_2.zip đã tồn tại sẵn.")
        
    label_extract_dir = os.path.join(args.output_dir, "training", "label_2")
    if not os.path.exists(label_extract_dir):
        extract_zip(label_zip, os.path.join(args.output_dir))
        
    # Phân tích nhãn VRU
    if os.path.exists(label_extract_dir):
        parse_kitti_labels(label_extract_dir)
        
    # 2. Tải gói ảnh Image_2 (~12GB) nếu được yêu cầu
    if args.download_images and not args.download_labels_only:
        image_zip = os.path.join(args.output_dir, "data_object_image_2.zip")
        if not os.path.exists(image_zip):
            download_file(KITTI_IMAGE_URL, image_zip)
        else:
            print("[!] File ảnh data_object_image_2.zip đã tồn tại sẵn.")
            
        extract_zip(image_zip, os.path.join(args.output_dir))
    elif not args.download_images:
        print("[💡] Mẹo: Bạn có thể chạy kèm cờ '--download_images' để tải trọn bộ ảnh 12GB.")

    print(f"\n[🎉] THÀNH CÔNG! Dữ liệu KITTI đã được chuẩn bị tại: {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main()
