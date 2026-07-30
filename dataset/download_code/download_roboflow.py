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

# Danh sách các tập dữ liệu Roboflow được trích xuất từ dataset/source.txt
# Lưu ý: Chuyển 'universe.roboflow.com' sang 'app.roboflow.com' để tránh Cloudflare Bot Challenge
ROBOFLOW_DATASETS = [
    {
        "name": "stroller_1",
        "category": "Xe đẩy trẻ em (Stroller 1)",
        "url": "https://app.roboflow.com/ds/5qI3xNJ3Qf?key=aXX1sskOrj",
        "folder": "stroller_1"
    },
    {
        "name": "stroller_2",
        "category": "Xe đẩy trẻ em (Stroller 2)",
        "url": "https://app.roboflow.com/ds/4s3nveqS12?key=ytHnpYeoI1",
        "folder": "stroller_2"
    },
    {
        "name": "mixed",
        "category": "Hỗn hợp VRU (Mixed)",
        "url": "https://app.roboflow.com/ds/8VYNSklS7I?key=kjUGGZbBKn",
        "folder": "mixed"
    },
    {
        "name": "wheelchair_1",
        "category": "Xe lăn (Wheelchair 1)",
        "url": "https://app.roboflow.com/ds/BHLLQMynYm?key=02ElkVvivb",
        "folder": "wheelchair_1"
    },
    {
        "name": "wheelchair_2",
        "category": "Xe lăn (Wheelchair 2)",
        "url": "https://app.roboflow.com/ds/SRcgiSanGG?key=vkrG5326on",
        "folder": "wheelchair_2"
    }
]

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, output_path, desc):
    print(f"\n[+] Đang tải [{desc}] từ Roboflow...")
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    )
    
    with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
        total_size = int(response.headers.get('content-length', 0))
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
            buffer_size = 8192
            while True:
                buffer = response.read(buffer_size)
                if not buffer:
                    break
                out_file.write(buffer)
                pbar.update(len(buffer))

def extract_zip(zip_path, extract_to):
    print(f"[+] Đang giải nén vào thư mục: {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[✓] Giải nén thành công {os.path.basename(zip_path)}")

def main():
    parser = argparse.ArgumentParser(description="Tự động tải và chuẩn hóa cấu trúc bộ dữ liệu Roboflow từ dataset/source.txt")
    parser.add_argument("--output_dir", type=str, default="./dataset/Roboflow", help="Thư mục chính lưu các bộ dữ liệu Roboflow")
    parser.add_argument("--keep_zip", action="store_true", help="Giữ lại file .zip sau khi giải nén")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.output_dir)
    os.makedirs(base_dir, exist_ok=True)

    print("==================================================")
    print(" ROBOFLOW DATASET DOWNLOADER FOR VRU")
    print("==================================================")
    print(f" Thư mục lưu trữ: {base_dir}")
    print(f" Số lượng bộ dữ liệu: {len(ROBOFLOW_DATASETS)}")
    print("==================================================")

    for idx, ds in enumerate(ROBOFLOW_DATASETS, 1):
        ds_name = ds["name"]
        ds_url = ds["url"]
        target_folder = os.path.join(base_dir, ds["folder"])
        os.makedirs(target_folder, exist_ok=True)

        zip_path = os.path.join(base_dir, f"{ds_name}.zip")
        print(f"\n({idx}/{len(ROBOFLOW_DATASETS)}) Đang xử lý: {ds_name} - {ds['category']}")

        # Nếu thư mục đích đã có dữ liệu thì bỏ qua
        if len(os.listdir(target_folder)) > 0:
            print(f"[!] Thư mục '{ds['folder']}' đã tồn tại dữ liệu. Bỏ qua tải lại.")
            continue

        try:
            if not os.path.exists(zip_path):
                download_file(ds_url, zip_path, ds_name)
            
            extract_zip(zip_path, target_folder)

            if not args.keep_zip and os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"[+] Đã dọn dẹp file tạm {ds_name}.zip")

        except Exception as e:
            print(f"[❌] Lỗi khi tải hoặc giải nén {ds_name}: {e}")

    print("\n==================================================")
    print(f"[🎉] Tải và sắp xếp toàn bộ Roboflow dataset hoàn tất!")
    print(f"     Địa điểm: {base_dir}")
    print("==================================================")

if __name__ == "__main__":
    main()
