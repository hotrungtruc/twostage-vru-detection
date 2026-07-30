import os
import sys
import zipfile
import urllib.request
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    print(f"[+] Đang tải dữ liệu từ: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def main():
    target_dir = "./dataset/Pedestrian_VRU"
    os.makedirs(target_dir, exist_ok=True)
    
    # PennFudanPed Dataset URL (Tập dữ liệu chuẩn cho Người đi bộ)
    dataset_url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = os.path.join(target_dir, "PennFudanPed.zip")
    
    print("==================================================")
    print(" TỰ ĐỘNG TẢI BỘ DỮ LIỆU NGƯỜI ĐI BỘ (VRU)")
    print("==================================================")
    
    if not os.path.exists(zip_path):
        download_url(dataset_url, zip_path)
    else:
        print("[!] File zip đã tồn tại, tiến hành giải nén...")
        
    print("[+] Đang giải nén tập dữ liệu...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    print(f"\n[🎉] THÀNH CÔNG! Dữ liệu đã được lưu tại: {os.path.abspath(target_dir)}")

if __name__ == "__main__":
    main()
