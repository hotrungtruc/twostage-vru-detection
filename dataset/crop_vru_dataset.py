import os
import sys
import glob
import re
import csv
import zipfile
import argparse
from pathlib import Path
from PIL import Image

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def get_padded_bbox(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.35):
    """
    Applies symmetric padding around a bounding box [x1, y1, x2, y2]
    and clamps to image dimensions [0, img_w] and [0, img_h].
    """
    w = x2 - x1
    h = y2 - y1
    
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio
    
    px1 = max(0, int(round(x1 - pad_w / 2.0)))
    py1 = max(0, int(round(y1 - pad_h / 2.0)))
    px2 = min(img_w, int(round(x2 + pad_w / 2.0)))
    py2 = min(img_h, int(round(y2 + pad_h / 2.0)))
    
    return px1, py1, px2, py2

def boxes_intersect_or_near(boxA, boxB):
    """
    Checks if boxA [x1, y1, x2, y2] and boxB overlap or are close to each other.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h
    if inter_area > 0:
        return True
    
    # Check center distance normalized by box sizes
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    dist = ((cA[0] - cB[0])**2 + (cA[1] - cB[1])**2)**0.5
    diagA = ((boxA[2] - boxA[0])**2 + (boxA[3] - boxA[1])**2)**0.5
    diagB = ((boxB[2] - boxB[0])**2 + (boxB[3] - boxB[1])**2)**0.5
    max_diag = max(diagA, diagB, 1.0)
    
    return (dist / max_diag) < 1.2

class KITTIProcessor:
    def __init__(self, raw_kitti_dir, output_dir, padding=0.35):
        self.kitti_dir = Path(raw_kitti_dir)
        self.output_dir = Path(output_dir)
        self.padding = padding
        self.label_dir = self.kitti_dir / "training" / "label_2"
        self.image_dir = self.kitti_dir / "training" / "image_2"
        
    def ensure_images_extracted(self):
        if not self.image_dir.exists() or len(list(self.image_dir.glob("*.png"))) == 0:
            zip_path = self.kitti_dir / "data_object_image_2.zip"
            if zip_path.exists():
                print(f"[KITTI] Attempting image_2 extraction from {zip_path}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        zf.extractall(self.kitti_dir)
                    print(f"[KITTI] Zipfile extraction completed.")
                except Exception as e:
                    print(f"[KITTI WARNING] Standard zipfile extract failed ({e}). Checking if images exist or are being extracted externally...")
            else:
                print(f"[KITTI WARNING] Image directory and zip file not found!")
                
    def process(self, metadata_writer):
        self.ensure_images_extracted()
        if not self.label_dir.exists() or not self.image_dir.exists():
            print("[KITTI] Skipping: missing labels or images directory.")
            return 0
        
        count = 0
        label_files = list(self.label_dir.glob("*.txt"))
        print(f"[KITTI] Processing {len(label_files)} annotation files...")
        
        for lfile in label_files:
            img_path = self.image_dir / (lfile.stem + ".png")
            if not img_path.exists():
                img_path = self.image_dir / (lfile.stem + ".jpg")
            if not img_path.exists():
                continue
                
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"[KITTI Error] Cannot open image {img_path}: {e}")
                continue
                
            with open(lfile, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                cls_name = parts[0]
                
                target_cls = None
                if cls_name == "Pedestrian":
                    target_cls = "Pedestrian"
                elif cls_name == "Cyclist":
                    target_cls = "Riding"
                elif cls_name == "Person_sitting":
                    # Explicitly ignored per requirement
                    continue
                else:
                    continue
                
                try:
                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])
                except ValueError:
                    continue
                
                # Check valid bbox bounds
                if x2 <= x1 or y2 <= y1:
                    continue
                
                px1, py1, px2, py2 = get_padded_bbox(x1, y1, x2, y2, img_w, img_h, self.padding)
                if px2 <= px1 or py2 <= py1:
                    continue
                
                crop_img = img.crop((px1, py1, px2, py2))
                out_name = f"kitti_{lfile.stem}_obj{idx}_{target_cls}.png"
                out_path = self.output_dir / target_cls / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                crop_img.save(out_path)
                count += 1
                
                metadata_writer.writerow([
                    out_name, "KITTI", target_cls, str(img_path),
                    f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}",
                    f"{px1},{py1},{px2},{py2}",
                    img_w, img_h
                ])
                
        print(f"[KITTI] Completed. Cropped {count} objects.")
        return count

class PennFudanProcessor:
    def __init__(self, raw_pf_dir, output_dir, padding=0.35):
        self.pf_dir = Path(raw_pf_dir)
        self.output_dir = Path(output_dir)
        self.padding = padding
        
    def process(self, metadata_writer):
        annot_dir = self.pf_dir / "Annotation"
        img_dir = self.pf_dir / "PNGImages"
        
        if not annot_dir.exists() or not img_dir.exists():
            print("[PennFudanPed] Skipping: missing directory.")
            return 0
            
        annot_files = list(annot_dir.glob("*.txt"))
        print(f"[PennFudanPed] Processing {len(annot_files)} annotation files...")
        count = 0
        
        bbox_regex = re.compile(r'Bounding box for object \d+ .*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)')
        
        for afile in annot_files:
            img_path = img_dir / (afile.stem + ".png")
            if not img_path.exists():
                continue
                
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"[PennFudan Error] Cannot open {img_path}: {e}")
                continue
                
            content = afile.read_text(encoding='utf-8', errors='ignore')
            matches = bbox_regex.findall(content)
            
            for idx, m in enumerate(matches):
                x1, y1, x2, y2 = map(float, m)
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                target_cls = "Pedestrian"
                px1, py1, px2, py2 = get_padded_bbox(x1, y1, x2, y2, img_w, img_h, self.padding)
                if px2 <= px1 or py2 <= py1:
                    continue
                    
                crop_img = img.crop((px1, py1, px2, py2))
                out_name = f"pennfudan_{afile.stem}_obj{idx}_{target_cls}.png"
                out_path = self.output_dir / target_cls / out_name
                out_path.parent.mkdir(parents=True, exist_ok=True)
                
                crop_img.save(out_path)
                count += 1
                
                metadata_writer.writerow([
                    out_name, "PennFudanPed", target_cls, str(img_path),
                    f"{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}",
                    f"{px1},{py1},{px2},{py2}",
                    img_w, img_h
                ])
                
        print(f"[PennFudanPed] Completed. Cropped {count} objects.")
        return count

class RoboflowProcessor:
    def __init__(self, raw_rf_dir, output_dir, padding=0.35):
        self.rf_dir = Path(raw_rf_dir)
        self.output_dir = Path(output_dir)
        self.padding = padding
        
    def process(self, metadata_writer):
        if not self.rf_dir.exists():
            print("[Roboflow] Skipping: directory does not exist.")
            return 0
            
        count = 0
        subdirs = [d for d in self.rf_dir.iterdir() if d.is_dir()]
        
        for sdir in subdirs:
            print(f"[Roboflow] Processing sub-dataset: {sdir.name}...")
            
            # Map names from data.yaml if available
            names_map = {}
            yaml_files = list(sdir.rglob("*.yaml"))
            if yaml_files:
                import yaml
                try:
                    ydata = yaml.safe_load(yaml_files[0].read_text(encoding='utf-8'))
                    names = ydata.get('names', [])
                    if isinstance(names, list):
                        names_map = {idx: name for idx, name in enumerate(names)}
                    elif isinstance(names, dict):
                        names_map = {int(k): v for k, v in names.items()}
                except Exception as e:
                    print(f"  [YAML Warning] Could not parse {yaml_files[0]}: {e}")
            
            # Fallback default names maps if yaml was missing/empty
            if not names_map:
                if "wheelchair_1" in sdir.name:
                    names_map = {0: "person", 1: "wheelchair"}
                elif "wheelchair_2" in sdir.name:
                    names_map = {0: "wheelchair"}
                elif "stroller" in sdir.name:
                    names_map = {0: "bicycle", 1: "human", 2: "stroller", 3: "suitcase", 4: "wheelchair"}
            
            # Single-pass indexing of images and label txt files for ultra-fast processing
            img_map = {}
            label_files = []
            valid_exts = {'.jpg', '.jpeg', '.png', '.JPG', '.PNG'}
            
            for root, dirs, files in os.walk(sdir):
                for fname in files:
                    ext = os.path.splitext(fname)[1]
                    fpath = Path(root) / fname
                    if ext in valid_exts:
                        img_map[fpath.stem] = fpath
                    elif ext == '.txt' and "README" not in fname and "data" not in fname:
                        label_files.append(fpath)
            
            for lfile in label_files:
                img_path = img_map.get(lfile.stem)
                if not img_path or not img_path.exists():
                    continue
                    
                try:
                    img = Image.open(img_path)
                    img_w, img_h = img.size
                except Exception as e:
                    print(f"  [Roboflow Error] Cannot open {img_path}: {e}")
                    continue
                    
                # Read YOLO format annotations
                persons = []
                objects = []  # (class_name, [x1, y1, x2, y2])
                
                person_aliases = {'person', 'human', 'pedestrian', 'children', 'child', 'people'}
                stroller_aliases = {'stroller', 'pram', 'prams', 'pushchair', 'strollers'}
                wheelchair_aliases = {'wheelchair', 'wheelchairs'}
                bicycle_aliases = {'bicycle', 'cyclist', 'bike', 'bicycles'}
                
                with open(lfile, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cls_name = names_map.get(cls_id, f"class_{cls_id}").lower()
                        
                        xc, yc, w, h = map(float, parts[1:5])
                        x1 = (xc - w / 2.0) * img_w
                        y1 = (yc - h / 2.0) * img_h
                        x2 = (xc + w / 2.0) * img_w
                        y2 = (yc + h / 2.0) * img_h
                        
                        if cls_name in person_aliases:
                            persons.append([x1, y1, x2, y2])
                        elif cls_name in stroller_aliases:
                            objects.append(('stroller', [x1, y1, x2, y2]))
                        elif cls_name in wheelchair_aliases:
                            objects.append(('wheelchair', [x1, y1, x2, y2]))
                        elif cls_name in bicycle_aliases:
                            objects.append(('bicycle', [x1, y1, x2, y2]))
                
                used_person_indices = set()
                
                # Pair objects (wheelchair, stroller, bicycle) with persons
                for obj_cls, obj_box in objects:
                    best_person_idx = None
                    
                    # Search for overlapping or close person
                    for p_idx, p_box in enumerate(persons):
                        if boxes_intersect_or_near(p_box, obj_box):
                            best_person_idx = p_idx
                            break
                    
                    # If there's only 1 person in the image and no exact overlap, match them
                    if best_person_idx is None and len(persons) == 1:
                        best_person_idx = 0
                        
                    if best_person_idx is not None:
                        used_person_indices.add(best_person_idx)
                        p_box = persons[best_person_idx]
                        
                        # Union bbox
                        ux1 = min(p_box[0], obj_box[0])
                        uy1 = min(p_box[1], obj_box[1])
                        ux2 = max(p_box[2], obj_box[2])
                        uy2 = max(p_box[3], obj_box[3])
                        
                        target_cls = None
                        if obj_cls == 'wheelchair':
                            target_cls = "Wheelchair_user"
                        elif obj_cls == 'stroller':
                            target_cls = "Stroller_pusher"
                        elif obj_cls == 'bicycle':
                            target_cls = "Riding"
                            
                        if target_cls:
                            px1, py1, px2, py2 = get_padded_bbox(ux1, uy1, ux2, uy2, img_w, img_h, self.padding)
                            if px2 > px1 and py2 > py1:
                                crop_img = img.crop((px1, py1, px2, py2))
                                stem_clean = lfile.stem[:60]
                                out_name = f"rf_{sdir.name}_{stem_clean}_{obj_cls}_{target_cls}.png"
                                out_path = (self.output_dir / target_cls / out_name).resolve()
                                out_path.parent.mkdir(parents=True, exist_ok=True)
                                
                                # Use long path prefix for Windows compatibility
                                save_path = '\\\\?\\' + str(out_path)
                                crop_img.save(save_path)
                                count += 1
                                
                                metadata_writer.writerow([
                                    out_name, f"Roboflow_{sdir.name}", target_cls, str(img_path),
                                    f"{ux1:.1f},{uy1:.1f},{ux2:.1f},{uy2:.1f}",
                                    f"{px1},{py1},{px2},{py2}",
                                    img_w, img_h
                                ])
                    else:
                        # Isolated wheelchair / stroller / bicycle without person -> SKIP as per requirement
                        pass
                
                # Process remaining standalone persons as Pedestrian
                for p_idx, p_box in enumerate(persons):
                    if p_idx not in used_person_indices:
                        target_cls = "Pedestrian"
                        px1, py1, px2, py2 = get_padded_bbox(p_box[0], p_box[1], p_box[2], p_box[3], img_w, img_h, self.padding)
                        if px2 > px1 and py2 > py1:
                            crop_img = img.crop((px1, py1, px2, py2))
                            stem_clean = lfile.stem[:60]
                            out_name = f"rf_{sdir.name}_{stem_clean}_p{p_idx}_{target_cls}.png"
                            out_path = (self.output_dir / target_cls / out_name).resolve()
                            out_path.parent.mkdir(parents=True, exist_ok=True)
                            
                            save_path = '\\\\?\\' + str(out_path)
                            crop_img.save(save_path)
                            count += 1
                            
                            metadata_writer.writerow([
                                out_name, f"Roboflow_{sdir.name}", target_cls, str(img_path),
                                f"{p_box[0]:.1f},{p_box[1]:.1f},{p_box[2]:.1f},{p_box[3]:.1f}",
                                f"{px1},{py1},{px2},{py2}",
                                img_w, img_h
                            ])
                            
        print(f"[Roboflow] Completed. Cropped {count} objects.")
        return count

def main():
    parser = argparse.ArgumentParser(description="Crop VRU dataset images with +35% padding for target VRU categories.")
    parser.add_argument("--raw_dir", type=str, default="./dataset/raw_dataset", help="Path to raw_dataset directory")
    parser.add_argument("--output_dir", type=str, default="./dataset/preprocessed_dataset/crops", help="Output crops directory")
    parser.add_argument("--padding", type=float, default=0.35, help="Padding ratio (default: 0.35)")
    args = parser.parse_args()
    
    raw_dir = Path(args.raw_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir.parent / "crops_metadata.csv"
    
    print("==================================================")
    print(" VRU DATASET IMAGE CROPPING PIPELINE")
    print("==================================================")
    print(f" Raw Dataset Path: {raw_dir}")
    print(f" Output Crops Path: {output_dir}")
    print(f" Padding Ratio: +{args.padding * 100:.0f}%")
    print(" Target Classes: Pedestrian, Wheelchair_user, Stroller_pusher, Riding")
    print("==================================================")
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "crop_filename", "source_dataset", "target_category",
            "source_image_path", "original_bbox", "padded_bbox",
            "img_width", "img_height"
        ])
        
        total_crops = 0
        
        # 1. Process KITTI
        kitti_processor = KITTIProcessor(raw_dir / "KITTI", output_dir, args.padding)
        total_crops += kitti_processor.process(writer)
        
        # 2. Process PennFudanPed
        pf_processor = PennFudanProcessor(raw_dir / "Pedestrian_VRU" / "PennFudanPed", output_dir, args.padding)
        total_crops += pf_processor.process(writer)
        
        # 3. Process Roboflow
        rf_processor = RoboflowProcessor(raw_dir / "Roboflow", output_dir, args.padding)
        total_crops += rf_processor.process(writer)
        
    print("\n==================================================")
    print(f"[SUCCESS] SUMMARY: Successfully generated {total_crops} cropped images!")
    print(f"          Cropped Images: {output_dir}")
    print(f"          Metadata CSV:   {csv_path}")
    print("==================================================")

if __name__ == "__main__":
    main()
