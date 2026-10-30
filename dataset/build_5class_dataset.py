import os
import sys
import re
import csv
import json
import shutil
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def to_long_path(path_str):
    """Ensure Windows MAX_PATH (>260 chars) compatibility."""
    abs_path = os.path.abspath(str(path_str))
    if os.name == 'nt' and not abs_path.startswith('\\\\?\\'):
        return '\\\\?\\' + abs_path
    return abs_path

def get_padded_bbox(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.35):
    """
    Applies symmetric padding around [x1, y1, x2, y2]
    and clamps to image boundary [0, img_w], [0, img_h].
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
    """Checks if boxA and boxB overlap or are close."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    if inter_w * inter_h > 0:
        return True
    cA = ((boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0)
    cB = ((boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0)
    dist = ((cA[0] - cB[0])**2 + (cA[1] - cB[1])**2)**0.5
    diagA = ((boxA[2] - boxA[0])**2 + (boxA[3] - boxA[1])**2)**0.5
    diagB = ((boxB[2] - boxB[0])**2 + (boxB[3] - boxB[1])**2)**0.5
    return (dist / max(diagA, diagB, 1.0)) < 1.2

def compute_dhash(img_cv):
    """Compute 64-bit difference hash."""
    resized = cv2.resize(img_cv, (9, 8))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def is_duplicate_hash(h, hash_dict, threshold=6):
    for _, existing_h in hash_dict.items():
        dist = bin(h ^ existing_h).count('1')
        if dist <= threshold:
            return True
    return False

def main():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "dataset" / "raw_dataset"
    out_5cls_dir = base_dir / "dataset" / "dataset_5classes"
    labeled_v2_dir = base_dir / "DINOv3" / "training" / "v2_new" / "labeled_data_v2_new"
    custom_dataset_dir = base_dir / "DINOv3" / "custom_classes" / "dataset"

    out_5cls_dir.mkdir(parents=True, exist_ok=True)
    metadata_csv_path = out_5cls_dir / "dataset_metadata.csv"
    summary_json_path = out_5cls_dir / "dataset_summary.json"

    classes = ["pushing", "riding", "walking", "stroller", "wheelchair"]
    for split in ["train", "valid"]:
        for cls in classes:
            (out_5cls_dir / split / cls).mkdir(parents=True, exist_ok=True)
            (labeled_v2_dir / split / cls).mkdir(parents=True, exist_ok=True)
            (custom_dataset_dir / cls / split).mkdir(parents=True, exist_ok=True)

    metadata_records = []
    stats = {cls: {"train": 0, "valid": 0} for cls in classes}

    print("=" * 60)
    print(" SYNTHESIZING COMPLETE 5-CLASS VRU DATASET")
    print(" Target Classes: pushing, riding, walking, stroller, wheelchair")
    print(f" Target Output: {out_5cls_dir}")
    print("=" * 60)

    # ----------------------------------------------------
    # 1. Integrate existing classes: riding & pushing
    # ----------------------------------------------------
    print("\n[1/5] Integrating existing 'riding' and 'pushing' classes from labeled_data_v2_new...")
    for cls in ["riding", "pushing"]:
        for split in ["train", "valid"]:
            src_folder = labeled_v2_dir / split / cls
            dst_folder = out_5cls_dir / split / cls
            custom_dst_folder = custom_dataset_dir / cls / split

            if not src_folder.exists():
                print(f"  [!] Missing source {src_folder}")
                continue

            img_files = list(src_folder.glob("*.jpg")) + list(src_folder.glob("*.png"))
            print(f"  -> Copying {cls} [{split}]: {len(img_files)} images...")
            for img_f in tqdm(img_files, desc=f"{cls}_{split}"):
                dst_f = dst_folder / img_f.name
                custom_dst_f = custom_dst_folder / img_f.name
                
                # Copy to dataset_5classes if not exists
                if not dst_f.exists():
                    shutil.copy2(img_f, dst_f)
                # Copy to custom_classes if not exists
                if not custom_dst_f.exists():
                    shutil.copy2(img_f, custom_dst_f)

                stats[cls][split] += 1
                metadata_records.append({
                    "filename": img_f.name,
                    "class": cls,
                    "split": split,
                    "source": "labeled_data_v2_new"
                })

    # ----------------------------------------------------
    # 2. Process Pedestrian / Walking
    # ----------------------------------------------------
    print("\n[2/5] Processing 'walking' (Pedestrian) from PennFudanPed & walking dataset...")
    # Hashes to prevent duplicate crops within each split
    walking_hashes = {"train": {}, "valid": {}}

    # 2a. PennFudanPed
    pf_dir = raw_dir / "Pedestrian_VRU" / "PennFudanPed"
    pf_annot_dir = pf_dir / "Annotation"
    pf_img_dir = pf_dir / "PNGImages"

    if pf_annot_dir.exists() and pf_img_dir.exists():
        annot_files = sorted(list(pf_annot_dir.glob("*.txt")))
        print(f"  -> PennFudanPed has {len(annot_files)} image annotations.")
        bbox_regex = re.compile(r'Bounding box for object \d+ .*?:\s*\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)')
        
        # 80/20 split based on image filename so images don't leak between splits
        random.seed(42)
        shuffled_annots = annot_files.copy()
        random.shuffle(shuffled_annots)
        split_idx = int(len(shuffled_annots) * 0.8)
        train_annots = set(shuffled_annots[:split_idx])

        for afile in tqdm(annot_files, desc="PennFudanPed"):
            split = "train" if afile in train_annots else "valid"
            img_path = pf_img_dir / (afile.stem + ".png")
            if not img_path.exists():
                continue

            try:
                img_pil = Image.open(to_long_path(img_path))
                img_w, img_h = img_pil.size
            except Exception:
                continue

            content = afile.read_text(encoding='utf-8', errors='ignore')
            matches = bbox_regex.findall(content)
            for idx, m in enumerate(matches):
                x1, y1, x2, y2 = map(float, m)
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 30:
                    continue

                px1, py1, px2, py2 = get_padded_bbox(x1, y1, x2, y2, img_w, img_h, padding_ratio=0.35)
                if px2 <= px1 or py2 <= py1:
                    continue

                crop_pil = img_pil.crop((px1, py1, px2, py2))
                crop_cv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
                h = compute_dhash(crop_cv)

                if is_duplicate_hash(h, walking_hashes[split], threshold=6):
                    continue

                crop_name = f"pf_{afile.stem}_obj{idx}_walking.png"
                dst_path = out_5cls_dir / split / "walking" / crop_name
                v2_path = labeled_v2_dir / split / "walking" / crop_name
                custom_path = custom_dataset_dir / "walking" / split / crop_name

                crop_pil.save(to_long_path(dst_path))
                crop_pil.save(to_long_path(v2_path))
                crop_pil.save(to_long_path(custom_path))

                walking_hashes[split][crop_name] = h
                stats["walking"][split] += 1
                metadata_records.append({
                    "filename": crop_name,
                    "class": "walking",
                    "split": split,
                    "source": "PennFudanPed"
                })

    # 2b. Merge existing walking from custom_classes/dataset/walking
    custom_walking_dir = custom_dataset_dir / "walking"
    for split in ["train", "valid"]:
        split_dir = custom_walking_dir / split
        if split_dir.exists():
            files = list(split_dir.glob("*.jpg")) + list(split_dir.glob("*.png"))
            for f in files:
                if f.name.startswith("pf_"):
                    continue # already processed
                dst_path = out_5cls_dir / split / "walking" / f.name
                v2_path = labeled_v2_dir / split / "walking" / f.name
                if not dst_path.exists():
                    shutil.copy2(f, dst_path)
                if not v2_path.exists():
                    shutil.copy2(f, v2_path)
                stats["walking"][split] += 1
                metadata_records.append({
                    "filename": f.name,
                    "class": "walking",
                    "split": split,
                    "source": "custom_classes_walking"
                })

    print(f"  -> Total walking samples: {stats['walking']['train']} train | {stats['walking']['valid']} valid")

    # ----------------------------------------------------
    # 3. Process Wheelchair (Wheelchair User)
    # ----------------------------------------------------
    print("\n[3/5] Processing 'wheelchair' from Roboflow wheelchair_1 & wheelchair_2...")
    wheelchair_hashes = {"train": {}, "valid": {}}
    import yaml

    for ds_name in ["wheelchair_1", "wheelchair_2"]:
        rf_path = raw_dir / "Roboflow" / ds_name
        if not rf_path.exists():
            continue

        yf_list = list(rf_path.rglob("*.yaml"))
        if not yf_list:
            continue
        ydata = yaml.safe_load(yf_list[0].read_text(encoding='utf-8', errors='ignore'))
        names = ydata.get('names', [])
        names_map = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}

        for orig_split in ["train", "valid", "test"]:
            s_dir = rf_path / orig_split
            if not s_dir.exists():
                continue

            target_split = "train" if orig_split == "train" else "valid"
            lbl_dir = s_dir / "labels"
            img_dir = s_dir / "images"

            txt_files = sorted(list(lbl_dir.glob("*.txt")))
            for txt_f in tqdm(txt_files, desc=f"{ds_name}_{orig_split}"):
                img_f = img_dir / (txt_f.stem + ".jpg")
                if not img_f.exists():
                    img_f = img_dir / (txt_f.stem + ".png")
                if not img_f.exists():
                    continue

                try:
                    img_pil = Image.open(to_long_path(img_f))
                    img_w, img_h = img_pil.size
                except Exception:
                    continue

                persons = []
                wheelchairs = []
                try:
                    with open(to_long_path(txt_f), 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5: continue
                            cid = int(parts[0])
                            cname = names_map.get(cid, str(cid)).lower()
                            cname_clean = re.sub(r'^[0-9]+[_\-\s]*', '', cname).strip()

                            xc, yc, w, h = map(float, parts[1:5])
                            x1 = (xc - w / 2.0) * img_w
                            y1 = (yc - h / 2.0) * img_h
                            x2 = (xc + w / 2.0) * img_w
                            y2 = (yc + h / 2.0) * img_h

                            if 'person' in cname_clean or 'human' in cname_clean:
                                persons.append([x1, y1, x2, y2])
                            elif 'wheelchair' in cname_clean:
                                wheelchairs.append([x1, y1, x2, y2])
                except Exception:
                    continue

                # Crop wheelchair boxes (pair with overlapping person when available)
                for w_idx, w_box in enumerate(wheelchairs):
                    best_p = None
                    for p_box in persons:
                        if boxes_intersect_or_near(p_box, w_box):
                            best_p = p_box
                            break
                    if best_p is None and len(persons) == 1:
                        best_p = persons[0]

                    if best_p is not None:
                        ux1 = min(best_p[0], w_box[0])
                        uy1 = min(best_p[1], w_box[1])
                        ux2 = max(best_p[2], w_box[2])
                        uy2 = max(best_p[3], w_box[3])
                    else:
                        ux1, uy1, ux2, uy2 = w_box

                    if (ux2 - ux1) < 25 or (uy2 - uy1) < 25:
                        continue

                    px1, py1, px2, py2 = get_padded_bbox(ux1, uy1, ux2, uy2, img_w, img_h, padding_ratio=0.35)
                    if px2 <= px1 or py2 <= py1:
                        continue

                    crop_pil = img_pil.crop((px1, py1, px2, py2))
                    crop_cv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
                    h = compute_dhash(crop_cv)

                    if is_duplicate_hash(h, wheelchair_hashes[target_split], threshold=6):
                        continue

                    clean_stem = txt_f.stem[:45]
                    crop_name = f"wc_{ds_name}_{clean_stem}_w{w_idx}.png"
                    dst_path = out_5cls_dir / target_split / "wheelchair" / crop_name
                    v2_path = labeled_v2_dir / target_split / "wheelchair" / crop_name
                    custom_path = custom_dataset_dir / "wheelchair" / target_split / crop_name

                    crop_pil.save(to_long_path(dst_path))
                    crop_pil.save(to_long_path(v2_path))
                    crop_pil.save(to_long_path(custom_path))

                    wheelchair_hashes[target_split][crop_name] = h
                    stats["wheelchair"][target_split] += 1
                    metadata_records.append({
                        "filename": crop_name,
                        "class": "wheelchair",
                        "split": target_split,
                        "source": f"Roboflow_{ds_name}"
                    })

    print(f"  -> Total wheelchair samples: {stats['wheelchair']['train']} train | {stats['wheelchair']['valid']} valid")

    # ----------------------------------------------------
    # 4. Process Stroller (Stroller Pusher)
    # ----------------------------------------------------
    print("\n[4/5] Processing 'stroller' from Roboflow stroller_1 & stroller_2...")
    stroller_hashes = {"train": {}, "valid": {}}
    max_stroller_train = 1500  # Cap train to maintain reasonable balance with other classes
    max_stroller_valid = 400

    for ds_name in ["stroller_2", "stroller_1"]: # process stroller_2 first (high quality), then stroller_1
        rf_path = raw_dir / "Roboflow" / ds_name
        if not rf_path.exists():
            continue

        yf_list = list(rf_path.rglob("*.yaml"))
        if not yf_list:
            continue
        ydata = yaml.safe_load(yf_list[0].read_text(encoding='utf-8', errors='ignore'))
        names = ydata.get('names', [])
        names_map = {i: n for i, n in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}

        subdirs = [d.name for d in rf_path.iterdir() if d.is_dir()]
        has_val_test = any(s in subdirs for s in ["valid", "test"])

        for orig_split in subdirs:
            if orig_split not in ["train", "valid", "test"]:
                continue

            s_dir = rf_path / orig_split
            lbl_dir = s_dir / "labels"
            img_dir = s_dir / "images"

            txt_files = sorted(list(lbl_dir.glob("*.txt")))
            
            # If stroller dataset only has train folder (like stroller_2), split deterministically
            random.seed(42)
            if not has_val_test:
                shuffled_txts = txt_files.copy()
                random.shuffle(shuffled_txts)
                val_cutoff = int(len(shuffled_txts) * 0.8)
                train_txt_set = set(shuffled_txts[:val_cutoff])
            else:
                train_txt_set = set(txt_files) if orig_split == "train" else set()

            for txt_f in tqdm(txt_files, desc=f"{ds_name}_{orig_split}"):
                if not has_val_test:
                    target_split = "train" if txt_f in train_txt_set else "valid"
                else:
                    target_split = "train" if orig_split == "train" else "valid"

                if target_split == "train" and stats["stroller"]["train"] >= max_stroller_train:
                    continue
                if target_split == "valid" and stats["stroller"]["valid"] >= max_stroller_valid:
                    continue

                img_f = img_dir / (txt_f.stem + ".jpg")
                if not img_f.exists():
                    img_f = img_dir / (txt_f.stem + ".png")
                if not img_f.exists():
                    continue

                try:
                    img_pil = Image.open(to_long_path(img_f))
                    img_w, img_h = img_pil.size
                except Exception:
                    continue

                humans = []
                strollers = []
                try:
                    with open(to_long_path(txt_f), 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) < 5: continue
                            cid = int(parts[0])
                            cname = names_map.get(cid, str(cid)).lower()
                            cname_clean = re.sub(r'^[0-9]+[_\-\s]*', '', cname).strip()

                            xc, yc, w, h = map(float, parts[1:5])
                            x1 = (xc - w / 2.0) * img_w
                            y1 = (yc - h / 2.0) * img_h
                            x2 = (xc + w / 2.0) * img_w
                            y2 = (yc + h / 2.0) * img_h

                            if 'human' in cname_clean or 'person' in cname_clean:
                                humans.append([x1, y1, x2, y2])
                            elif 'stroller' in cname_clean or 'pram' in cname_clean:
                                strollers.append([x1, y1, x2, y2])
                except Exception:
                    continue

                for s_idx, s_box in enumerate(strollers):
                    best_h = None
                    for h_box in humans:
                        if boxes_intersect_or_near(h_box, s_box):
                            best_h = h_box
                            break
                    if best_h is None and len(humans) == 1:
                        best_h = humans[0]

                    if best_h is not None:
                        ux1 = min(best_h[0], s_box[0])
                        uy1 = min(best_h[1], s_box[1])
                        ux2 = max(best_h[2], s_box[2])
                        uy2 = max(best_h[3], s_box[3])
                    else:
                        ux1, uy1, ux2, uy2 = s_box

                    if (ux2 - ux1) < 25 or (uy2 - uy1) < 25:
                        continue

                    px1, py1, px2, py2 = get_padded_bbox(ux1, uy1, ux2, uy2, img_w, img_h, padding_ratio=0.35)
                    if px2 <= px1 or py2 <= py1:
                        continue

                    crop_pil = img_pil.crop((px1, py1, px2, py2))
                    crop_cv = cv2.cvtColor(np.array(crop_pil), cv2.COLOR_RGB2BGR)
                    h = compute_dhash(crop_cv)

                    if is_duplicate_hash(h, stroller_hashes[target_split], threshold=6):
                        continue

                    clean_stem = txt_f.stem[:45]
                    crop_name = f"st_{ds_name}_{clean_stem}_s{s_idx}.png"
                    dst_path = out_5cls_dir / target_split / "stroller" / crop_name
                    v2_path = labeled_v2_dir / target_split / "stroller" / crop_name
                    custom_path = custom_dataset_dir / "stroller" / target_split / crop_name

                    crop_pil.save(to_long_path(dst_path))
                    crop_pil.save(to_long_path(v2_path))
                    crop_pil.save(to_long_path(custom_path))

                    stroller_hashes[target_split][crop_name] = h
                    stats["stroller"][target_split] += 1
                    metadata_records.append({
                        "filename": crop_name,
                        "class": "stroller",
                        "split": target_split,
                        "source": f"Roboflow_{ds_name}"
                    })

    print(f"  -> Total stroller samples: {stats['stroller']['train']} train | {stats['stroller']['valid']} valid")

    # ----------------------------------------------------
    # 5. Save Metadata & Generate Summary
    # ----------------------------------------------------
    print("\n[5/5] Generating dataset summary and metadata...")
    with open(to_long_path(metadata_csv_path), 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "class", "split", "source"])
        writer.writeheader()
        writer.writerows(metadata_records)

    with open(to_long_path(summary_json_path), 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=4)

    print("\n" + "=" * 60)
    print(" 5-CLASS VRU DATASET GENERATION SUMMARY")
    print("=" * 60)
    total_train = 0
    total_valid = 0
    for cls in classes:
        tr = stats[cls]['train']
        vl = stats[cls]['valid']
        tot = tr + vl
        total_train += tr
        total_valid += vl
        print(f"  • {cls:<15}: {tr:>5} train | {vl:>5} valid | {tot:>5} total")
    print("-" * 60)
    grand_total = total_train + total_valid
    print(f"  TOTAL:            {total_train:>5} train | {total_valid:>5} valid | {grand_total:>5} samples")
    print("=" * 60)
    print(f"[✓] Standalone dataset:      {out_5cls_dir}")
    print(f"[✓] Updated v2_new dataset:  {labeled_v2_dir}")
    print(f"[✓] Updated custom dataset:  {custom_dataset_dir}")
    print(f"[✓] Metadata CSV:            {metadata_csv_path}")

if __name__ == "__main__":
    main()

