#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.helpers import calculate_iou

def extract_crops_from_video(
    video_path: Path,
    yolo,
    out_dir: Path,
    padding: float = 0.3,
    target_fps: float = 3.0,
    min_box_size: int = 100,
    iou_thresh: float = 0.1,
    conf_thresh: float = 0.7,
    device=None,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0
    frame_step = max(1, int(round(fps / target_fps)))
    video_stem = video_path.stem

    video_out_dir = out_dir / video_stem
    video_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[>] {video_path.name}: {total_frames} frames @ {fps:.1f} FPS")
    print(f"    Frame step: every {frame_step} frame(s) | Padding: {padding*100:.0f}% | Output: {video_out_dir}")

    saved_count = 0
    frame_idx = 0
    crop_counter = 0

    pbar = tqdm(total=total_frames, desc=f"  {video_stem}", unit="fr")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = frame_idx
        frame_idx += 1
        pbar.update(1)

        if (current_frame % frame_step) != 0:
            continue

        # Detect everything, bicycles with low conf to avoid missing
        results = yolo.predict(frame, classes=[0, 1, 3], conf=0.05, verbose=False, device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        persons = boxes[clss == 0]
        person_confs = confs[clss == 0]
        bikes   = boxes[(clss == 1) | (clss == 3)]

        cyclist_boxes = []
        for p, conf in zip(persons, person_confs):
            if conf < conf_thresh:
                continue

            is_cyclist = False
            for b in bikes:
                # Detect cyclist by checking if person overlaps bike (IoU > 0)
                ix1, iy1 = max(p[0], b[0]), max(p[1], b[1])
                ix2, iy2 = min(p[2], b[2]), min(p[3], b[3])
                if ix1 < ix2 and iy1 < iy2:
                    is_cyclist = True
                    break
            
            # If it IS a cyclist, we want to crop it
            if is_cyclist:
                cyclist_boxes.append(p)

        # NMS
        if len(cyclist_boxes) > 1:
            suppressed = set()
            for i in range(len(cyclist_boxes)):
                if i in suppressed:
                    continue
                for j in range(i + 1, len(cyclist_boxes)):
                    if j not in suppressed and calculate_iou(cyclist_boxes[i], cyclist_boxes[j]) > 0.5:
                        suppressed.add(j)
            cyclist_boxes = [b for idx, b in enumerate(cyclist_boxes) if idx not in suppressed]

        for box in cyclist_boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            if w < min_box_size or h < min_box_size:
                continue

            pad_w = int(w * padding)
            pad_h = int(h * padding)
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(frame.shape[1], x2 + pad_w)
            y2_pad = min(frame.shape[0], y2 + pad_h)

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue

            filename = f"crop_{video_stem}_f{current_frame:06d}_{crop_counter:04d}.jpg"
            save_path = video_out_dir / filename
            cv2.imwrite(str(save_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_counter += 1
            saved_count += 1

    pbar.close()
    cap.release()
    return saved_count

import shutil

def split_dataset_chronologically(temp_dir: Path, out_dir: Path, split_ratio: float = 0.8):
    train_dir = out_dir / "riding" / "train"
    valid_dir = out_dir / "riding" / "valid"
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    print("\n[*] Splitting dataset chronologically (80/20) to prevent data leakage...")
    
    total_train = 0
    total_valid = 0

    for video_dir in temp_dir.iterdir():
        if not video_dir.is_dir():
            continue
        
        crops = list(video_dir.glob("*.jpg"))
        if not crops:
            continue
        
        def get_frame(p):
            parts = p.stem.split('_')
            for part in parts:
                if part.startswith('f') and part[1:].isdigit():
                    return int(part[1:])
            return 0
        
        crops.sort(key=get_frame)
        min_frame = get_frame(crops[0])
        max_frame = get_frame(crops[-1])
        
        split_frame = min_frame + (max_frame - min_frame) * split_ratio
        gap = 90  # ~3 seconds gap to ensure no person crosses the split boundary
        
        train_count = 0
        valid_count = 0
        
        for crop in crops:
            frame = get_frame(crop)
            if frame < split_frame - gap:
                shutil.move(str(crop), str(train_dir / crop.name))
                train_count += 1
            elif frame > split_frame + gap:
                shutil.move(str(crop), str(valid_dir / crop.name))
                valid_count += 1
            else:
                crop.unlink() # Delete discarded crops
            
        print(f"  {video_dir.name}: {train_count} train, {valid_count} valid (Discarded {len(crops) - train_count - valid_count} in gap)")
        total_train += train_count
        total_valid += valid_count
        
        try:
            video_dir.rmdir()
        except:
            pass
        
    try:
        temp_dir.rmdir()
    except:
        pass
        
    print(f"[*] Split complete. Train: {total_train}, Valid: {total_valid}")

def extract_crops_from_image_dir(
    img_dir: Path,
    yolo,
    out_dir: Path,
    padding: float = 0.3,
    min_box_size: int = 100,
    iou_thresh: float = 0.1,
    conf_thresh: float = 0.7,
    device=None,
) -> int:
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(img_dir.rglob(ext)))
    
    if not image_paths:
        print(f"  [!] No images found in: {img_dir}")
        return 0

    dir_stem = f"{img_dir.parent.name}_{img_dir.name}"
    img_out_dir = out_dir / dir_stem
    img_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[>] {img_dir.name}: {len(image_paths)} images")
    print(f"    Padding: {padding*100:.0f}% | Output: {img_out_dir}")

    saved_count = 0
    crop_counter = 0

    pbar = tqdm(total=len(image_paths), desc=f"  {dir_stem}", unit="img")

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        pbar.update(1)
        if frame is None:
            continue

        results = yolo.predict(frame, classes=[0, 1, 3], conf=0.05, verbose=False, device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        persons = boxes[clss == 0]
        person_confs = confs[clss == 0]
        bikes   = boxes[(clss == 1) | (clss == 3)]

        cyclist_boxes = []
        for p, conf in zip(persons, person_confs):
            if conf < conf_thresh:
                continue

            is_cyclist = False
            for b in bikes:
                ix1, iy1 = max(p[0], b[0]), max(p[1], b[1])
                ix2, iy2 = min(p[2], b[2]), min(p[3], b[3])
                if ix1 < ix2 and iy1 < iy2:
                    is_cyclist = True
                    break
            
            if is_cyclist:
                cyclist_boxes.append(p)

        if len(cyclist_boxes) > 1:
            suppressed = set()
            for i in range(len(cyclist_boxes)):
                if i in suppressed:
                    continue
                for j in range(i + 1, len(cyclist_boxes)):
                    if j not in suppressed and calculate_iou(cyclist_boxes[i], cyclist_boxes[j]) > 0.5:
                        suppressed.add(j)
            cyclist_boxes = [b for idx, b in enumerate(cyclist_boxes) if idx not in suppressed]

        for box in cyclist_boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            if w < min_box_size or h < min_box_size:
                continue

            pad_w = int(w * padding)
            pad_h = int(h * padding)
            x1_pad = max(0, x1 - pad_w)
            y1_pad = max(0, y1 - pad_h)
            x2_pad = min(frame.shape[1], x2 + pad_w)
            y2_pad = min(frame.shape[0], y2 + pad_h)

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size == 0:
                continue

            filename = f"crop_{img_path.stem}_{crop_counter:04d}.jpg"
            save_path = img_out_dir / filename
            cv2.imwrite(str(save_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_counter += 1
            saved_count += 1

    pbar.close()
    return saved_count


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save cyclist crop images from video(s).",
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        metavar="VIDEO",
        help="Path(s) to input video file(s).",
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=str,
        default=str(project_root / "custom_classes" / "dataset"),
        help="Root output directory for crops",
    )
    parser.add_argument(
        "--padding", "-p",
        type=float,
        default=0.30,
        help="Bounding box padding ratio as a fraction of box size (default: 0.30)",
    )
    parser.add_argument(
        "--target-fps", "-f",
        type=float,
        default=3.0,
        help="Target frames per second to extract (default: 3.0)",
    )
    parser.add_argument(
        "--conf-thresh", "-c",
        type=float,
        default=0.7,
        help="Confidence threshold for YOLO detections (default: 0.7 for high quality)",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=100,
        help="Minimum width/height in pixels for a crop to be saved (default: 100)",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.1,
        help="IoU threshold for pairing person + bicycle (default: 0.1)",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=None,
        help="Path to YOLO weights file.",
    )
    args = parser.parse_args()

    video_paths = [Path(p) for p in args.input]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from ultralytics import YOLO

    if args.yolo_weights:
        yolo_path = Path(args.yolo_weights)
    else:
        yolo_path = project_root / "models" / "weights" / "yolo26l.pt"
        if not yolo_path.exists():
            yolo_path = project_root / "models" / "weights" / "yolov8n.pt"

    print("=" * 60)
    print(" CYCLIST CROP EXTRACTOR ".center(60, "="))
    print("=" * 60)
    print(f"[*] Device   : {device}")
    print(f"[*] YOLO     : {yolo_path}")
    print(f"[*] Output   : {out_dir}")
    print(f"[*] Padding  : {args.padding * 100:.0f}%")
    print(f"[*] Target FPS: {args.target_fps}")
    print(f"[*] Conf Thresh: {args.conf_thresh}")
    
    valid_paths = [vp for vp in video_paths if vp.exists()]
    if not valid_paths:
        print("\n[!] No valid video files found. Exiting.")
        sys.exit(1)

    print(f"\n[*] Loading YOLO ({yolo_path.name})...")
    yolo = YOLO(str(yolo_path))
    if device == "cuda":
        yolo.to(device)

    total_saved = 0
    per_video_stats = {}
    
    temp_dir = out_dir / "temp_riding"

    for vp in valid_paths:
        if vp.is_dir():
            n = extract_crops_from_image_dir(
                img_dir=vp,
                yolo=yolo,
                out_dir=temp_dir,
                padding=args.padding,
                min_box_size=args.min_box_size,
                iou_thresh=args.iou_thresh,
                conf_thresh=args.conf_thresh,
                device=device,
            )
        else:
            n = extract_crops_from_video(
                video_path=vp,
                yolo=yolo,
                out_dir=temp_dir,
                padding=args.padding,
                target_fps=args.target_fps,
                min_box_size=args.min_box_size,
                iou_thresh=args.iou_thresh,
                conf_thresh=args.conf_thresh,
                device=device,
            )
        per_video_stats[vp.name] = n
        total_saved += n
        print(f"  [+] Saved {n:,} crops from {vp.name}")

    print("\n" + "=" * 60)
    print(" EXTRACTION COMPLETE ".center(60, "="))
    print("=" * 60)
    for vname, count in per_video_stats.items():
        print(f"  {vname:<30s}: {count:>6,} crops")
    print(f"  {'TOTAL':<30s}: {total_saved:>6,} crops")
    
    split_dataset_chronologically(temp_dir, out_dir, split_ratio=0.8)
    
    print(f"\n[*] All processing complete. Data saved in {out_dir.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()
