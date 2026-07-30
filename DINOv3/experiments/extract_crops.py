#!/usr/bin/env python3
"""
Extract and save cyclist crop images from one or more video files.

Detects cyclists (person + bicycle paired via IoU), crops each bounding box
with optional padding, and saves the crops as individual JPEG files.

Output structure:
    data/crops/
    └── <video_stem>/
        ├── crop_<video_stem>_f<frame_idx>_<count>.jpg
        └── ...

Usage:
    python experiments/extract_crops.py --input data/00044.mp4
    python experiments/extract_crops.py --input data/00044.mp4 data/00045.mp4 --padding 0.1
    python experiments/extract_crops.py --input data/00044.mp4 --frame-step 5 --out-dir data/crops
"""

import argparse
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ── Dynamic Path Setup ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils.helpers import calculate_iou


def extract_crops_from_video(
    video_path: Path,
    yolo,
    out_dir: Path,
    padding: float = 0.1,
    frame_step: int = 1,
    min_box_size: int = 30,
    iou_thresh: float = 0.1,
    device=None,
) -> int:
    """
    Process a single video, detect cyclists, and save crop images.

    Returns:
        Total number of crop images saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [!] Cannot open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_stem = video_path.stem

    # Create per-video output subdirectory
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

        # Skip frames based on step
        if (current_frame % frame_step) != 0:
            continue

        # ── YOLO Detection ────────────────────────────────────────────────────
        results = yolo.predict(frame, classes=[0, 1], conf=0.3, verbose=False, device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy()

        persons = boxes[clss == 0]
        bikes   = boxes[clss == 1]

        # ── Pair Person + Bicycle → Cyclist box ───────────────────────────────
        cyclist_boxes = []
        used_bike_indices = set()
        for p in persons:
            for bi, b in enumerate(bikes):
                if bi not in used_bike_indices and calculate_iou(p, b) > iou_thresh:
                    cx1 = min(p[0], b[0])
                    cy1 = min(p[1], b[1])
                    cx2 = max(p[2], b[2])
                    cy2 = max(p[3], b[3])
                    cyclist_boxes.append([cx1, cy1, cx2, cy2])
                    used_bike_indices.add(bi)
                    break

        # ── Greedy NMS to remove duplicate boxes ──────────────────────────────
        if len(cyclist_boxes) > 1:
            suppressed = set()
            for i in range(len(cyclist_boxes)):
                if i in suppressed:
                    continue
                for j in range(i + 1, len(cyclist_boxes)):
                    if j not in suppressed and calculate_iou(cyclist_boxes[i], cyclist_boxes[j]) > 0.5:
                        suppressed.add(j)
            cyclist_boxes = [b for idx, b in enumerate(cyclist_boxes) if idx not in suppressed]

        # ── Crop & Save ───────────────────────────────────────────────────────
        for box in cyclist_boxes:
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1

            # Skip boxes that are too small
            if w < min_box_size or h < min_box_size:
                continue

            # Apply padding
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


def main():
    parser = argparse.ArgumentParser(
        description="Extract and save cyclist crop images from video(s).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        required=True,
        metavar="VIDEO",
        help="Path(s) to input video file(s). e.g. --input data/00044.mp4",
    )
    parser.add_argument(
        "--out-dir", "-o",
        type=str,
        default=str(project_root / "data" / "crops"),
        help="Root output directory for crops (default: data/crops/)",
    )
    parser.add_argument(
        "--padding", "-p",
        type=float,
        default=0.10,
        help="Bounding box padding ratio as a fraction of box size (default: 0.10 = 10%%)",
    )
    parser.add_argument(
        "--frame-step", "-s",
        type=int,
        default=3,
        help="Save a crop every N frames (default: 3, reduces duplicates)",
    )
    parser.add_argument(
        "--min-box-size",
        type=int,
        default=30,
        help="Minimum width/height in pixels for a crop to be saved (default: 30)",
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
        help="Path to YOLO weights file. Auto-resolved if not specified.",
    )
    args = parser.parse_args()

    # ── Resolve paths ─────────────────────────────────────────────────────────
    video_paths = [Path(p) for p in args.input]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load YOLO ─────────────────────────────────────────────────────────────
    from ultralytics import YOLO

    if args.yolo_weights:
        yolo_path = Path(args.yolo_weights)
    else:
        yolo_path = project_root / "models" / "weights" / "yolov8n.pt"
        if not yolo_path.exists():
            yolo_path = project_root / "models" / "weights" / "yolo26n.pt"

    print("=" * 60)
    print(" CYCLIST CROP EXTRACTOR ".center(60, "="))
    print("=" * 60)
    print(f"[*] Device   : {device}")
    print(f"[*] YOLO     : {yolo_path}")
    print(f"[*] Output   : {out_dir}")
    print(f"[*] Padding  : {args.padding * 100:.0f}%")
    print(f"[*] Frame step: every {args.frame_step} frame(s)")
    print(f"[*] Videos to process:")
    valid_paths = []
    for vp in video_paths:
        status = "[OK]" if vp.exists() else "[NOT FOUND]"
        print(f"    {status} {vp}")
        if vp.exists():
            valid_paths.append(vp)

    if not valid_paths:
        print("\n[!] No valid video files found. Exiting.")
        sys.exit(1)

    print(f"\n[*] Loading YOLO ({yolo_path.name})...")
    yolo = YOLO(str(yolo_path))
    if device == "cuda":
        yolo.to(device)

    # ── Process each video ────────────────────────────────────────────────────
    total_saved = 0
    per_video_stats = {}

    for vp in valid_paths:
        n = extract_crops_from_video(
            video_path=vp,
            yolo=yolo,
            out_dir=out_dir,
            padding=args.padding,
            frame_step=args.frame_step,
            min_box_size=args.min_box_size,
            iou_thresh=args.iou_thresh,
            device=device,
        )
        per_video_stats[vp.name] = n
        total_saved += n
        print(f"  [+] Saved {n:,} crops from {vp.name}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" EXTRACTION COMPLETE ".center(60, "="))
    print("=" * 60)
    for vname, count in per_video_stats.items():
        print(f"  {vname:<30s}: {count:>6,} crops")
    print(f"  {'TOTAL':<30s}: {total_saved:>6,} crops")
    print(f"\n[*] All crops saved to: {out_dir.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
