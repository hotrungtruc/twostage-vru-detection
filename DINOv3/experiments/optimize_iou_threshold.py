#!/usr/bin/env python3
"""
optimize_iou_threshold.py
=========================
Runs YOLOv8 object detection (Person, Bicycle) on videos: data/00038.mp4 and data/0422.test.mp4.
Displays overlapping Person & Bicycle boxes (IoU > 0) with connecting lines, highlighted 
intersection regions, and calculated IoU values. Saves the annotated videos.
"""

import os
import sys
import cv2
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ── Dynamic Path Setup ─────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.utils.helpers import calculate_iou

def draw_label(img, box, label, score, box_color, text_color=(255, 255, 255)):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    label_text = f" {label} {score:.2f} "
    (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    cv2.rectangle(img, (x1, y1 - lbl_h - 6), (x1 + lbl_w, y1), box_color, -1)
    cv2.putText(img, label_text, (x1, y1 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

def draw_dashed_line(img, pt1, pt2, color, thickness=1, dash_length=10):
    dist = np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist == 0:
        return
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = (
            int(pt1[0] + (pt2[0] - pt1[0]) * (i / dashes)),
            int(pt1[1] + (pt2[1] - pt1[1]) * (i / dashes))
        )
        end = (
            int(pt1[0] + (pt2[0] - pt1[0]) * ((i + 0.5) / dashes)),
            int(pt1[1] + (pt2[1] - pt1[1]) * ((i + 0.5) / dashes))
        )
        cv2.line(img, start, end, color, thickness, cv2.LINE_AA)

def draw_iou_label(img, pt, iou_val, color=(255, 0, 255), text_color=(255, 255, 255)):
    text = f"IoU: {iou_val:.3f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x1 = pt[0] - tw // 2 - 4
    y1 = pt[1] - th // 2 - 4
    x2 = pt[0] + tw // 2 + 4
    y2 = pt[1] + th // 2 + 4
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(img, text, (x1 + 4, y2 - 4), font, font_scale, text_color, thickness, cv2.LINE_AA)

def process_video(video_path, output_path, model, device):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n[INFO] Processing video: {video_path.name}")
    print(f"       Dimensions: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    # Output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # BGR Colors
    color_person = (255, 0, 0)       # Blue
    color_bike = (0, 200, 255)       # Yellow/Orange
    color_overlap = (255, 0, 255)    # Purple/Magenta

    for f_idx in tqdm(range(total_frames), desc=f"Annotating {video_path.name}"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO prediction (classes: 0 = person, 1 = bicycle)
        results = model.predict(frame, classes=[0, 1], conf=0.3, verbose=False, device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        clss = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        persons = [{"box": b, "conf": c} for b, c, cls in zip(boxes, confs, clss) if cls == 0]
        bicycles = [{"box": b, "conf": c} for b, c, cls in zip(boxes, confs, clss) if cls == 1]

        annotated_frame = frame.copy()

        # Draw Person boxes
        for p in persons:
            draw_label(annotated_frame, p["box"], "Person", p["conf"], color_person)

        # Draw Bicycle boxes
        for b in bicycles:
            draw_label(annotated_frame, b["box"], "Bicycle", b["conf"], color_bike)

        overlap_count = 0
        # Calculate overlap and draw lines for IoU > 0
        for p in persons:
            p_box = p["box"]
            p_center = (int((p_box[0] + p_box[2]) / 2), int((p_box[1] + p_box[3]) / 2))
            
            for b in bicycles:
                b_box = b["box"]
                b_center = (int((b_box[0] + b_box[2]) / 2), int((b_box[1] + b_box[3]) / 2))
                
                iou_val = calculate_iou(p_box, b_box)
                if iou_val > 0:
                    overlap_count += 1
                    # Draw connecting dashed line
                    draw_dashed_line(annotated_frame, p_center, b_center, color_overlap, thickness=2, dash_length=8)
                    
                    # Highlight overlap intersection region
                    ix1 = max(p_box[0], b_box[0])
                    iy1 = max(p_box[1], b_box[1])
                    ix2 = min(p_box[2], b_box[2])
                    iy2 = min(p_box[3], b_box[3])
                    
                    if ix1 < ix2 and iy1 < iy2:
                        overlay = annotated_frame.copy()
                        cv2.rectangle(overlay, (int(ix1), int(iy1)), (int(ix2), int(iy2)), color_overlap, -1)
                        cv2.addWeighted(overlay, 0.4, annotated_frame, 0.6, 0, annotated_frame)
                        cv2.rectangle(annotated_frame, (int(ix1), int(iy1)), (int(ix2), int(iy2)), color_overlap, 1, cv2.LINE_8)
                    
                    # Draw IoU value pill at midpoint
                    mid_point = (int((p_center[0] + b_center[0]) / 2), int((p_center[1] + b_center[1]) / 2))
                    draw_iou_label(annotated_frame, mid_point, iou_val, color_overlap)

        # Draw HUD Panel on top-left
        hud_w, hud_h = 300, 110
        hud_overlay = annotated_frame.copy()
        cv2.rectangle(hud_overlay, (10, 10), (10 + hud_w, 10 + hud_h), (0, 0, 0), -1)
        cv2.addWeighted(hud_overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        cv2.rectangle(annotated_frame, (10, 10), (10 + hud_w, 10 + hud_h), (255, 255, 255), 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(annotated_frame, f"Video: {video_path.name}", (20, 32), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Frame: {f_idx} / {total_frames}", (20, 52), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Detections: Persons: {len(persons)} | Bicycles: {len(bicycles)}", (20, 72), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated_frame, f"Overlaps (IoU > 0): {overlap_count}", (20, 92), font, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        # Write frame to video
        out_video.write(annotated_frame)

    cap.release()
    out_video.release()
    print(f"[SUCCESS] Saved annotated video to: {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(" YOLOv8 DETECTIONS & IoU OVERLAP VISUALIZATION ".center(70, "■"))
    print("=" * 70)
    print(f"[INFO] Running on device: {device}")

    # Paths Setup
    out_dir = project_root / "output" / "detection_iou_viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load YOLO
    yolo_weights = project_root / "models" / "weights" / "yolo26n.pt"
    if not yolo_weights.exists():
        yolo_weights = project_root / "models" / "weights" / "yolov8n.pt"
    if not yolo_weights.exists():
        yolo_weights = Path("yolov8n.pt")
    
    print(f"[INFO] Loading YOLO detector from: {yolo_weights}")
    model = YOLO(str(yolo_weights))
    if device.type == 'cuda':
        model.to(device)

    videos_to_process = [
        project_root / "data" / "00038.mp4",
        project_root / "data" / "00045.mp4"
    ]

    for v_path in videos_to_process:
        if not v_path.exists():
            print(f"[ERROR] Input video not found: {v_path}")
            continue
        
        out_name = f"{v_path.stem}_detection_iou.mp4"
        out_path = out_dir / out_name
        process_video(v_path, out_path, model, device)

    print("=" * 70)
    print(" PROCESSING COMPLETED SUCCESSFULLY ".center(70, "■"))
    print("=" * 70)

if __name__ == "__main__":
    main()
