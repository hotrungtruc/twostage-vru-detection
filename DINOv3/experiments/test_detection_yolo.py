import os
import sys
import cv2
import csv
import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ── Dynamic Path Setup ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[0]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.utils.helpers import calculate_iou

def draw_split_label(img, box, label, score, box_color, score_bg_color, text_color=(255, 255, 255)):
    x1, y1, x2, y2 = map(int, box)
    
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
    
    # Text setup
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    label_text = f" {label} "
    score_text = f" {score:.2f} "
    
    (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, font, font_scale, thickness)
    (scr_w, scr_h), _ = cv2.getTextSize(score_text, font, font_scale, thickness)
    
    # Draw label bg
    cv2.rectangle(img, (x1, y1 - lbl_h - 6), (x1 + lbl_w, y1), box_color, -1)
    # Draw score bg next to it
    cv2.rectangle(img, (x1 + lbl_w, y1 - scr_h - 6), (x1 + lbl_w + scr_w, y1), score_bg_color, -1)
    
    # Put text
    cv2.putText(img, label_text, (x1, y1 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.putText(img, score_text, (x1 + lbl_w, y1 - 3), font, font_scale, text_color, thickness, cv2.LINE_AA)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="YOLO-only cyclist detection logic test.")
    parser.add_argument("--limit-frames", type=int, default=None, help="Limit number of frames to process.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "data" / "input.mp4"),
        help="Path to input video file"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" YOLOv8 DETECTION LOGIC TEST & VIDEO GENERATION ".center(60, "■"))
    print("=" * 60)
    print(f"[INFO] Device: {device}")

    # Paths setup
    video_path = args.input
    if not os.path.exists(video_path):
        print(f"[ERROR] Input video not found at '{video_path}'. Please specify the correct path using '--input <path_to_video>'.")
        sys.exit(1)

    out_dir = project_root / "output" / "test_detection"
    os.makedirs(out_dir, exist_ok=True)

    # Load YOLO
    print("\n[INFO] Loading YOLO detector...")
    yolo_weights = str(project_root / "models" / "weights" / "yolo26n.pt")
    if not os.path.exists(yolo_weights):
        yolo_weights = "yolo26n.pt"
    model = YOLO(yolo_weights)
    if device.type == 'cuda':
        model.to(device)

    # Video setup
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video file: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video loaded: {video_path}")
    print(f"[INFO] Dimensions: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

    # Output video writer
    out_video_path = os.path.join(out_dir, "test_detection_result.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    limit = args.limit_frames if args.limit_frames is not None else total_frames
    print(f"[INFO] Processing up to {limit} frames...")

    frame_records = []
    
    # Trackers for aggregate stats
    total_persons = 0
    total_bicycles = 0
    total_cyclists = 0

    # BGR Colors
    color_person = (255, 0, 0)      # Blue
    color_bike = (0, 255, 255)     # Yellow
    color_cyclist = (0, 200, 0)     # Green
    color_blue_score = (255, 0, 0)  # Blue for cyclist score background

    for f_idx in tqdm(range(limit), desc="Processing video frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO prediction (classes: 0 = person, 1 = bicycle)
        results = model.predict(frame, classes=[0, 1], conf=0.3, verbose=False, device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        clss = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        # Step 2: Pair Person & Bicycle -> Cyclist (Exact logic from inference_pipeline.py)
        persons = boxes[clss == 0]
        person_confs = confs[clss == 0]
        bikes = boxes[clss == 1]
        bike_confs = confs[clss == 1]

        cyclist_boxes = []
        cyclist_confs = []
        used_bike_indices = set()
        
        persons_list = []
        for p_box, p_conf in zip(persons, person_confs):
            persons_list.append({"box": [float(x) for x in p_box], "conf": float(p_conf)})
            
        bicycles_list = []
        for b_box, b_conf in zip(bikes, bike_confs):
            bicycles_list.append({"box": [float(x) for x in b_box], "conf": float(b_conf)})

        for p_idx, p in enumerate(persons):
            p_conf = person_confs[p_idx]
            for bi, b in enumerate(bikes):
                b_conf = bike_confs[bi]
                if bi not in used_bike_indices and calculate_iou(p, b) > 0.1:
                    cx1 = min(p[0], b[0])
                    cy1 = min(p[1], b[1])
                    cx2 = max(p[2], b[2])
                    cy2 = max(p[3], b[3])
                    cyclist_boxes.append([cx1, cy1, cx2, cy2])
                    cyclist_confs.append(float((p_conf + b_conf) / 2))
                    used_bike_indices.add(bi)
                    break

        # Greedy NMS: remove duplicate cyclist boxes
        if len(cyclist_boxes) > 1:
            suppressed = set()
            for i in range(len(cyclist_boxes)):
                if i in suppressed:
                    continue
                for j in range(i + 1, len(cyclist_boxes)):
                    if j not in suppressed and calculate_iou(cyclist_boxes[i], cyclist_boxes[j]) > 0.5:
                        suppressed.add(j)
            cyclist_boxes = [b for idx, b in enumerate(cyclist_boxes) if idx not in suppressed]
            cyclist_confs = [c for idx, c in enumerate(cyclist_confs) if idx not in suppressed]

        # Draw annotations on the frame
        annotated_frame = frame.copy()

        # Draw Person boxes
        for p in persons_list:
            draw_split_label(annotated_frame, p["box"], "Person", p["conf"], color_person, color_person)

        # Draw Bicycle boxes
        for b in bicycles_list:
            draw_split_label(annotated_frame, b["box"], "Bicycle", b["conf"], color_bike, color_bike)

        # Draw Cyclist boxes (Split background label: Green class bg, Blue score bg)
        for cy_box, cy_conf in zip(cyclist_boxes, cyclist_confs):
            draw_split_label(annotated_frame, cy_box, "Cyclist", cy_conf, color_cyclist, color_blue_score)

        # Draw Frame Index on top-left
        cv2.rectangle(annotated_frame, (10, 10), (220, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, f"Frame: {f_idx}", (20, 38), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        # Write to video
        out_video.write(annotated_frame)

        # Update stats
        total_persons += len(persons_list)
        total_bicycles += len(bicycles_list)
        total_cyclists += len(cyclist_boxes)

        record = {
            "frame_index": f_idx,
            "timestamp_seconds": round(f_idx / fps, 2),
            "person_count": len(persons_list),
            "bicycle_count": len(bicycles_list),
            "cyclist_count": len(cyclist_boxes),
            "persons": persons_list,
            "bicycles": bicycles_list,
            "cyclists": [{"box": [float(x) for x in b], "conf": float(c)} for b, c in zip(cyclist_boxes, cyclist_confs)]
        }
        frame_records.append(record)

    cap.release()
    out_video.release()

    # Create CSV & JSON summaries
    summary_data = {
        "metadata": {
            "video_path": video_path,
            "total_frames_in_video": total_frames,
            "processed_frames_count": len(frame_records),
            "fps": fps,
            "resolution": f"{width}x{height}",
            "total_persons_detected": total_persons,
            "total_bicycles_detected": total_bicycles,
            "total_cyclists_detected": total_cyclists,
        },
        "frame_details": frame_records
    }

    # Save JSON report
    json_path = os.path.join(out_dir, "detection_test_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    # Save CSV report (frame by frame)
    csv_path = os.path.join(out_dir, "detection_test_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame Index", "Timestamp (s)", "Person Count", "Bicycle Count", "Cyclist Count"])
        for record in frame_records:
            writer.writerow([
                record["frame_index"],
                record["timestamp_seconds"],
                record["person_count"],
                record["bicycle_count"],
                record["cyclist_count"]
            ])

    # Save summary stats CSV
    summary_csv_path = os.path.join(out_dir, "detection_summary_statistics.csv")
    with open(summary_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Video Path", video_path])
        writer.writerow(["Total Frames in Video", total_frames])
        writer.writerow(["Processed Frames Count", len(frame_records)])
        writer.writerow(["FPS", fps])
        writer.writerow(["Resolution", f"{width}x{height}"])
        writer.writerow(["Total Persons Detected", total_persons])
        writer.writerow(["Total Bicycles Detected", total_bicycles])
        writer.writerow(["Total Cyclists Detected", total_cyclists])

    print("=" * 60)
    print(" YOLOv8 DETECTION LOGIC TEST SUCCESSFUL ".center(60, "■"))
    print("=" * 60)
    print(f"[SUCCESS] JSON report saved to: {json_path}")
    print(f"[SUCCESS] CSV frame report saved to: {csv_path}")
    print(f"[SUCCESS] CSV summary statistics saved to: {summary_csv_path}")
    print(f"[SUCCESS] Output detection video saved to: {out_video_path}")
    print("=" * 60)

if __name__ == "__main__":
    main()
