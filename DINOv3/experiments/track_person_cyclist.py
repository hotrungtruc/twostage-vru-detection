#!/usr/bin/env python3
"""
track_person_cyclist.py
=======================
A unified script that detects and tracks both Cyclists (Riding/Pushing states) and standalone Persons.
- Standalone Persons bypass DINOv3 state classification for optimized performance.
- Results are saved to output/result_dinov3/ by default.
"""

import os
import sys
import cv2
import json
import queue
import threading
import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ── Dynamic Path Setup ─────────────────────────────────────────────────────────
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.inference.inference_pipeline import TrafficInferencePipeline, Detection
from src.utils.helpers import calculate_iou, calculate_ioa



# ── Custom Pipeline Subclass ──────────────────────────────────────────────────
class PersonCyclistInferencePipeline(TrafficInferencePipeline):
    def __init__(self, device=None):
        # Initialize the base TrafficInferencePipeline
        super().__init__(device=device)
        # Extend target classes to track 'Person' as well
        self.classes = ['Riding', 'Pushing', 'Person']
        # Tuning parameters for better person tracking
        self.tracker.iou_thresh = 0.1
        self.tracker.max_age = 30

    def process_frame(self, frame, frame_idx=0):
        # 1. Frame Skipping Logic (for pipeline speed optimization)
        if self.frame_skip_interval > 1 and frame_idx % self.frame_skip_interval != 0:
            self.tracker.predict_only()
            frame_payload = {
                "frame_idx": int(frame_idx),
                "raw_detections": [],
                "cyclist_detections": [],
                "tracks": [self._track_to_dict(track) for track in self.tracker.active_tracks],
                "skipped": True
            }
            return self.tracker.active_tracks, self.classes, frame_payload

        # 2. Run YOLOv8 on classes [0, 1] (person, bicycle)
        results = self.yolo.predict(frame, classes=[0, 1], conf=0.3, verbose=False, device=self.device)[0]
        boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
        clss = results.boxes.cls.cpu().numpy() if results.boxes is not None else []
        confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else []

        raw_detections = []
        raw_class_names = {0: 'person', 1: 'bicycle'}
        for box, cls_id, conf in zip(boxes, clss, confs):
            raw_class_id = int(cls_id)
            raw_detections.append({
                "bbox_xyxy": [round(float(value), 2) for value in box.tolist()],
                "class_id": raw_class_id,
                "label": raw_class_names.get(raw_class_id, str(raw_class_id)),
                "score": round(float(conf), 4),
            })

        persons = boxes[clss == 0]
        person_confs = confs[clss == 0]
        bikes = boxes[clss == 1]

        # 3. Pair Person & Bicycle -> Cyclist (IoU > 0.1)
        cyclist_boxes = []
        cyclist_sources = []
        used_bike_indices = set()
        paired_person_indices = set()

        for p_idx, p in enumerate(persons):
            for bi, b in enumerate(bikes):
                if bi not in used_bike_indices and calculate_iou(p, b) > 0.1:
                    cx1 = min(p[0], b[0])
                    cy1 = min(p[1], b[1])
                    cx2 = max(p[2], b[2])
                    cy2 = max(p[3], b[3])
                    cyclist_boxes.append([cx1, cy1, cx2, cy2])
                    cyclist_sources.append({
                        "person_bbox_xyxy": [round(float(value), 2) for value in p.tolist()],
                        "bike_bbox_xyxy": [round(float(value), 2) for value in b.tolist()],
                    })
                    used_bike_indices.add(bi)
                    paired_person_indices.add(p_idx)
                    break

        # 4. Classify Cyclists using DINOv3
        detections = []
        cyclist_detections = []
        crops_to_classify = []
        indices_to_classify = []
        prev_tracks = list(self.tracker.active_tracks)

        for idx, box in enumerate(cyclist_boxes):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            is_too_small = (w < self.min_box_size) or (h < self.min_box_size)
            
            best_iou = 0.0
            best_t = None
            for t in prev_tracks:
                iou_val = calculate_iou(box, t.bbox_xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_t = t

            reuse_classification = False
            assigned_class = 0
            assigned_score = 0.5

            if best_t is not None and best_iou > 0.5:
                tid = int(best_t.id)
                # Only reuse classification if the previous track was a cyclist (class Riding/Pushing)
                if best_t.class_id in [0, 1]:
                    frames_since_dino = self.track_dino_count.get(tid, 0)
                    is_stable = best_t.hits >= 3 and best_t.score >= 0.6
                    if is_stable and frames_since_dino < self.periodic_classification_interval and not is_too_small:
                        reuse_classification = True
                        assigned_class = int(best_t.class_id)
                        assigned_score = float(best_t.score)
                        self.track_dino_count[tid] = frames_since_dino + self.frame_skip_interval

            if is_too_small and not reuse_classification:
                reuse_classification = True
                assigned_class = 0
                assigned_score = 0.5

            if reuse_classification:
                detections.append(Detection(
                    bbox_xyxy=np.array(box, dtype=np.float32),
                    score=assigned_score,
                    class_id=assigned_class
                ))
                cyclist_detections.append({
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                    "class_id": assigned_class,
                    "label": self.classes[assigned_class],
                    "score": round(assigned_score, 4),
                    "source": cyclist_sources[idx] if idx < len(cyclist_sources) else {},
                    "dino_skipped": True
                })
            else:
                # Add to queue for DINO classification
                h_pad, w_pad = h // 10, w // 10
                x1_pad, y1_pad = max(0, x1 - w_pad), max(0, y1 - h_pad)
                x2_pad, y2_pad = min(frame.shape[1], x2 + w_pad), min(frame.shape[0], y2 + h_pad)
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                if crop.size > 0:
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    crops_to_classify.append(self.transform(crop_pil))
                    indices_to_classify.append(idx)
                else:
                    detections.append(Detection(
                        bbox_xyxy=np.array(box, dtype=np.float32),
                        score=0.5,
                        class_id=0
                    ))
                    cyclist_detections.append({
                        "bbox_xyxy": [round(float(value), 2) for value in box],
                        "class_id": 0,
                        "label": self.classes[0],
                        "score": 0.5,
                        "source": cyclist_sources[idx] if idx < len(cyclist_sources) else {},
                        "dino_skipped": True
                    })

        # Batch Inference for DINOv3
        if crops_to_classify:
            batch_tensor = torch.stack(crops_to_classify).to(self.device)
            if self.device.type == 'cuda':
                batch_tensor = batch_tensor.half()
            with torch.no_grad():
                logits, _ = self.dino(batch_tensor)
                probs = torch.softmax(logits, dim=1)
                scores, preds = torch.max(probs, dim=1)
                preds = preds.cpu().numpy()
                scores = scores.cpu().numpy()
            
            for i, idx in enumerate(indices_to_classify):
                label_id = int(preds[i])
                score_val = float(scores[i])
                box = cyclist_boxes[idx]
                detections.append(Detection(
                    bbox_xyxy=np.array(box, dtype=np.float32),
                    score=score_val,
                    class_id=label_id
                ))
                cyclist_detections.append({
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                    "class_id": label_id,
                    "label": self.classes[label_id],
                    "score": round(score_val, 4),
                    "source": cyclist_sources[idx] if idx < len(cyclist_sources) else {},
                    "dino_skipped": False
                })

        # 5. Process unpaired persons (class 2: 'Person') - BYPASS DINOv3!
        for p_idx, p in enumerate(persons):
            if p_idx not in paired_person_indices:
                p_box = p
                # Heuristic: If this person is largely inside a newly detected cyclist box,
                # they are the same physical object. We suppress the duplicate standalone Person detection.
                is_duplicate = False
                            
                # Check newly paired cyclists in current frame
                for cy_box in cyclist_boxes:
                    if calculate_ioa(p_box, cy_box) > 0.6:
                        is_duplicate = True
                        break
                            
                if not is_duplicate:
                    detections.append(Detection(
                        bbox_xyxy=np.array(p_box, dtype=np.float32),
                        score=float(person_confs[p_idx]),
                        class_id=2  # Standalone Person
                    ))

        # 6. Tracking Update with OCSORT
        self.tracker.update(detections, timestamp=frame_idx)

        # Manage DINOv3 skips for active cyclist tracks
        for track in self.tracker.active_tracks:
            tid = int(track.id)
            if track.class_id in [0, 1]:  # Cyclists only
                matched_det = None
                for cd in cyclist_detections:
                    if calculate_iou(track.bbox_xyxy, cd["bbox_xyxy"]) > 0.5:
                        matched_det = cd
                        break
                if matched_det is not None:
                    if not matched_det.get("dino_skipped", False):
                        self.track_dino_count[tid] = 0
                    else:
                        if tid not in self.track_dino_count:
                            self.track_dino_count[tid] = 0
                else:
                    if tid in self.track_dino_count:
                        self.track_dino_count[tid] += self.frame_skip_interval

        # Clean stale track entries
        active_ids = {int(track.id) for track in self.tracker.active_tracks}
        self.track_dino_count = {k: v for k, v in self.track_dino_count.items() if k in active_ids}

        frame_payload = {
            "frame_idx": int(frame_idx),
            "raw_detections": raw_detections,
            "cyclist_detections": cyclist_detections,
            "tracks": [self._track_to_dict(track) for track in self.tracker.active_tracks],
        }
        return self.tracker.active_tracks, self.classes, frame_payload


# ── Custom Video Processing Loops ──────────────────────────────────────────────

def _default_json_output_path(output_path):
    return str(Path(output_path).with_suffix(".json"))

def process_video_custom(video_path, output_path, json_output_path=None, device=None):
    print(f"🚀 Initializing Custom Person + Cyclist standard pipeline: {video_path}")
    pipeline = PersonCyclistInferencePipeline(device=device)
    classes = pipeline.classes
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Info: {w}x{h} @ {fps} FPS, Total Frames: {frame_count}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if json_output_path is None:
        json_output_path = _default_json_output_path(output_path)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_queue = queue.Queue(maxsize=30)
    yolo_queue = queue.Queue(maxsize=30)
    tracker_queue = queue.Queue(maxsize=30)

    def reader_worker():
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((count, frame))
            count += 1
        frame_queue.put(None)

    def yolo_worker():
        while True:
            item = frame_queue.get()
            if item is None:
                tracker_queue.put(None)
                break
            count, frame = item
            active_tracks, classes_out, frame_payload = pipeline.process_frame(frame, frame_idx=count)
            
            tracks_snapshot = []
            for track in active_tracks:
                tracks_snapshot.append({
                    "id": int(track.id),
                    "bbox_xyxy": track.bbox_xyxy.copy(),
                    "class_id": int(track.class_id)
                })
            tracker_queue.put((count, frame, tracks_snapshot, frame_payload))

    t1 = threading.Thread(target=reader_worker, name="ReaderThread")
    t2 = threading.Thread(target=yolo_worker, name="TrackerThread")
    t1.start()
    t2.start()

    json_frames = []
    processed_count = 0
    
    while True:
        item = tracker_queue.get()
        if item is None:
            break
            
        count, frame, tracks_snapshot, frame_payload = item
        json_frames.append(frame_payload)
        
        # Draw Frame number on top left
        frame_text = f"Frame: {count}"
        cv2.putText(frame, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        for t in tracks_snapshot:
            x1, y1, x2, y2 = map(int, t["bbox_xyxy"])
            label = classes[t["class_id"]]
            
            if label == 'Riding':
                color = (0, 255, 0)      # Green
            elif label == 'Pushing':
                color = (0, 165, 255)    # Orange
            else:
                color = (255, 0, 0)      # Blue (Person)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"ID:{t['id']} {label}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        out.write(frame)
        processed_count += 1
        if processed_count % 30 == 0:
            print(f"⏳ Processed {processed_count}/{frame_count} frames...")

    t1.join()
    t2.join()
    
    cap.release()
    out.release()

    final_tracks = [
        {
            **track,
            "label": classes[track["class_id"]] if 0 <= track["class_id"] < len(classes) else str(track["class_id"]),
        }
        for track in pipeline.tracker.finalize(round_digits=2)
    ]

    result_json = {
        "video_input": video_path,
        "video_output": output_path,
        "json_output": json_output_path,
        "video_info": {
            "width": w,
            "height": h,
            "fps": fps,
            "frame_count": frame_count,
        },
        "classes": classes,
        "frames": json_frames,
        "tracks": final_tracks,
    }

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    print(f"✅ Success! Output video: {output_path}")
    print(f"✅ Success! Output JSON: {json_output_path}")
    return {
        "video_path": output_path,
        "json_path": json_output_path,
        "frame_count": processed_count,
    }



# ── Main Entrypoint ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified DINOv3 Custom Person + Cyclist Tracking CLI"
    )

    parser.add_argument(
        "--input",
        type=str,
        default="data/input.mp4",
        help="Input video path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/result_dinov3/person_cyclist_tracking.mp4",
        help="Output path for video results"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    
    args = parser.parse_args()

    # Apply output directory redirection matching the main system behavior
    output_path = Path(args.output)
    output_dir = project_root / "output" / "result_dinov3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        output_parts = output_path.parts
        if "output" in output_parts:
            idx = output_parts.index("output")
            rel_parts = output_parts[idx+1:]
            new_output_path = output_dir.joinpath(*rel_parts)
        else:
            new_output_path = output_dir / output_path.name
    except Exception:
        new_output_path = output_dir / output_path.name
        
    final_output = str(new_output_path)

    print("=" * 70)
    print(" PERSON + CYCLIST DINOv3 TRACKING CLI ".center(70, "■"))
    print("=" * 70)
    print(f"[INFO] Input video : {args.input}")
    print(f"[INFO] Output video: {final_output}")
    print(f"[INFO] Device      : {args.device}")
    print("=" * 70)

    if not os.path.exists(args.input):
        print(f"[ERROR] Input video not found at: '{args.input}'")
        sys.exit(1)

    process_video_custom(args.input, final_output, device=args.device)

    print("=" * 70)
    print(" PIPELINE COMPLETED SUCCESSFULLY ".center(70, "■"))
    print("=" * 70)

if __name__ == "__main__":
    main()
