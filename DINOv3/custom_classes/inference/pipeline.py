import cv2
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import timm
import os
import sys
import json
import queue
import threading
from ultralytics import YOLO
from PIL import Image
from pathlib import Path

# Add project root to sys path
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.detector import Detection
from src.trackers.ocsort.ocsort import OCSortTracker
from src.utils.helpers import calculate_iou

class LightweightDINOInference(nn.Module):
    def __init__(self, model_name="vit_small_patch16_dinov3.lvd1689m", num_classes=2):
        super(LightweightDINOInference, self).__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x) 
        logits = self.classifier(features)
        return logits, features

class DynamicInferencePipeline:
    def __init__(self, yolo_path=None, dino_weights=None, device=None, 
                 frame_skip_interval=2, periodic_classification_interval=10, min_box_size=30, class_thresh=0.60):
        
        current_dir = Path(__file__).resolve().parent
        custom_classes_dir = current_dir.parent
        project_root = custom_classes_dir.parent
        
        if yolo_path is None:
            yolo_path = str(project_root / "models" / "weights" / "yolo26l.pt")
        if dino_weights is None:
            dino_weights = str(custom_classes_dir / "output_train" / "dinov3_custom_classifier.pth")
            
        if device is not None:
            if "cuda" in str(device) and not torch.cuda.is_available():
                print("⚠️ CUDA device requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.frame_skip_interval = int(frame_skip_interval)
        self.periodic_classification_interval = int(periodic_classification_interval)
        self.min_box_size = int(min_box_size)
        self.class_thresh = float(class_thresh)
        self.track_dino_count = {}
        
        # 1. YOLO (Fast)
        print("Loading YOLO...")
        self.yolo = YOLO(yolo_path)
        if self.device.type == 'cuda':
            self.yolo.to(self.device)
        
        # 2. DINOv3 + Classifier Head
        print("Loading Custom DINOv3 Classifier...")
        self.classes = []
        
        if not os.path.exists(dino_weights):
            raise FileNotFoundError(f"Cannot find checkpoint: {dino_weights}")
            
        checkpoint = torch.load(dino_weights, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict) and "class_names" in checkpoint:
            self.classes = list(checkpoint["class_names"])
            state_dict = checkpoint.get('model_state_dict', checkpoint)
        else:
            raise ValueError("Checkpoint does not contain 'class_names'!")
            
        print(f"Loaded dynamic classes from checkpoint: {self.classes}")
            
        num_classes = len(self.classes)
        self.dino = LightweightDINOInference(num_classes=num_classes).to(self.device)
        self.dino.load_state_dict(state_dict, strict=False)
        self.dino.eval()
        
        if self.device.type == 'cuda':
            self.dino.half()
            
        model_cfg = timm.data.resolve_model_data_config(self.dino.backbone)
        self.transform = timm.data.create_transform(**model_cfg, is_training=False)
        
        # 3. Tracker (OCSORT)
        print("Loading OCSortTracker...")
        self.tracker = OCSortTracker(
            det_thresh=0.0,
            iou_thresh=0.3,
            max_age=30,
            min_hits=3,
        )

    def _track_to_dict(self, track, round_digits: int = 2) -> dict[str, Any]:
        bbox = [round(float(value), round_digits) for value in track.bbox_xyxy.tolist()]
        class_id = int(track.class_id)
        if class_id == -1:
            label = "person"
        else:
            label = self.classes[class_id] if 0 <= class_id < len(self.classes) else str(class_id)
        return {
            "id": int(track.id),
            "bbox_xyxy": bbox,
            "class_id": class_id,
            "label": label,
            "score": round(float(track.score), round_digits),
            "age": int(track.age),
            "hits": int(track.hits),
            "time_since_update": int(track.time_since_update),
        }
 
    def detect_yolo(self, frame):
        # Only detect class 0 (person)
        results = self.yolo.predict(frame, classes=[0], conf=0.3, verbose=False, device=self.device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        person_boxes = []
        raw_detections = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # Skip bounding boxes that are too small
            if w < self.min_box_size or h < self.min_box_size:
                continue
                
            person_boxes.append([float(x) for x in box])
            raw_detections.append({
                "bbox_xyxy": [round(float(value), 2) for value in box.tolist()],
                "class_id": 0,
                "label": "person",
                "score": round(float(conf), 4),
            })
            
        return person_boxes, raw_detections

    def classify_and_track(self, frame, person_boxes, raw_detections, frame_idx=0):
        if not person_boxes:
            self.tracker.predict_only()
            frame_payload = {
                "frame_idx": int(frame_idx),
                "raw_detections": raw_detections,
                "person_detections": [],
                "tracks": [self._track_to_dict(track) for track in self.tracker.active_tracks],
            }
            return self.tracker.active_tracks, frame_payload

        detections = []
        person_detections = []
        
        crops_to_classify = []
        indices_to_classify = []
        
        prev_tracks = list(self.tracker.active_tracks)
        
        for idx, box in enumerate(person_boxes):
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
            assigned_class = -1 # default fallback
            assigned_score = 0.5
            
            if best_t is not None and best_iou > 0.5:
                tid = int(best_t.id)
                frames_since_dino = self.track_dino_count.get(tid, 0)
                is_stable = best_t.hits >= 3 and best_t.score >= 0.6
                
                if is_stable and frames_since_dino < self.periodic_classification_interval and not is_too_small:
                    reuse_classification = True
                    assigned_class = int(best_t.class_id)
                    assigned_score = float(best_t.score)
                    self.track_dino_count[tid] = frames_since_dino + self.frame_skip_interval
            
            if is_too_small and not reuse_classification:
                reuse_classification = True
                assigned_class = -1
                assigned_score = 0.5
                
            if reuse_classification:
                detections.append(Detection(
                    bbox_xyxy=np.array(box, dtype=np.float32),
                    score=assigned_score,
                    class_id=assigned_class
                ))
                person_detections.append({
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                    "class_id": assigned_class,
                    "label": "person" if assigned_class == -1 else (self.classes[assigned_class] if assigned_class < len(self.classes) else str(assigned_class)),
                    "score": round(assigned_score, 4),
                    "dino_skipped": True
                })
            else:
                # 30% padding logic directly applied
                pad_w = int(w * 0.3)
                pad_h = int(h * 0.3)
                x1_pad, y1_pad = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2_pad, y2_pad = min(frame.shape[1], x2 + pad_w), min(frame.shape[0], y2 + pad_h)
                
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size > 0:
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    crops_to_classify.append(self.transform(crop_pil))
                    indices_to_classify.append(idx)
                else:
                    detections.append(Detection(
                        bbox_xyxy=np.array(box, dtype=np.float32),
                        score=0.5,
                        class_id=-1
                    ))
                    person_detections.append({
                        "bbox_xyxy": [round(float(value), 2) for value in box],
                        "class_id": -1,
                        "label": "person",
                        "score": 0.5,
                        "dino_skipped": True
                    })
        
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
                score_val = float(scores[i])
                if score_val >= self.class_thresh:
                    label_id = int(preds[i])
                else:
                    label_id = -1
                
                box = person_boxes[idx]
                
                detections.append(Detection(
                    bbox_xyxy=np.array(box, dtype=np.float32),
                    score=score_val,
                    class_id=label_id
                ))
                person_detections.append({
                    "bbox_xyxy": [round(float(value), 2) for value in box],
                    "class_id": label_id,
                    "label": "person" if label_id == -1 else (self.classes[label_id] if label_id < len(self.classes) else str(label_id)),
                    "score": round(score_val, 4),
                    "dino_skipped": False
                })
        
        self.tracker.update(detections, timestamp=frame_idx)
        
        for track in self.tracker.active_tracks:
            tid = int(track.id)
            matched_det = None
            for cd in person_detections:
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

        active_ids = {int(track.id) for track in self.tracker.active_tracks}
        self.track_dino_count = {k: v for k, v in self.track_dino_count.items() if k in active_ids}

        frame_payload = {
            "frame_idx": int(frame_idx),
            "raw_detections": raw_detections,
            "person_detections": person_detections,
            "tracks": [self._track_to_dict(track) for track in self.tracker.active_tracks],
        }
        return self.tracker.active_tracks, frame_payload

def process_video_dynamic(video_path, output_path, dino_weights=None, json_output_path=None, device=None, class_thresh=0.60):
    print(f"🚀 Starting dynamic video processing: {video_path}")
    
    pipeline = DynamicInferencePipeline(dino_weights=dino_weights, device=device, class_thresh=class_thresh)
    classes = pipeline.classes
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video info: {w}x{h} @ {fps} FPS, Total frames: {frame_count}")
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        
    if json_output_path is None:
        json_output_path = str(Path(output_path).with_suffix(".json"))
        
    json_dir = os.path.dirname(json_output_path)
    if json_dir:
        os.makedirs(json_dir, exist_ok=True)
    
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
                yolo_queue.put(None)
                break
            count, frame = item
            
            if pipeline.frame_skip_interval > 1 and count % pipeline.frame_skip_interval != 0:
                yolo_queue.put((count, frame, None, None, True))
            else:
                person_boxes, raw_detections = pipeline.detect_yolo(frame)
                yolo_queue.put((count, frame, person_boxes, raw_detections, False))

    def tracker_worker():
        while True:
            item = yolo_queue.get()
            if item is None:
                tracker_queue.put(None)
                break
            count, frame, person_boxes, raw_detections, skipped = item
            
            if skipped:
                pipeline.tracker.predict_only()
                frame_payload = {
                    "frame_idx": int(count),
                    "raw_detections": [],
                    "tracks": [pipeline._track_to_dict(track) for track in pipeline.tracker.active_tracks],
                    "skipped": True
                }
                active_tracks = pipeline.tracker.active_tracks
            else:
                active_tracks, frame_payload = pipeline.classify_and_track(
                    frame, person_boxes, raw_detections, count
                )
                
            tracks_snapshot = []
            for track in active_tracks:
                tracks_snapshot.append({
                    "id": int(track.id),
                    "bbox_xyxy": track.bbox_xyxy.copy(),
                    "class_id": int(track.class_id)
                })
                
            tracker_queue.put((count, frame, tracks_snapshot, frame_payload))

    t1 = threading.Thread(target=reader_worker, name="ReaderThread")
    t2 = threading.Thread(target=yolo_worker, name="YoloThread")
    t3 = threading.Thread(target=tracker_worker, name="TrackerThread")
    
    t1.start()
    t2.start()
    t3.start()

    json_frames = []
    processed_count = 0
    
    import colorsys
    def get_color(class_id, total_classes):
        hue = class_id / max(1, total_classes)
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
    class_colors = {i: get_color(i, len(classes)) for i in range(len(classes))}

    while True:
        item = tracker_queue.get()
        if item is None:
            break
            
        count, frame, tracks_snapshot, frame_payload = item
        json_frames.append(frame_payload)
        
        frame_text = f"Frame: {count}"
        cv2.putText(frame, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        for t in tracks_snapshot:
            x1, y1, x2, y2 = map(int, t["bbox_xyxy"])
            class_id = t["class_id"]
            
            if class_id == -1:
                label = "person"
                color = (128, 128, 128)
            else:
                label = classes[class_id] if 0 <= class_id < len(classes) else str(class_id)
                color = class_colors.get(class_id, (0, 255, 0))
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label_text = f"ID:{t['id']} {label}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        out.write(frame)
        
        processed_count += 1
        if processed_count % 30 == 0:
            print(f"⏳ Written {processed_count}/{frame_count} frames...")

    t1.join()
    t2.join()
    t3.join()
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    final_tracks = [
        {
            **track,
            "label": "person" if track["class_id"] == -1 else (classes[track["class_id"]] if 0 <= track["class_id"] < len(classes) else str(track["class_id"])),
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

    print(f"✅ Complete! Video saved to: {output_path}")
    print(f"✅ JSON result saved to: {json_output_path}")
