import cv2
from typing import Any
import torch
import torch.nn as nn
import numpy as np
import timm
import os
import sys
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from pathlib import Path

# Ensure the project root (two levels above this file: src/inference/ → src/ → root)
# is on sys.path so that `src.*` imports resolve correctly when running scripts
# directly (e.g. `PYTHONPATH=. python experiments/optimize_crops.py`).
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src.core.detector import Detection
from src.trackers.ocsort.ocsort import OCSortTracker
from src.utils.helpers import calculate_iou

DINOV3_BACKBONE_NAME = "vit_small_patch16_dinov3.lvd1689m"

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class LightweightDINOInference(nn.Module):
    def __init__(self, model_name=DINOV3_BACKBONE_NAME, num_classes=2):
        super(LightweightDINOInference, self).__init__()
        # Load Backbone
        try:
            self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        except RuntimeError as exc:
            if "Unknown model" in str(exc):
                raise RuntimeError(
                    f"DINOv3 backbone '{model_name}' is not available in the installed timm version. "
                    "Install timm>=1.0.20, or load the official DINOv3 repo with torch.hub instead."
                ) from exc
            raise
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # Classifier Head (Using the same architecture as training)
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x) # [Batch, 384]
        # Get logits
        logits = self.classifier(features)
        return logits, features

# ==========================================
# 2. MAIN INFERENCE PIPELINE (OCSORT)
# ==========================================
class TrafficInferencePipeline:
    def __init__(self, yolo_path=None, dino_weights=None, device=None, 
                 frame_skip_interval=2, periodic_classification_interval=10, min_box_size=30):
        dino_root = Path(__file__).resolve().parents[2]
        if yolo_path is None:
            p = dino_root / "models" / "weights" / "yolo26n.pt"
            yolo_path = str(p) if p.exists() else r"models\weights\yolo26n.pt"
        if dino_weights is None:
            p = dino_root / "models" / "weights" / "dinov3_lightweight_classifier.pth"
            dino_weights = str(p) if p.exists() else r"models\weights\dinov3_lightweight_classifier.pth"
            
        if device is not None:
            if "cuda" in str(device) and not torch.cuda.is_available():
                print("⚠️ CUDA device requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Optimization settings
        self.frame_skip_interval = int(frame_skip_interval)
        self.periodic_classification_interval = int(periodic_classification_interval)
        self.min_box_size = int(min_box_size)
        self.track_dino_count = {}
        
        # 1. YOLO (Fast)
        print("Loading YOLOv8n...")
        self.yolo = YOLO(yolo_path)
        if self.device.type == 'cuda':
            self.yolo.to(self.device)
        
        # 2. DINOv3 + Classifier Head
        print("Loading DINOv3 Classifier...")
        self.classes = ["pushing", "riding"]
        self.optimal_threshold = None
        state_dict = None
        
        if os.path.exists(dino_weights):
            checkpoint = torch.load(dino_weights, map_location=self.device, weights_only=True)
            if isinstance(checkpoint, dict):
                if "class_names" in checkpoint:
                    self.classes = list(checkpoint["class_names"])
                if "optimal_threshold" in checkpoint:
                    self.optimal_threshold = float(checkpoint["optimal_threshold"])
                    print(f"Loaded optimal threshold: {self.optimal_threshold:.4f}")
                state_dict = checkpoint.get('model_state_dict', checkpoint)
            else:
                state_dict = checkpoint
            print("Loaded fine-tuned DINOv3 metadata successfully.")
            
        num_classes = len(self.classes)
        self.dino = LightweightDINOInference(num_classes=num_classes).to(self.device)
        
        if state_dict is not None:
            self.dino.load_state_dict(state_dict, strict=False)
            print("Loaded fine-tuned DINOv3 weights successfully.")
            
        self.dino.eval()
        
        # Optimization: FP16
        if self.device.type == 'cuda':
            self.dino.half()
            
        # Transforms
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
        lower_classes = [name.lower() for name in self.classes]
        self.pushing_class_id = lower_classes.index("pushing") if "pushing" in lower_classes else 0
        self.riding_class_id = lower_classes.index("riding") if "riding" in lower_classes else 1
        print(f"DINOv3 class mapping: {self.classes} (pushing={self.pushing_class_id}, riding={self.riding_class_id})")

    def _track_to_dict(self, track, round_digits: int = 2) -> dict[str, Any]:
        bbox = [round(float(value), round_digits) for value in track.bbox_xyxy.tolist()]
        class_id = int(track.class_id)
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
        # Step 1: Detection (YOLO)
        results = self.yolo.predict(frame, classes=[0, 1], conf=0.3, verbose=False, device=self.device)[0]
        
        boxes = results.boxes.xyxy.cpu().numpy()
        clss = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        raw_class_names = {0: 'person', 1: 'bicycle'}
        raw_detections = []
        for box, cls_id, conf in zip(boxes, clss, confs):
            raw_class_id = int(cls_id)
            raw_detections.append({
                "bbox_xyxy": [round(float(value), 2) for value in box.tolist()],
                "class_id": raw_class_id,
                "label": raw_class_names.get(raw_class_id, str(raw_class_id)),
                "score": round(float(conf), 4),
            })
        
        # Step 2: Pair Person & Bicycle -> Cyclist
        persons = boxes[clss == 0]
        bikes = boxes[clss == 1]

        cyclist_boxes = []
        cyclist_sources = []
        used_bike_indices = set()
        for p in persons:
            for bi, b in enumerate(bikes):
                if bi not in used_bike_indices and calculate_iou(p, b) > 0.1:  # default IoU threshold
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
            cyclist_sources = [src for idx, src in enumerate(cyclist_sources) if idx not in suppressed]
            
        return cyclist_boxes, cyclist_sources, raw_detections

    def classify_and_track(self, frame, cyclist_boxes, cyclist_sources, raw_detections, frame_idx=0):
        if not cyclist_boxes:
            self.tracker.predict_only()
            frame_payload = {
                "frame_idx": int(frame_idx),
                "raw_detections": raw_detections,
                "cyclist_detections": [],
                "tracks": [self._track_to_dict(track) for track in self.tracker.active_tracks],
            }
            return self.tracker.active_tracks, frame_payload

        # 3. Dynamic Classification / Algorithmic Optimization
        detections = []
        cyclist_detections = []
        
        crops_to_classify = []
        indices_to_classify = []
        
        prev_tracks = list(self.tracker.active_tracks)
        
        for idx, box in enumerate(cyclist_boxes):
            x1, y1, x2, y2 = map(int, box)
            w, h = x2 - x1, y2 - y1
            
            # Size Filtering
            is_too_small = (w < self.min_box_size) or (h < self.min_box_size)
            
            # Find best match in previous tracks
            best_iou = 0.0
            best_t = None
            for t in prev_tracks:
                iou_val = calculate_iou(box, t.bbox_xyxy)
                if iou_val > best_iou:
                    best_iou = iou_val
                    best_t = t
            
            reuse_classification = False
            assigned_class = self.riding_class_id
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
                # Bypass DINOv3 for very small boxes and default to Riding
                reuse_classification = True
                assigned_class = self.riding_class_id
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
                # Needs DINOv3 classification
                # Crop with padding
                h_pad, w_pad = h // 10, w // 10
                x1_pad, y1_pad = max(0, x1 - w_pad), max(0, y1 - h_pad)
                x2_pad, y2_pad = min(frame.shape[1], x2 + w_pad), min(frame.shape[0], y2 + h_pad)
                crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
                
                if crop.size > 0:
                    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    crops_to_classify.append(self.transform(crop_pil))
                    indices_to_classify.append(idx)
                else:
                    # Fallback
                    detections.append(Detection(
                        bbox_xyxy=np.array(box, dtype=np.float32),
                        score=0.5,
                        class_id=self.riding_class_id
                    ))
                    cyclist_detections.append({
                        "bbox_xyxy": [round(float(value), 2) for value in box],
                        "class_id": self.riding_class_id,
                        "label": self.classes[self.riding_class_id],
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
                
                if self.optimal_threshold is not None and len(self.classes) == 2:
                    pushing_probs = probs[:, self.pushing_class_id]
                    preds = torch.where(
                        pushing_probs >= self.optimal_threshold,
                        torch.tensor(self.pushing_class_id, device=self.device),
                        torch.tensor(self.riding_class_id, device=self.device)
                    )
                    scores = probs.gather(1, preds.unsqueeze(1)).squeeze(1)
                else:
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
        
        # Step 4: Tracking Update with OCSORT
        self.tracker.update(detections, timestamp=frame_idx)
        
        # Manage DINOv3 age count for active tracks
        for track in self.tracker.active_tracks:
            tid = int(track.id)
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
        return self.tracker.active_tracks, frame_payload

    def process_frame(self, frame, frame_idx=0):
        # 1. Frame Skipping Logic
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

        cyclist_boxes, cyclist_sources, raw_detections = self.detect_yolo(frame)
        active_tracks, frame_payload = self.classify_and_track(
            frame, cyclist_boxes, cyclist_sources, raw_detections, frame_idx
        )
        return active_tracks, self.classes, frame_payload
