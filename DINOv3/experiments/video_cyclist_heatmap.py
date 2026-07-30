import os
import sys
import cv2
import torch
import numpy as np
import timm
from PIL import Image
from pathlib import Path
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[0]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.inference.inference_pipeline import TrafficInferencePipeline

def main():
    video_path = project_root / "data" / "2_riding_00044.mp4"
    if not video_path.exists():
        print(f"❌ Video not found: {video_path}")
        return

    out_dir = project_root / "output" / "2_riding_00044_crops_heatmaps"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load Pipeline
    print("⏳ Loading YOLO and DINOv3...")
    pipeline = TrafficInferencePipeline(device=device)
    model = pipeline.dino
    transform = pipeline.transform
    patch_size = model.backbone.patch_embed.patch_size[0]

    # Register Hook for Attention Maps
    attention_maps = []
    def hook(module, input, output):
        attention_maps.append(input[0].detach().cpu())

    if hasattr(model.backbone.blocks[-1].attn, 'fused_attn'):
        model.backbone.blocks[-1].attn.fused_attn = False

    handle = model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(hook)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📸 Processing video: {video_path.name} ({total_frames} frames)")
    print(f"💾 Saving cropped heatmaps to: {out_dir}")
    
    progress = tqdm(total=total_frames, desc="Extracting Crops")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        cyclist_boxes, _, _ = pipeline.detect_yolo(frame)
        
        # Merge all cyclist boxes into one crop
        if not cyclist_boxes:
            frame_idx += 1
            progress.update(1)
            continue
            
        x1 = int(min(b[0] for b in cyclist_boxes))
        y1 = int(min(b[1] for b in cyclist_boxes))
        x2 = int(max(b[2] for b in cyclist_boxes))
        y2 = int(max(b[3] for b in cyclist_boxes))
        
        w, h = x2 - x1, y2 - y1
        
        # Crop with padding
        h_pad, w_pad = h // 10, w // 10
        x1_pad, y1_pad = max(0, x1 - w_pad), max(0, y1 - h_pad)
        x2_pad, y2_pad = min(frame.shape[1], x2 + w_pad), min(frame.shape[0], y2 + h_pad)
        
        crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
        if crop.size == 0:
            frame_idx += 1
            progress.update(1)
            continue

        attention_maps.clear()
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img_tensor = transform(crop_pil).unsqueeze(0).to(device)
        
        if device.type == 'cuda':
            img_tensor = img_tensor.half()

        with torch.no_grad():
            _ = model(img_tensor)

        if not attention_maps:
            frame_idx += 1
            progress.update(1)
            continue

        attn = attention_maps[0][0]
        num_prefix_tokens = getattr(model.backbone, 'num_prefix_tokens', 1)
        cls_attn = attn[:, 0, num_prefix_tokens:]
        cls_attn = cls_attn.mean(dim=0)
        
        # Resolve shape
        th = img_tensor.shape[2] // patch_size
        tw = img_tensor.shape[3] // patch_size
        
        attn_map = cls_attn.reshape(th, tw).float().numpy()
        
        # Normalize
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = np.uint8(255 * attn_map)
        
        # Resize and colorize
        attn_map_resized = cv2.resize(attn_map, (crop.shape[1], crop.shape[0]))
        heatmap_bgr = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
        
        # Overlay
        alpha = 0.5
        overlay_bgr = cv2.addWeighted(crop, alpha, heatmap_bgr, 1 - alpha, 0)
        
        # Create colorbar
        cb_height = crop.shape[0]
        cb_width = 60
        colorbar_bg = np.zeros((cb_height, cb_width, 3), dtype=np.uint8)
        
        gradient = np.linspace(255, 0, cb_height, dtype=np.uint8)
        gradient = np.tile(gradient[:, np.newaxis], (1, 20))
        colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        colorbar_bg[:, :20] = colorbar
        
        cv2.putText(colorbar_bg, 'High', (22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(colorbar_bg, 'Low', (22, cb_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Concatenate: Original Crop | Overlay | Colorbar
        concat_img = np.hstack((crop, overlay_bgr, colorbar_bg))
        
        # Save
        out_img_name = f"frame_{frame_idx:04d}_merged_cyclists.jpg"
        out_img_path = out_dir / out_img_name
        cv2.imwrite(str(out_img_path), concat_img)

        frame_idx += 1
        progress.update(1)

    progress.close()
    cap.release()
    handle.remove()
    print(f"✅ Extracted and saved all cropped heatmaps to: {out_dir}")

if __name__ == "__main__":
    main()
