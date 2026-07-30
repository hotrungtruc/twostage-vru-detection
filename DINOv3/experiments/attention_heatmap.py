import os
import sys
import random
import glob
import cv2
import torch
import numpy as np
import timm
from PIL import Image
from pathlib import Path

# Setup path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.inference.inference_pipeline import LightweightDINOInference

def unnormalize(tensor, mean, std):
    """Un-normalize tensor for original image visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def generate_attention_map():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load Model DINOv3
    print("⏳ Loading DINOv3 model...")
    dino_weights = str(project_root / "models" / "weights" / "dinov3_lightweight_classifier.pth")
    model = LightweightDINOInference().to(device)
    
    if os.path.exists(dino_weights):
        checkpoint = torch.load(dino_weights, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded fine-tuned weights.")
    else:
        print("⚠️ Warning: Fine-tuned weights not found. Using pretrained backbone.")
        
    model.eval()

    # Initialize standard model transform
    model_cfg = timm.data.resolve_model_data_config(model.backbone)
    transform = timm.data.create_transform(**model_cfg, is_training=False)
    mean = model_cfg.get('mean', (0.485, 0.456, 0.406))
    std = model_cfg.get('std', (0.229, 0.224, 0.225))

    # Register Hook to extract Attention Scores from the last layer
    attention_maps = []
    def hook(module, input, output):
        attention_maps.append(input[0].detach().cpu())
        
    if hasattr(model.backbone.blocks[-1].attn, 'fused_attn'):
        model.backbone.blocks[-1].attn.fused_attn = False

    handle = model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(hook)

    # Output directory
    out_dir = str(project_root / "output" / "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    # Data directory
    cluster0_dir = str(project_root / "data" / "labeled_data" / "cluster0")
    cluster1_dir = str(project_root / "data" / "labeled_data" / "cluster1")
    
    cluster0_imgs = glob.glob(os.path.join(cluster0_dir, "*.*"))
    cluster1_imgs = glob.glob(os.path.join(cluster1_dir, "*.*"))
    
    # Randomly select 20 images from each cluster
    num_samples = min(20, len(cluster0_imgs))
    selected_c0 = random.sample(cluster0_imgs, num_samples) if cluster0_imgs else []
    
    num_samples = min(20, len(cluster1_imgs))
    selected_c1 = random.sample(cluster1_imgs, num_samples) if cluster1_imgs else []
    
    selected_imgs = [(img, 'cluster0') for img in selected_c0] + [(img, 'cluster1') for img in selected_c1]

    target_img_paths = [
        os.path.join(cluster0_dir, "crop_00045_f0_0.jpg"),
        os.path.join(cluster0_dir, "crop_00045_f7449_829.jpg")
    ]
    for target_img_path in target_img_paths:
        if os.path.exists(target_img_path) and not any(p == target_img_path for p, _ in selected_imgs):
            selected_imgs.append((target_img_path, 'cluster0'))

    if not selected_imgs:
        print("❌ Could not find any images in the two cluster directories!")
        return

    # Get patch size
    patch_size = model.backbone.patch_embed.patch_size[0]

    for img_path, label in selected_imgs:
        print(f"📸 Processing {label}: {os.path.basename(img_path)}")
        attention_maps.clear()
        
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        
        with torch.no_grad():
            logits, features = model(img_tensor.unsqueeze(0).to(device))
            
        attn = attention_maps[0][0]
        
        num_prefix_tokens = getattr(model.backbone, 'num_prefix_tokens', 1)
        cls_attn_all = attn[:, 0, num_prefix_tokens:]
        cls_attn_mean = cls_attn_all.mean(dim=0)
        
        h = img_tensor.shape[1] // patch_size
        w = img_tensor.shape[2] // patch_size
        
        attn_map_raw = cls_attn_mean.reshape(h, w).numpy()
        
        attn_map_normalized = (attn_map_raw - attn_map_raw.min()) / (attn_map_raw.max() - attn_map_raw.min() + 1e-8)
        attn_map_uint8 = np.uint8(255 * attn_map_normalized)
        
        img_display = unnormalize(img_tensor, mean, std)
        img_display = img_display.permute(1, 2, 0).cpu().numpy()
        img_display = np.clip(img_display, 0, 1)
        img_display = np.uint8(255 * img_display)
        
        img_display_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        # Calculate statistics
        mean_attn_normalized = float(np.mean(attn_map_normalized))
        print(f"\n[INFO] Analytics for {os.path.basename(img_path)}:")
        print(f"       - Mean Normalized Attention (patch-level): {mean_attn_normalized:.6f}")
        print(f"       - Final CLS Token shape: {features.shape}")
        print(f"       - Final CLS Token values (first 5): {features[0, :5].cpu().numpy()}")
        
        # Save raw and normalized attention matrices
        base_name = os.path.basename(img_path)
        raw_txt_path = os.path.join(out_dir, f"attn_raw_{base_name}.txt")
        norm_txt_path = os.path.join(out_dir, f"attn_norm_{base_name}.txt")
        cls_txt_path = os.path.join(out_dir, f"cls_token_{base_name}.txt")
        np.savetxt(raw_txt_path, attn_map_raw, fmt='%.6f')
        np.savetxt(norm_txt_path, attn_map_normalized, fmt='%.6f')
        np.savetxt(cls_txt_path, features[0].cpu().numpy(), fmt='%.6f')
        print(f"       - Raw attention matrix saved to: {raw_txt_path}")
        print(f"       - Normalized attention matrix saved to: {norm_txt_path}")
        print(f"       - CLS Token saved to: {cls_txt_path}\n")

        # Scale up for better visualization of grid text
        VIS_SCALE = 3
        scaled_h = img_display_bgr.shape[0] * VIS_SCALE
        scaled_w = img_display_bgr.shape[1] * VIS_SCALE
        scaled_patch = patch_size * VIS_SCALE
        
        img_display_bgr_scaled = cv2.resize(img_display_bgr, (scaled_w, scaled_h))
        
        # Visualize attention score of each patch (grid on the image)
        grid_img_bgr = img_display_bgr_scaled.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.35
        thickness = 1
        
        for i in range(h):
            for j in range(w):
                y1, y2 = i * scaled_patch, (i + 1) * scaled_patch
                x1, x2 = j * scaled_patch, (j + 1) * scaled_patch
                
                # Draw grid cell
                cv2.rectangle(grid_img_bgr, (x1, y1), (x2, y2), (255, 255, 255), 1)
                
                # Write normalized attention score
                score = attn_map_normalized[i, j]
                text = f"{score:.2f}"
                
                # Center text in patch
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = x1 + (scaled_patch - text_size[0]) // 2
                text_y = y1 + (scaled_patch + text_size[1]) // 2
                
                # Put text (green color for visibility)
                cv2.putText(grid_img_bgr, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        attn_map_resized = cv2.resize(attn_map_uint8, (scaled_w, scaled_h))
        
        heatmap_bgr = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
        
        alpha = 0.5
        overlay_bgr = cv2.addWeighted(img_display_bgr_scaled, alpha, heatmap_bgr, 1 - alpha, 0)
        cv2.putText(overlay_bgr, 'Mean Attn', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Process individual heads
        head_overlays = []
        num_heads = cls_attn_all.shape[0]
        for head_idx in range(num_heads):
            head_attn_raw = cls_attn_all[head_idx].reshape(h, w).numpy()
            head_attn_norm = (head_attn_raw - head_attn_raw.min()) / (head_attn_raw.max() - head_attn_raw.min() + 1e-8)
            head_attn_uint8 = np.uint8(255 * head_attn_norm)
            
            # Save raw and normalized attention matrices for each head
            head_raw_txt_path = os.path.join(out_dir, f"attn_raw_head{head_idx+1}_{base_name}.txt")
            head_norm_txt_path = os.path.join(out_dir, f"attn_norm_head{head_idx+1}_{base_name}.txt")
            np.savetxt(head_raw_txt_path, head_attn_raw, fmt='%.6f')
            np.savetxt(head_norm_txt_path, head_attn_norm, fmt='%.6f')
            
            head_attn_resized = cv2.resize(head_attn_uint8, (scaled_w, scaled_h))
            head_heatmap_bgr = cv2.applyColorMap(head_attn_resized, cv2.COLORMAP_JET)
            
            head_overlay = cv2.addWeighted(img_display_bgr_scaled, alpha, head_heatmap_bgr, 1 - alpha, 0)
            cv2.putText(head_overlay, f'Head {head_idx+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            head_overlays.append(head_overlay)
        
        cb_height = scaled_h
        cb_width = 60
        colorbar_bg = np.zeros((cb_height, cb_width, 3), dtype=np.uint8)
        
        gradient = np.linspace(255, 0, cb_height, dtype=np.uint8)
        gradient = np.tile(gradient[:, np.newaxis], (1, 20))
        colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        colorbar_bg[:, :20] = colorbar
        
        cv2.putText(colorbar_bg, 'High', (22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(colorbar_bg, 'Low', (22, cb_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        concat_img_parts = [img_display_bgr_scaled, grid_img_bgr, overlay_bgr] + head_overlays + [colorbar_bg]
        concat_img = np.hstack(concat_img_parts)
        
        out_name = f"{label}_{os.path.basename(img_path)}"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, concat_img)
        print(f"   => Saved heatmap to: {out_path}")

    handle.remove()
    print("✅ Complete!")

def generate_video_attention_map(video_path, max_frames=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # Load Model DINOv3
    print("⏳ Loading DINOv3 model...")
    dino_weights = str(project_root / "models" / "weights" / "dinov3_lightweight_classifier.pth")
    model = LightweightDINOInference().to(device)
    
    if os.path.exists(dino_weights):
        checkpoint = torch.load(dino_weights, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict, strict=False)
        print("✅ Loaded fine-tuned weights.")
    else:
        print("⚠️ Warning: Fine-tuned weights not found. Using pretrained backbone.")
        
    model.eval()

    # Initialize standard model transform
    model_cfg = timm.data.resolve_model_data_config(model.backbone)
    transform = timm.data.create_transform(**model_cfg, is_training=False)
    mean = model_cfg.get('mean', (0.485, 0.456, 0.406))
    std = model_cfg.get('std', (0.229, 0.224, 0.225))

    # Register Hook to extract Attention Scores from the last layer
    attention_maps = []
    def hook(module, input, output):
        attention_maps.append(input[0].detach().cpu())
        
    if hasattr(model.backbone.blocks[-1].attn, 'fused_attn'):
        model.backbone.blocks[-1].attn.fused_attn = False

    handle = model.backbone.blocks[-1].attn.attn_drop.register_forward_hook(hook)

    # Output directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = str(project_root / "output" / f"{video_name}_heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
        
    patch_size = model.backbone.patch_embed.patch_size[0]
    
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        print(f"📸 Processing frame {frame_idx}/{max_frames}")
        attention_maps.clear()
        
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tensor = transform(img)
        
        with torch.no_grad():
            logits, features = model(img_tensor.unsqueeze(0).to(device))
            
        attn = attention_maps[0][0]
        
        num_prefix_tokens = getattr(model.backbone, 'num_prefix_tokens', 1)
        cls_attn_all = attn[:, 0, num_prefix_tokens:]
        cls_attn_mean = cls_attn_all.mean(dim=0)
        
        h = img_tensor.shape[1] // patch_size
        w = img_tensor.shape[2] // patch_size
        
        attn_map = cls_attn_mean.reshape(h, w).numpy()
        
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        attn_map = np.uint8(255 * attn_map)
        
        img_display = unnormalize(img_tensor, mean, std)
        img_display = img_display.permute(1, 2, 0).cpu().numpy()
        img_display = np.clip(img_display, 0, 1)
        img_display = np.uint8(255 * img_display)
        
        img_display_bgr = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        
        attn_map_resized = cv2.resize(attn_map, (img_display_bgr.shape[1], img_display_bgr.shape[0]))
        
        heatmap_bgr = cv2.applyColorMap(attn_map_resized, cv2.COLORMAP_JET)
        
        alpha = 0.5
        overlay_bgr = cv2.addWeighted(img_display_bgr, alpha, heatmap_bgr, 1 - alpha, 0)
        cv2.putText(overlay_bgr, 'Mean Attn', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Process individual heads
        head_overlays = []
        num_heads = cls_attn_all.shape[0]
        for head_idx in range(num_heads):
            head_attn_raw = cls_attn_all[head_idx].reshape(h, w).numpy()
            head_attn_norm = (head_attn_raw - head_attn_raw.min()) / (head_attn_raw.max() - head_attn_raw.min() + 1e-8)
            head_attn_uint8 = np.uint8(255 * head_attn_norm)
            
            head_attn_resized = cv2.resize(head_attn_uint8, (img_display_bgr.shape[1], img_display_bgr.shape[0]))
            head_heatmap_bgr = cv2.applyColorMap(head_attn_resized, cv2.COLORMAP_JET)
            
            head_overlay = cv2.addWeighted(img_display_bgr, alpha, head_heatmap_bgr, 1 - alpha, 0)
            cv2.putText(head_overlay, f'Head {head_idx+1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            head_overlays.append(head_overlay)
        
        cb_height = img_display_bgr.shape[0]
        cb_width = 60
        colorbar_bg = np.zeros((cb_height, cb_width, 3), dtype=np.uint8)
        
        gradient = np.linspace(255, 0, cb_height, dtype=np.uint8)
        gradient = np.tile(gradient[:, np.newaxis], (1, 20))
        colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_JET)
        colorbar_bg[:, :20] = colorbar
        
        cv2.putText(colorbar_bg, 'High', (22, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(colorbar_bg, 'Low', (22, cb_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        concat_img_parts = [img_display_bgr, overlay_bgr] + head_overlays + [colorbar_bg]
        concat_img = np.hstack(concat_img_parts)
        
        out_name = f"frame_{frame_idx:04d}.jpg"
        out_path = os.path.join(out_dir, out_name)
        cv2.imwrite(out_path, concat_img)
        
        cls_txt_path = os.path.join(out_dir, f"cls_token_frame_{frame_idx:04d}.txt")
        np.savetxt(cls_txt_path, features[0].cpu().numpy(), fmt='%.6f')
        
        frame_idx += 1

    cap.release()
    handle.remove()
    print(f"✅ Complete! Saved {frame_idx} heatmaps to {out_dir}")

if __name__ == "__main__":
    generate_attention_map()
    # video_path = str(project_root / "data" / "00045.mp4")
    # generate_video_attention_map(video_path, max_frames=100)
