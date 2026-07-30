import os
import sys
import cv2
import json
import torch
import numpy as np
import timm
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ── Dynamic Path Setup ─────────────────────────────────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(current_dir).resolve().parents[0]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from src.inference.inference_pipeline import LightweightDINOInference
from src.utils.helpers import calculate_iou, crop_with_padding

def save_sample_visualization(sample, best_padding, padding_candidates, out_path):
    """
    Saves a visualization showing the crop images for each padding candidate
    along with their predicted class/confidence score, and a line plot of scores.
    """
    import matplotlib.pyplot as plt
    import cv2

    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, len(padding_candidates), height_ratios=[1.2, 1.0])
    
    scores = []
    classes = []
    
    # 1. Top row: crop images
    for i, p in enumerate(padding_candidates):
        p_str = str(p)
        run = sample[p_str]
        crop_bgr = run["crop"]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(crop_rgb)
        ax.axis('off')
        
        pred_class = run["class"]
        conf = run["confidence"]
        scores.append(conf)
        classes.append(pred_class)
        
        # Title above each crop (Padding % and Predicted Class + Conf)
        ax.set_title(f"{int(p*100)}%\n{pred_class} {conf:.2f}", fontsize=9, pad=5)
        
    # 2. Bottom row: score line chart
    ax_plot = fig.add_subplot(gs[1, :])
    x_indices = list(range(len(padding_candidates)))
    ax_plot.plot(x_indices, scores, marker='o', markersize=8, color='#1e70b8', linewidth=2.5)
    
    # Style bottom plot
    ax_plot.grid(True, linestyle=':', alpha=0.5, color='#cccccc')
    ax_plot.set_xlim(-0.5, len(padding_candidates) - 0.5)
    
    ymin = min(scores)
    ymax = max(scores)
    if ymax - ymin < 0.05:
        ax_plot.set_ylim(ymin - 0.02, 1.02)
    else:
        ax_plot.set_ylim(ymin - 0.05, ymax + 0.01)
        
    ax_plot.set_ylabel("Score", fontsize=11)
    ax_plot.set_xlabel("Padding ratio", fontsize=11)
    ax_plot.set_xticks(x_indices)
    ax_plot.set_xticklabels([f"{int(p*100)}%" for p in padding_candidates], fontsize=10)
    
    for spine in ax_plot.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.8)

    fig.tight_layout()
    plt.savefig(str(out_path), bbox_inches='tight', dpi=300)
    plt.close(fig)



def scan_video(video_path, yolo, dino, transform, device, padding_candidates, classes):
    """
    Scan a single video frame-by-frame (full video) and collect
    multi-padding cyclist records for every detected cyclist.
    Returns list of cyclist_runs dicts.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ❌ Cannot open: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"  📹 {video_path.name}: {total_frames} frames @ {fps} FPS (full scan)")

    records = []
    for _ in tqdm(range(total_frames), desc=f"  {video_path.stem}", unit="fr"):
        ret, frame = cap.read()
        if not ret:
            break

        # Detect Person & Bike
        results = yolo.predict(frame, classes=[0, 1], conf=0.3, verbose=False, device=device)[0]
        boxes = results.boxes.xyxy.cpu().numpy()
        clss  = results.boxes.cls.cpu().numpy()
        persons = boxes[clss == 0]
        bikes   = boxes[clss == 1]

        # Pair Cyclists
        cyclist_boxes = []
        used_bike_indices = set()
        for p in persons:
            for bi, b in enumerate(bikes):
                if bi not in used_bike_indices and calculate_iou(p, b) > 0.1:
                    cx1 = min(p[0], b[0]);  cy1 = min(p[1], b[1])
                    cx2 = max(p[2], b[2]);  cy2 = max(p[3], b[3])
                    cyclist_boxes.append([cx1, cy1, cx2, cy2])
                    used_bike_indices.add(bi)
                    break

        # Greedy NMS
        if len(cyclist_boxes) > 1:
            suppressed = set()
            for i in range(len(cyclist_boxes)):
                if i in suppressed:
                    continue
                for j in range(i + 1, len(cyclist_boxes)):
                    if j not in suppressed and calculate_iou(cyclist_boxes[i], cyclist_boxes[j]) > 0.5:
                        suppressed.add(j)
            cyclist_boxes = [b for idx, b in enumerate(cyclist_boxes) if idx not in suppressed]

        # Multi-Padding Sweep
        for box in cyclist_boxes:
            cyclist_runs = {}
            for p_val in padding_candidates:
                crop = crop_with_padding(frame, box, p_val)
                if crop.size == 0:
                    continue
                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_tensor = transform(crop_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    logits, _ = dino(crop_tensor)
                    probs = torch.softmax(logits, dim=1)
                    score, pred = torch.max(probs, dim=1)
                    cyclist_runs[str(p_val)] = {
                        "class": classes[pred.item()],
                        "confidence": score.item(),
                        "crop": crop.copy(),
                    }
            if len(cyclist_runs) == len(padding_candidates):
                records.append(cyclist_runs)

    cap.release()
    return records



def save_visualizations_for_video(records, best_padding, padding_candidates, video_name, out_dir, n_viz=10):
    """Pick n_viz evenly-spread records from a video and save visualization PNGs."""
    n = len(records)
    if n == 0:
        print(f"  ⚠️  No records for {video_name}, skipping visualizations.")
        return

    n_viz = min(n_viz, n)
    if n_viz == 1:
        viz_indices = [0]
    else:
        step = (n - 1) / (n_viz - 1)
        viz_indices = [round(i * step) for i in range(n_viz)]

    video_out_dir = out_dir / video_name
    video_out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  🎨 Saving {n_viz} visualizations for {video_name} → {video_out_dir.name}/")
    for rank, idx in enumerate(viz_indices):
        sample = records[idx]
        out_path = video_out_dir / f"sample_{rank + 1:02d}_of_{n_viz}.png"
        try:
            save_sample_visualization(sample, best_padding, padding_candidates, out_path)
            print(f"    [{rank + 1:2d}/{n_viz}] record #{idx:4d} → {out_path.name}")
        except Exception as ex:
            print(f"    ⚠️  sample {rank + 1} (record #{idx}): {ex}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="DINOv3 Crop Padding Optimization — sweep padding ratios across one or more videos."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        metavar="VIDEO",
        help="Path(s) to input video file(s). Example: --input data/v1.mp4 data/v2.mp4",
    )
    args = parser.parse_args()

    video_paths = [Path(p) for p in args.input]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" DINOv3 CROP SIZE OPTIMIZATION PIPELINE ".center(60, "■"))
    print("=" * 60)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Videos to process:")
    for vp in video_paths:
        status = "✅" if vp.exists() else "❌ NOT FOUND"
        print(f"        {status} {vp}")

    valid_paths = [vp for vp in video_paths if vp.exists()]
    if not valid_paths:
        print("\n❌ No valid video files found. Exiting.")
        sys.exit(1)


    out_dir = project_root / "output" / "optimize_crops"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load models ──────────────────────────────────────────────────────────────
    print("\n[INFO] Loading YOLO detector...")
    yolo_weights = project_root / "models" / "weights" / "yolo26n.pt"
    yolo = YOLO(str(yolo_weights) if yolo_weights.exists() else "yolo26n.pt")
    if device.type == "cuda":
        yolo.to(device)

    print("[INFO] Loading DINOv3 Classifier Head...")
    dino_weights = project_root / "models" / "weights" / "dinov3_lightweight_classifier.pth"
    dino = LightweightDINOInference().to(device)
    if dino_weights.exists():
        checkpoint = torch.load(str(dino_weights), map_location=device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        dino.load_state_dict(state_dict, strict=False)
        print("✅ Loaded DINOv3 fine-tuned weights successfully.")
    else:
        print("⚠️  Warning: DINOv3 fine-tuned weights not found.")
    dino.eval()

    model_cfg = timm.data.resolve_model_data_config(dino.backbone)
    transform  = timm.data.create_transform(**model_cfg, is_training=False)

    # ── Sweep config ─────────────────────────────────────────────────────────────
    padding_candidates = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    classes = ["Riding", "Pushing"]

    # ── Per-video scan + visualizations ──────────────────────────────────────────
    all_records = []          # aggregated across all videos for joint optimization
    per_video_records = {}    # {video_stem: list_of_records}

    for vp in valid_paths:

        print(f"\n{'─' * 60}")
        print(f"🎬 Processing: {vp.name}")
        print(f"{'─' * 60}")

        records = scan_video(vp, yolo, dino, transform, device, padding_candidates, classes)
        print(f"  ✅ Gathered {len(records)} cyclist samples from {vp.name}")

        per_video_records[vp.stem] = records
        all_records.extend(records)

    # ── Joint optimization across all videos ─────────────────────────────────────
    n_samples = len(all_records)
    print(f"\n{'=' * 60}")
    print(f"[INFO] Total samples from all videos: {n_samples}")

    if n_samples == 0:
        print("❌ No valid cyclist samples gathered. Optimization aborted.")
        return

    # Compute consensus vote per record
    for record in all_records:
        class_votes = [record[str(p)]["class"] for p in padding_candidates]
        record["consensus"] = max(set(class_votes), key=class_votes.count)

    padding_stats = {}
    for p_val in padding_candidates:
        p_str = str(p_val)
        confidences = [r[p_str]["confidence"] for r in all_records]
        flips       = sum(1 for r in all_records if r[p_str]["class"] != r["consensus"])
        mean_conf  = float(np.mean(confidences))
        std_conf   = float(np.std(confidences))
        flip_rate  = flips / n_samples
        perf_score = mean_conf * (1.0 - flip_rate)
        padding_stats[p_str] = {
            "mean_confidence":  round(mean_conf,  4),
            "std_confidence":   round(std_conf,   4),
            "flip_rate":        round(flip_rate,  4),
            "stability_rate":   round(1.0 - flip_rate, 4),
            "performance_score": round(perf_score, 4),
        }

    best_padding = max(padding_stats, key=lambda k: padding_stats[k]["performance_score"])
    best_stats   = padding_stats[best_padding]

    # Save JSON report
    report = {
        "metadata": {
            "videos": [vp.name for vp in video_paths if vp.exists()],
            "total_samples_analyzed": n_samples,
            "samples_per_video": {stem: len(recs) for stem, recs in per_video_records.items()},
            "padding_candidates_swept": padding_candidates,
        },
        "optimal_padding_result": {
            "optimal_padding_ratio":   float(best_padding),
            "optimal_padding_percent": f"{float(best_padding)*100:.1f}%",
            "mean_confidence_percent": f"{best_stats['mean_confidence']*100:.2f}%",
            "stability_rate_percent":  f"{best_stats['stability_rate']*100:.2f}%",
            "performance_score":       best_stats["performance_score"],
        },
        "sweep_statistics": padding_stats,
    }
    report_path = out_dir / "crop_optimization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False)

    # ── Combined optimization curve (all videos pooled) ───────────────────────────
    try:
        import matplotlib.pyplot as plt

        p_ratios    = [float(p) for p in padding_stats]
        confs       = [padding_stats[str(p)]["mean_confidence"]  for p in p_ratios]
        stabilities = [padding_stats[str(p)]["stability_rate"]   for p in p_ratios]
        scores      = [padding_stats[str(p)]["performance_score"] for p in p_ratios]

        plt.figure(figsize=(10, 6))
        plt.plot(p_ratios, confs,       marker="o", color="#10B981", linewidth=2.5, label="Mean Confidence")
        plt.plot(p_ratios, stabilities, marker="s", color="#F59E0B", linewidth=2, linestyle="--", label="Stability Rate")
        plt.plot(p_ratios, scores,      marker="D", color="#3B82F6", linewidth=3,  label="Combined Performance Score")
        opt_p     = float(best_padding)
        opt_score = best_stats["performance_score"]
        plt.scatter([opt_p], [opt_score], color="#EF4444", s=150, zorder=5,
                    label=f"Optimal Padding ({opt_p*100:.0f}%)")

        n_videos = len(per_video_records)
        plt.title(
            f"DINOv3 Crop Padding Optimization — {n_videos} videos pooled ({n_samples} samples)",
            fontsize=13, fontweight="bold", pad=14,
        )
        plt.xlabel("Crop Padding Ratio (Margin % of width/height)", fontsize=12)
        plt.ylabel("Performance Metrics", fontsize=12)
        plt.xticks(p_ratios, [f"{p*100:.0f}%" for p in p_ratios])
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4)
        plt.tight_layout()

        plot_path = out_dir / "optimization_curve.png"
        plt.savefig(str(plot_path), bbox_inches="tight", dpi=300)
        plt.close()
        print(f"\n📊 Combined optimization curve → {plot_path}")
    except Exception as e:
        print(f"⚠️  Could not plot optimization curve: {e}")

    # ── Per-video visualizations (10 per video = 30 total) ───────────────────────
    print(f"\n🎨 Generating 10 visualizations per video (30 total)...")
    for stem, records in per_video_records.items():
        save_visualizations_for_video(
            records, best_padding, padding_candidates,
            video_name=stem, out_dir=out_dir, n_viz=10,
        )

    # ── Console summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" DINOv3 CROP OPTIMIZATION SUCCESSFUL ".center(60, "■"))
    print("=" * 60)
    print(f"👉 JSON Report          : {report_path}")
    print(f"🎯 Optimal Padding      : {float(best_padding)*100:.1f}%")
    print(f"📈 Mean Confidence      : {best_stats['mean_confidence']*100:.2f}%")
    print(f"📈 Stability Rate       : {best_stats['stability_rate']*100:.2f}%")
    print("-" * 60)
    print("Sweep details (all videos pooled):")
    for p, stats in padding_stats.items():
        print(f"  * Padding {float(p)*100:4.1f}% | Conf: {stats['mean_confidence']*100:5.2f}% "
              f"| Stability: {stats['stability_rate']*100:5.2f}% | Score: {stats['performance_score']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
