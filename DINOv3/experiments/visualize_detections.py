#!/usr/bin/env python3
"""
visualize_detections.py
=======================
Overlay IoU-detection JSON results onto the original video.

Draws per frame:
  - Blue  box  : person bounding box
  - Green box  : bicycle bounding box
  - Red   box  : merged cyclist bounding box  (person ∩ bicycle)
  - IoU value  : displayed next to cyclist box
  - HUD overlay: frame index, counts, IoU threshold

Usage (from project root):
    python experiments/visualize_detections.py \\
        --json  output/optimize_iou_20260604_095617/detections/00041_iou0.010_detections.json \\
        --video data/00041.mp4

    # Custom output path:
    python experiments/visualize_detections.py \\
        --json  output/.../00041_iou0.010_detections.json \\
        --video data/00041.mp4 \\
        --output output/00041_viz.mp4

    # Faster: only write frames that contain at least 1 cyclist
    python experiments/visualize_detections.py \\
        --json  output/.../00041_iou0.010_detections.json \\
        --video data/00041.mp4 \\
        --cyclist_only
"""

import os, sys, json, argparse
from pathlib import Path
from tqdm import tqdm

# Force UTF-8 output on Windows
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import cv2
import numpy as np

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
CLR_PERSON   = (220, 120,  30)   # blue-ish orange
CLR_BIKE     = ( 40, 200,  40)   # green
CLR_CYCLIST  = ( 30,  30, 220)   # red
CLR_HUD_BG   = (255, 255, 255)
CLR_HUD_TXT  = ( 20,  20,  20)

FONT         = cv2.FONT_HERSHEY_SIMPLEX
THICKNESS    = 2

# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_box(img, bbox, color, label: str = "", thickness: int = THICKNESS):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), baseline = cv2.getTextSize(label, FONT, 0.52, 1)
        # Background pill behind text
        cv2.rectangle(img, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - baseline - 2),
                    FONT, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def draw_hud(img, frame_idx: int, n_persons: int, n_bikes: int,
             n_cyclists: int, iou_thresh: float):
    """Top-left HUD panel."""
    lines = [
        f"Frame : {frame_idx}",
        f"Persons : {n_persons}",
        f"Bikes   : {n_bikes}",
        f"Cyclists: {n_cyclists}",
        f"IoU thr : {iou_thresh:.3f}",
    ]
    pad, line_h = 8, 22
    panel_w = 200
    panel_h = len(lines) * line_h + pad * 2

    # semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (240, 240, 240), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    for i, txt in enumerate(lines):
        color = CLR_HUD_TXT
        if i == 3 and n_cyclists > 0:
            color = (0, 0, 200)   # red when cyclists detected
        cv2.putText(img, txt, (18, 10 + pad + (i + 1) * line_h - 4),
                    FONT, 0.52, color, 1, cv2.LINE_AA)


def compute_merged_box(person_bbox, bike_bbox):
    """Merge person + bike bbox exactly as inference_pipeline does."""
    x1 = min(person_bbox[0], bike_bbox[0])
    y1 = min(person_bbox[1], bike_bbox[1])
    x2 = max(person_bbox[2], bike_bbox[2])
    y2 = max(person_bbox[3], bike_bbox[3])
    return [x1, y1, x2, y2]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overlay IoU detection JSON results onto the original video."
    )
    parser.add_argument("--json",   required=True,
                        help="Path to *_detections.json produced by optimize_iou_threshold.py")
    parser.add_argument("--video",  required=True,
                        help="Path to the original source video (mp4)")
    parser.add_argument("--output", default=None,
                        help="Output video path. Default: same folder as JSON, auto-named.")
    parser.add_argument("--cyclist_only", action="store_true", default=False,
                        help="Only include frames that contain >= 1 cyclist pair.")
    parser.add_argument("--show_person_bike", action="store_true", default=True,
                        help="Draw individual person / bike boxes (default: True)")
    parser.add_argument("--no_person_bike", dest="show_person_bike", action="store_false")
    args = parser.parse_args()

    json_path  = Path(args.json)
    video_path = Path(args.video)

    if not json_path.exists():
        print(f"[ERROR] JSON not found: {json_path}")
        sys.exit(1)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}")
        sys.exit(1)

    # ── Load JSON ─────────────────────────────────────────────────────────────
    print(f"[INFO] Loading detections from: {json_path.name}")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    iou_threshold = data.get("iou_threshold", 0.0)
    frames_data   = data.get("frames", [])
    # Build lookup: frame_idx → record
    frame_lookup = {rec["frame_idx"]: rec for rec in frames_data}
    detected_frame_indices = set(frame_lookup.keys())

    # ── Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    W            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video: {total_frames} frames @ {fps:.1f} fps  ({W}x{H})")

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = "_cyclist_only" if args.cyclist_only else ""
        out_path = json_path.parent.parent / f"{video_path.stem}_iou{iou_threshold:.3f}{suffix}_viz.mp4"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
    print(f"[INFO] Writing output to: {out_path}")

    # ── Process ───────────────────────────────────────────────────────────────
    frame_idx = 0
    frames_written = 0
    n_cyclist_frames = 0

    progress = tqdm(total=total_frames, desc="Rendering", unit="fr")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in detected_frame_indices:
            rec        = frame_lookup[frame_idx]
            n_persons  = rec["n_persons"]
            n_bikes    = rec["n_bikes"]
            n_cyclists = rec["n_cyclists"]
            pairs      = rec["cyclist_pairs"]

            # Skip non-cyclist frames if --cyclist_only
            if args.cyclist_only and n_cyclists == 0:
                frame_idx += 1
                progress.update(1)
                continue

            if n_cyclists > 0:
                n_cyclist_frames += 1

            # Draw person + bike boxes from pair info
            drawn_persons = set()
            drawn_bikes   = set()

            for pair in pairs:
                pb = pair["person_bbox"]
                bb = pair["bike_bbox"]
                iou_val = pair.get("iou", 0.0)

                # Merged cyclist box
                cb = compute_merged_box(pb, bb)
                draw_box(frame, cb, CLR_CYCLIST,
                         label=f"Cyclist  IoU={iou_val:.2f}")

                if args.show_person_bike:
                    # Person (dashed-style: draw inner + outer)
                    draw_box(frame, pb, CLR_PERSON, label="Person",  thickness=1)
                    # Bike
                    draw_box(frame, bb, CLR_BIKE,   label="Bicycle", thickness=1)

            # HUD
            draw_hud(frame, frame_idx, n_persons, n_bikes, n_cyclists, iou_threshold)

        else:
            # Frame was skipped during detection (frame_step > 1) — draw minimal HUD
            draw_hud(frame, frame_idx, 0, 0, 0, iou_threshold)

        writer.write(frame)
        frames_written += 1
        frame_idx += 1
        progress.update(1)

    progress.close()
    cap.release()
    writer.release()

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Output video   : {out_path}")
    print(f"  Frames written : {frames_written} / {total_frames}")
    print(f"  Cyclist frames : {n_cyclist_frames}")
    print(f"  IoU threshold  : {iou_threshold}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
