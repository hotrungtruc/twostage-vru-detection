#!/usr/bin/env python3
"""
Run DINOv3 standard cyclist tracking pipeline.

Usage:
    # Run standard OCSort cyclist tracking:
    python main.py --input data/input.mp4 --output output/tracking_result.mp4
"""

import argparse
import logging
import os
import sys
import torch
from pathlib import Path
import cv2

# Patch cv2 GUI functions if running in Docker (headless mode)
if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(args):
    # Redirect output to output/result_dinov3
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
        
    args.output = str(new_output_path)

    logger.info("Initializing DINOv3 Cyclist Tracking...")
    logger.info(f"Input path: {args.input}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Device: {args.device}")

    # Check if input video exists
    if not os.path.exists(args.input):
        logger.error(f"Input video file not found at: '{args.input}'. Please place your video file there or specify its path using '--input <path_to_video>'.")
        sys.exit(1)

    # Set device environment if needed (the pipelines will resolve torch device)
    if args.device and "cuda" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.replace("cuda:", "")

    from src.core.pipeline import process_video
    logger.info("Running in standard tracking mode...")
    process_video(args.input, args.output, device=args.device)

    logger.info("Pipeline processing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unified DINOv3 Cyclist Tracking CLI"
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
        default="output/result_dinov3/tracking_result.mp4",
        help="Output path for video results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda:0, cpu, etc.)"
    )
    
    args = parser.parse_args()
    main(args)
