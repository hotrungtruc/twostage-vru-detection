import argparse
import logging
import os
import sys
import torch
from pathlib import Path
import cv2

# Patch cv2 GUI functions if running in Docker
if os.environ.get("QT_QPA_PLATFORM") == "offscreen":
    cv2.imshow = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: -1
    cv2.destroyAllWindows = lambda *a, **k: None

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
custom_classes_dir = current_dir.parent
project_root = custom_classes_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from pipeline import process_video_dynamic

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(args):
    output_path = Path(args.output)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Initializing Dynamic DINOv3 Inference...")
    logger.info(f"Input path: {args.input}")
    logger.info(f"Output path: {args.output}")
    logger.info(f"Device: {args.device}")
    
    if args.weights:
        logger.info(f"DINO Weights: {args.weights}")

    if not os.path.exists(args.input):
        logger.error(f"Input video file not found at: '{args.input}'")
        sys.exit(1)

    if args.device and "cuda" in args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.replace("cuda:", "")

    logger.info("Running in dynamic tracking mode...")
    process_video_dynamic(args.input, args.output, dino_weights=args.weights, device=args.device, class_thresh=args.class_thresh)

    logger.info("Pipeline processing completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Custom DINOv3 Tracking CLI")
    parser.add_argument("--input", type=str, required=True, help="Input video path")
    parser.add_argument("--output", type=str, default=str(custom_classes_dir / "output_inference" / "tracking_result.mp4"), help="Output filename or path")
    parser.add_argument("--weights", type=str, default=None, help="Path to custom DINOv3 classifier weights")
    parser.add_argument("--class-thresh", type=float, default=0.60, help="Confidence threshold for DINOv3 (default: 0.60)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    
    args = parser.parse_args()
    main(args)
