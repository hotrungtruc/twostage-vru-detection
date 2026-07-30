import os
import sys
import cv2
import json
import queue
import threading
from pathlib import Path

# Add src to sys.path to import submodules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # src
if project_root not in sys.path:
    sys.path.append(project_root)


def _default_json_output_path(output_path):
    return str(Path(output_path).with_suffix(".json"))


def process_video(video_path, output_path, json_output_path=None, device=None):
    from src.inference.inference_pipeline import TrafficInferencePipeline
    print(f"🚀 Starting video processing (Multi-threaded Pipeline): {video_path}")
    
    # Initialize OCSort Pipeline
    pipeline = TrafficInferencePipeline(device=device)
    classes = pipeline.classes
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return

    # Output video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 Video info: {w}x{h} @ {fps} FPS, Total frames: {frame_count}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if json_output_path is None:
        json_output_path = _default_json_output_path(output_path)
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Bounded queues to manage memory consumption
    frame_queue = queue.Queue(maxsize=30)
    yolo_queue = queue.Queue(maxsize=30)
    tracker_queue = queue.Queue(maxsize=30)

    # Worker 1: Frame Reading Thread
    def reader_worker():
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put((count, frame))
            count += 1
        frame_queue.put(None)  # Sentinel for exit

    # Worker 2: YOLO Detection Thread
    def yolo_worker():
        while True:
            item = frame_queue.get()
            if item is None:
                yolo_queue.put(None)
                break
            count, frame = item
            
            # Check for frame skipping based on config
            if pipeline.frame_skip_interval > 1 and count % pipeline.frame_skip_interval != 0:
                yolo_queue.put((count, frame, None, None, None, True))
            else:
                cyclist_boxes, cyclist_sources, raw_detections = pipeline.detect_yolo(frame)
                yolo_queue.put((count, frame, cyclist_boxes, cyclist_sources, raw_detections, False))

    # Worker 3: DINO Classification + OC-SORT Tracker Thread
    def tracker_worker():
        while True:
            item = yolo_queue.get()
            if item is None:
                tracker_queue.put(None)
                break
            count, frame, cyclist_boxes, cyclist_sources, raw_detections, skipped = item
            
            if skipped:
                pipeline.tracker.predict_only()
                active_tracks = pipeline.tracker.active_tracks
                frame_payload = {
                    "frame_idx": int(count),
                    "raw_detections": [],
                    "cyclist_detections": [],
                    "tracks": [pipeline._track_to_dict(track) for track in pipeline.tracker.active_tracks],
                    "skipped": True
                }
            else:
                active_tracks, frame_payload = pipeline.classify_and_track(
                    frame, cyclist_boxes, cyclist_sources, raw_detections, count
                )
                
            # Create a snapshot of current frame tracking data for thread-safe drawing
            tracks_snapshot = []
            for track in active_tracks:
                tracks_snapshot.append({
                    "id": int(track.id),
                    "bbox_xyxy": track.bbox_xyxy.copy(),
                    "class_id": int(track.class_id)
                })
                
            tracker_queue.put((count, frame, tracks_snapshot, frame_payload))

    # Start background threads
    t1 = threading.Thread(target=reader_worker, name="ReaderThread")
    t2 = threading.Thread(target=yolo_worker, name="YoloThread")
    t3 = threading.Thread(target=tracker_worker, name="TrackerThread")
    
    t1.start()
    t2.start()
    t3.start()

    # Worker 4: Draw BBox and Write Video (Runs on Main Thread for cv2 UI safety)
    json_frames = []
    processed_count = 0
    
    while True:
        item = tracker_queue.get()
        if item is None:
            break
            
        count, frame, tracks_snapshot, frame_payload = item
        json_frames.append(frame_payload)
        
        # Draw Frame info on top left
        frame_text = f"Frame: {count}"
        cv2.putText(frame, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4)
        cv2.putText(frame, frame_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Draw Bounding Box and Label on frame
        for t in tracks_snapshot:
            x1, y1, x2, y2 = map(int, t["bbox_xyxy"])
            label = pipeline.classes[t["class_id"]]
            
            # Green for Riding, Orange for Pushing
            color = (0, 255, 0) if label.lower() == 'riding' else (0, 165, 255)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw text background for readability
            label_text = f"ID:{t['id']} {label}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_w, y1), color, -1)
            
            # Draw text
            cv2.putText(frame, label_text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Write video
        out.write(frame)
        
        # Display frame on screen (optional)
        display_frame = cv2.resize(frame, (1280, 720)) if w > 1280 else frame
        cv2.imshow("DINOv3 OCSORT Cyclist Tracking", display_frame)
        
        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ User aborted the processing.")
            break
            
        processed_count += 1
        if processed_count % 30 == 0:
            print(f"⏳ Written {processed_count}/{frame_count} frames...")

    # Wait for child threads to finish
    t1.join()
    t2.join()
    t3.join()
    
    # Free up memory
    cap.release()
    out.release()
    cv2.destroyAllWindows()

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

    print(f"✅ Complete! Video saved to: {output_path}")
    print(f"✅ JSON result saved to: {json_output_path}")
    return {
        "video_path": output_path,
        "json_path": json_output_path,
        "frame_count": count,
    }

if __name__ == "__main__":
    import sys
    import os
    from pathlib import Path
    workspace_root = Path(__file__).resolve().parents[2]
    
    video_p = workspace_root / "data" / "input.mp4"
    VIDEO_INPUT = str(video_p)
    
    if not os.path.exists(VIDEO_INPUT):
        print(f"[ERROR] Input video file not found at: {VIDEO_INPUT}. Please place your video file there.")
        sys.exit(1)
        
    output_dir = workspace_root / "output" / "result_dinov3"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_p = output_dir / "tracking_result.mp4"
    OUTPUT_VIDEO = str(output_p)
    
    process_video(VIDEO_INPUT, OUTPUT_VIDEO)
