#!/usr/bin/env python3
"""
Analyze tracking output JSONs for label flips (e.g., walking<->riding) and overlapping detections (duplicates).

Usage:
  python analyze_tracking_quality.py --input-dir ../DINOv3/output/result_dinov3/result_dinov3 --out-dir report

Generates a JSON report with statistics per file and aggregated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Dict, List, Tuple

def iou(boxA, boxB):
    # boxes: [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    inter = interW * interH
    areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = areaA + areaB - inter
    if union <= 0.0:
        return 0.0
    return inter / union

def compress_label_segments(sequence: List[Tuple[int,str]]) -> List[Tuple[str,int,int]]:
    # sequence: list of (frame_idx,label) sorted by frame_idx
    if not sequence:
        return []
    segments = []
    cur_label = sequence[0][1]
    start = sequence[0][0]
    end = start
    for f,l in sequence[1:]:
        if l == cur_label:
            end = f
        else:
            segments.append((cur_label, start, end))
            cur_label = l
            start = f
            end = f
    segments.append((cur_label, start, end))
    return segments


def _norm_label(value) -> str:
    return str(value).strip().lower()


def _looks_like_raw_tracking(data: dict) -> bool:
    return isinstance(data, dict) and isinstance(data.get('frames'), list)


def _looks_like_aggregated_report(data: dict) -> bool:
    return isinstance(data, dict) and isinstance(data.get('files'), list)


def _make_report_from_report_entry(entry: dict, parent_path: Path) -> dict:
    fps = float(entry.get('fps', 30.0) or 30.0)
    flips = entry.get('flip_examples', []) or []

    normalized_flips = []
    flip_durations = []
    for f in flips:
        frames = f.get('frames', [None, None])
        start = frames[0] if isinstance(frames, (list, tuple)) and len(frames) > 0 else None
        end = frames[1] if isinstance(frames, (list, tuple)) and len(frames) > 1 else None
        dur_frames = f.get('duration_frames')
        if dur_frames is None and start is not None and end is not None:
            dur_frames = int(end) - int(start) + 1
        if dur_frames is None:
            dur_frames = 0
        dur_s = f.get('duration_s')
        if dur_s is None:
            dur_s = float(dur_frames) / fps if fps else 0.0
        normalized = {
            'track_id': f.get('track_id'),
            'from': f.get('from'),
            'via': f.get('via'),
            'frames': [start, end],
            'duration_frames': int(dur_frames),
            'duration_s': float(dur_s),
        }
        normalized_flips.append(normalized)
        flip_durations.append(int(dur_frames))

    overlap = entry.get('overlap', {}) or {}
    return {
        'file': entry.get('file', str(parent_path)),
        'fps': fps,
        'total_frames': int(entry.get('total_frames', 0) or 0),
        'n_tracks': int(entry.get('n_tracks', 0) or 0),
        'n_flips': int(entry.get('n_flips', len(normalized_flips)) or 0),
        'flip_examples': normalized_flips[:20],
        'flip_stats': {
            'count': len(flip_durations),
            'mean_frames': mean(flip_durations) if flip_durations else 0,
            'median_frames': median(flip_durations) if flip_durations else 0,
            'mean_seconds': (mean(flip_durations) / fps) if flip_durations else 0,
        },
        'overlap': {
            'frames_with_overlap': int(overlap.get('frames_with_overlap', 0) or 0),
            'overlap_rate': float(overlap.get('overlap_rate', 0.0) or 0.0),
            'overlap_frame_examples': overlap.get('overlap_frame_examples', []) or [],
        },
    }


def _analyze_raw_tracking_data(data: dict, source_path: Path, iou_thresh=0.5) -> dict:
    fps = float(data.get('video_info', {}).get('fps', 30.0) or 30.0)
    frames = data.get('frames', [])

    # build per-track label timeline
    track_timelines: Dict[int, List[Tuple[int, str]]] = {}
    # overlaps per frame
    overlap_frames = []

    for frame_payload in frames:
        fi = int(frame_payload.get('frame_idx', -1))
        dets = frame_payload.get('cyclist_detections', [])
        tracks = frame_payload.get('tracks', [])

        # detect overlaps among cyclist_detections (fallback to tracks if missing)
        boxes = []
        if dets:
            for d in dets:
                b = d.get('bbox_xyxy')
                if isinstance(b, list) and len(b) == 4:
                    boxes.append(b)
        else:
            for t in tracks:
                b = t.get('bbox_xyxy')
                if isinstance(b, list) and len(b) == 4:
                    boxes.append(b)

        overlaps = 0
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if iou(boxes[i], boxes[j]) > iou_thresh:
                    overlaps += 1
        if overlaps > 0:
            overlap_frames.append(fi)

        for t in tracks:
            tid_raw = t.get('id')
            if tid_raw is None:
                continue
            tid = int(tid_raw)
            label = t.get('label') or str(t.get('class_id'))
            track_timelines.setdefault(tid, []).append((fi, _norm_label(label)))

    # analyze flips per track
    flips = []
    flip_durations = []
    for tid, seq in track_timelines.items():
        seq_sorted = sorted(seq, key=lambda x: x[0])
        segments = compress_label_segments(seq_sorted)

        # look for label A -> B -> A patterns
        for i in range(len(segments)-2):
            a, a_s, a_e = segments[i]
            b, b_s, b_e = segments[i+1]
            a2, a2_s, a2_e = segments[i+2]
            if a == a2 and a != b:
                flips.append({
                    'track_id': tid,
                    'from': a,
                    'via': b,
                    'frames': [b_s, b_e],
                    'duration_frames': b_e - b_s + 1,
                    'duration_s': (b_e - b_s + 1) / fps,
                })
                flip_durations.append(b_e - b_s + 1)

    total_frames = frames[-1]['frame_idx'] + 1 if frames else 0
    overlap_rate = len(overlap_frames) / total_frames if total_frames > 0 else 0.0

    return {
        'file': str(source_path),
        'fps': fps,
        'total_frames': total_frames,
        'n_tracks': len(track_timelines),
        'n_flips': len(flips),
        'flip_examples': flips[:20],
        'flip_stats': {
            'count': len(flip_durations),
            'mean_frames': mean(flip_durations) if flip_durations else 0,
            'median_frames': median(flip_durations) if flip_durations else 0,
            'mean_seconds': mean(flip_durations) / fps if flip_durations else 0,
        },
        'overlap': {
            'frames_with_overlap': len(set(overlap_frames)),
            'overlap_rate': overlap_rate,
            'overlap_frame_examples': sorted(list(set(overlap_frames)))[:20],
        },
    }


def analyze_file(path: Path, iou_thresh=0.5):
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    if _looks_like_raw_tracking(data):
        return [_analyze_raw_tracking_data(data, path, iou_thresh=iou_thresh)]

    if _looks_like_aggregated_report(data):
        return [_make_report_from_report_entry(entry, path) for entry in data.get('files', [])]

    # A single report-like JSON entry without top-level "files"
    if isinstance(data, dict) and ('flip_examples' in data or 'n_flips' in data):
        return [_make_report_from_report_entry(data, path)]

    raise ValueError(f'Unsupported JSON format in: {path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'output')
    parser.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'output' / 'tracking_quality_report')
    parser.add_argument('--out-json', type=Path, default=None)
    parser.add_argument('--flip-from', type=str, default=None, help='Label A in A->B->A')
    parser.add_argument('--flip-to', type=str, default=None, help='Label B in A->B->A')
    parser.add_argument('--bidirectional', action='store_true', help='Match both A->B->A and B->A->B')
    parser.add_argument('--iou-thresh', type=float, default=0.8)
    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out_json is None:
        out_json = out_dir / 'tracking_analysis_filtered.json'
    else:
        out_json = args.out_json

    if not input_dir.exists():
        print('Input dir not found:', input_dir)
        return 1

    files = sorted(input_dir.rglob('*.json'))
    all_reports = []
    all_flips = []
    flip_from_norm = _norm_label(args.flip_from) if args.flip_from else None
    flip_to_norm = _norm_label(args.flip_to) if args.flip_to else None

    for p in files:
        print('Analyzing', p)
        try:
            reports = analyze_file(p, iou_thresh=args.iou_thresh)
            for r in reports:
                flips = r.get('flip_examples', [])
                filtered_flips = []
                for f in flips:
                    from_label = _norm_label(f.get('from', ''))
                    via_label = _norm_label(f.get('via', ''))
                    if flip_from_norm and flip_to_norm:
                        if args.bidirectional:
                            cond = (
                                (from_label == flip_from_norm and via_label == flip_to_norm)
                                or (from_label == flip_to_norm and via_label == flip_from_norm)
                            )
                            if cond:
                                filtered_flips.append(f)
                        else:
                            if from_label == flip_from_norm and via_label == flip_to_norm:
                                filtered_flips.append(f)
                    else:
                        filtered_flips.append(f)

                source_file = r.get('file', str(p))
                for ff in filtered_flips:
                    ff['source_file'] = source_file
                all_flips.extend(filtered_flips)
                r['filtered_flips'] = filtered_flips
                all_reports.append(r)
        except Exception as e:
            print('Error processing', p, e)

    # write aggregated JSON
    agg = {
        'n_files': len(all_reports),
        'total_filtered_flips': len(all_flips),
        'total_tracks': sum(r['n_tracks'] for r in all_reports),
        'files': all_reports,
    }
    with out_json.open('w', encoding='utf-8') as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)

    # write CSV detailed
    import csv
    csv_path = out_dir / 'flips_detail.csv'
    with csv_path.open('w', encoding='utf-8', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(['source_file','track_id','from','via','start_frame','end_frame','duration_frames','duration_seconds'])
        for ff in all_flips:
            writer.writerow([
                ff.get('source_file',''),
                ff.get('track_id',''),
                ff.get('from',''),
                ff.get('via',''),
                ff.get('frames',('',''))[0],
                ff.get('frames',('',''))[1],
                ff.get('duration_frames',''),
                ff.get('duration_s',''),
            ])

    # simple plots: duration histogram and counts per file
    try:
        import matplotlib.pyplot as plt
        durations = [f['duration_s'] for f in all_flips if 'duration_s' in f]
        if durations:
            plt.figure()
            plt.hist(durations, bins=20)
            plt.xlabel('Flip duration (s)')
            plt.ylabel('Count')
            plt.title('Flip durations')
            plt.tight_layout()
            plt.savefig(out_dir / 'flip_duration_hist.png')
            plt.close()

        # counts per file
        from collections import Counter
        counts = Counter([f['source_file'] for f in all_flips])
        if counts:
            files_n = list(counts.keys())
            vals = [counts[k] for k in files_n]
            plt.figure(figsize=(8, max(2, len(files_n)*0.4)))
            plt.barh(files_n, vals)
            plt.xlabel('Flip count')
            plt.title('Flip counts by file')
            plt.tight_layout()
            plt.savefig(out_dir / 'flip_counts_by_file.png')
            plt.close()
    except Exception as e:
        print('Plotting failed (matplotlib missing?):', e)

    print('Wrote aggregated JSON to', out_json)
    print('Wrote CSV to', csv_path)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
