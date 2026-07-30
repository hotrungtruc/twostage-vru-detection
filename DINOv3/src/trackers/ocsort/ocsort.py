from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.core.detector import Detection

LARGE_COST = 1e6


# ── Geometry helpers (inlined from src.core.geometry) ──────────────────────────

def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


# ── Kalman Filter ──────────────────────────────────────────────────────────────

def kf_predict(mean: np.ndarray, covariance: np.ndarray):
    """Simple constant velocity Kalman Filter prediction step."""
    # State: [x1, y1, x2, y2, dx1, dy1, dx2, dy2]
    # Transition matrix F
    F = np.eye(8)
    for i in range(4):
        F[i, i + 4] = 1

    # Process noise Q (simple diagonal)
    Q = np.eye(8) * 0.01
    Q[4:, 4:] *= 0.1

    new_mean = F @ mean
    new_covariance = F @ covariance @ F.T + Q
    return new_mean, new_covariance


def kf_update(mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
    """Simple Kalman Filter update step."""
    # Measurement: [x1, y1, x2, y2]
    # Measurement matrix H
    H = np.zeros((4, 8))
    H[:4, :4] = np.eye(4)

    # Measurement noise R
    R = np.eye(4) * 0.1

    y = measurement - (H @ mean)
    S = H @ covariance @ H.T + R
    K = covariance @ H.T @ np.linalg.inv(S)

    new_mean = mean + (K @ y)
    new_covariance = (np.eye(8) - (K @ H)) @ covariance
    return new_mean, new_covariance


# ── Track ──────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class Track:
    id: int
    bbox_xyxy: np.ndarray
    score: float
    class_id: int
    age: int = 1
    hits: int = 1
    time_since_update: int = 0
    history_timestamps: list[int] = field(default_factory=list)
    history_boxes: list[list[float]] = field(default_factory=list)

    # KF state
    mean: np.ndarray = field(init=False)
    covariance: np.ndarray = field(init=False)

    # OC-SORT specific
    last_observation: np.ndarray = field(init=False)
    observations: list[np.ndarray | None] = field(default_factory=list)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4))

    def __post_init__(self):
        # Initialize KF state
        self.mean = np.zeros(8)
        self.mean[:4] = self.bbox_xyxy
        self.covariance = np.eye(8) * 10.0
        self.covariance[4:, 4:] *= 100.0
        self.last_observation = self.bbox_xyxy.copy()
        self.observations.append(self.bbox_xyxy.copy())

    def predict(self) -> None:
        self.mean, self.covariance = kf_predict(self.mean, self.covariance)
        self.bbox_xyxy = self.mean[:4].copy()
        self.age += 1
        self.time_since_update += 1
        self.observations.append(None)

    def update(self, detection: Detection, timestamp: int) -> None:
        measurement = detection.bbox_xyxy.astype(np.float32)

        # Update velocity (momentum)
        if self.time_since_update > 0 and self.last_observation is not None:
            # OC-SORT: velocity = (current_obs - last_obs) / time_delta
            self.velocity = (measurement - self.last_observation) / self.time_since_update

        self.mean, self.covariance = kf_update(self.mean, self.covariance, measurement)
        self.bbox_xyxy = self.mean[:4].copy()
        self.score = float(detection.score)
        self.class_id = int(detection.class_id)
        self.hits += 1
        self.time_since_update = 0
        self.last_observation = measurement.copy()
        self.observations[-1] = measurement.copy()

        self.history_timestamps.append(int(timestamp))
        self.history_boxes.append(self.bbox_xyxy.astype(float).tolist())

    def to_output(self, round_digits: int) -> dict[str, object]:
        boxes = [
            [round(float(value), round_digits) for value in box]
            for box in self.history_boxes
        ]
        return {
            "id": int(self.id),
            "class_id": self.class_id,
            "timestamps": [int(timestamp) for timestamp in self.history_timestamps],
            "boxes": boxes,
        }


# ── OCSortTracker ──────────────────────────────────────────────────────────────

class OCSortTracker:
    def __init__(
        self,
        det_thresh: float = 0.5,
        iou_thresh: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        inertia: float = 0.2,
    ) -> None:
        self.det_thresh = float(det_thresh)
        self.iou_thresh = float(iou_thresh)
        self.max_age = int(max_age)
        self.min_hits = int(min_hits)
        self.inertia = float(inertia)

        self._next_track_id = 1
        self.active_tracks: list[Track] = []
        self.completed_tracks: list[Track] = []

    def update(self, detections: Sequence[Detection], timestamp: int) -> None:
        # 1. Predict
        for track in self.active_tracks:
            track.predict()

        # 2. Filter detections
        valid_dets = [d for d in detections if d.score >= self.det_thresh]

        # 3. First association (IoU + OCM)
        matches, unmatched_tracks, unmatched_dets = self._match(
            self.active_tracks, valid_dets, self.iou_thresh, use_ocm=True
        )

        for track_idx, det_idx in matches:
            self.active_tracks[track_idx].update(valid_dets[det_idx], timestamp)

        # 4. Second association: OCR (Observation-Centric Recovery)
        # Try to match remaining tracks with remaining detections using last observation
        if unmatched_tracks and unmatched_dets:
            remaining_tracks = [self.active_tracks[i] for i in unmatched_tracks]
            remaining_dets = [valid_dets[i] for i in unmatched_dets]

            ocr_matches, still_unmatched_tracks_local, still_unmatched_dets_local = (
                self._match(remaining_tracks, remaining_dets, self.iou_thresh, use_ocr=True)
            )

            for local_track_idx, local_det_idx in ocr_matches:
                track_idx = unmatched_tracks[local_track_idx]
                det_idx = unmatched_dets[local_det_idx]
                self.active_tracks[track_idx].update(valid_dets[det_idx], timestamp)

            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched_tracks_local]
            unmatched_dets = [unmatched_dets[i] for i in still_unmatched_dets_local]

        # 5. Initialize new tracks
        for det_idx in unmatched_dets:
            self._start_track(valid_dets[det_idx], timestamp)

        # 6. Cleanup lost tracks
        survivors: list[Track] = []
        for track in self.active_tracks:
            if track.time_since_update > self.max_age:
                if track.hits >= self.min_hits:
                    self.completed_tracks.append(track)
                continue
            survivors.append(track)
        self.active_tracks = survivors

    def predict_only(self) -> None:
        for track in self.active_tracks:
            track.predict()

        survivors: list[Track] = []
        for track in self.active_tracks:
            if track.time_since_update > self.max_age:
                if track.hits >= self.min_hits:
                    self.completed_tracks.append(track)
                continue
            survivors.append(track)
        self.active_tracks = survivors

    def finalize(self, round_digits: int) -> list[dict[str, object]]:
        finished = list(self.completed_tracks)
        finished.extend(
            track for track in self.active_tracks if track.hits >= self.min_hits
        )
        finished.sort(key=lambda track: track.id)
        return [track.to_output(round_digits=round_digits) for track in finished]

    def _start_track(self, detection: Detection, timestamp: int) -> None:
        # Guard: Do not create a new track if the detection overlaps significantly
        # with an active track (IoU > 0.5 -> likely the same physical object).
        for existing in self.active_tracks:
            if bbox_iou(existing.bbox_xyxy, detection.bbox_xyxy) > 0.5:
                return  # Area is already covered by an active track -> skip creating a new ID

        track = Track(
            id=self._next_track_id,
            bbox_xyxy=detection.bbox_xyxy.astype(np.float32, copy=True),
            score=float(detection.score),
            class_id=int(detection.class_id),
            history_timestamps=[int(timestamp)],
            history_boxes=[detection.bbox_xyxy.astype(float).tolist()],
        )
        self._next_track_id += 1
        self.active_tracks.append(track)

    def _match(
        self,
        tracks: Sequence[Track],
        detections: Sequence[Detection],
        iou_thresh: float,
        use_ocm: bool = False,
        use_ocr: bool = False,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not detections:
            return [], list(range(len(tracks))), list(range(len(detections)))

        # Use last observation for OCR, predicted bbox for others
        if use_ocr:
            track_boxes = np.stack([t.last_observation for t in tracks])
        else:
            track_boxes = np.stack([t.bbox_xyxy for t in tracks])

        det_boxes = np.stack([d.bbox_xyxy for d in detections])

        # IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detections)))
        for i, tb in enumerate(track_boxes):
            for j, db in enumerate(det_boxes):
                iou_matrix[i, j] = bbox_iou(tb, db)

        cost_matrix = 1.0 - iou_matrix

        # OCM (Observation-Centric Momentum)
        if use_ocm:
            for i, track in enumerate(tracks):
                if track.hits > 1:
                    for j, det in enumerate(detections):
                        det_velocity = (det.bbox_xyxy - track.last_observation) / max(1, track.time_since_update)

                        # Cosine similarity between track velocity and det velocity
                        v1 = track.velocity
                        v2 = det_velocity
                        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
                        if norm > 1e-6:
                            cos_sim = np.dot(v1, v2) / norm
                            # OC-SORT: inertia term reduces cost for similar directions
                            cost_matrix[i, j] += self.inertia * (1.0 - cos_sim)

        # Association
        cost_matrix[iou_matrix < iou_thresh] = LARGE_COST
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches: list[tuple[int, int]] = []
        for r, c in zip(row_indices, col_indices):
            if cost_matrix[r, c] < LARGE_COST:
                matches.append((r, c))

        matched_rows = {m[0] for m in matches}
        matched_cols = {m[1] for m in matches}
        unmatched_rows = [i for i in range(len(tracks)) if i not in matched_rows]
        unmatched_cols = [i for i in range(len(detections)) if i not in matched_cols]

        return matches, unmatched_rows, unmatched_cols
