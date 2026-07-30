from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class Detection:
    bbox_xyxy: np.ndarray
    score: float
    class_id: int
    embedding: np.ndarray | None = None
