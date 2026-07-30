"""
Core functionality for DINOv3 pipelines.
"""

import pkgutil
__path__ = pkgutil.extend_path(__path__, __name__)

from .pipeline import process_video
from .detector import Detection

__all__ = [
    "process_video",
    "Detection",
]
