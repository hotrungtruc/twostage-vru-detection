"""
DINOv3 Project Package

A comprehensive package for DINOv3 model inference, classification, auto-labeling,
and visualization utilities.

All dependencies (Detection dataclass, OCSortTracker) are now bundled inside this
repo under src/core/detector.py and src/trackers/ocsort/ocsort.py so the project
runs standalone after a plain `git clone`.
"""

import sys
from pathlib import Path
import pkgutil

# Extend path for namespace package behavior
__path__ = pkgutil.extend_path(__path__, __name__)

__version__ = "1.0.0"
