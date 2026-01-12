"""
Inference module for MouseHunter.

Provides:
- HailoEngine: Hailo-8L NPU inference wrapper
- Detection: Detection result data structures
- PreyDetector: High-level prey detection logic with debouncing
"""

from .detection import Detection, BoundingBox
from .hailo_engine import HailoEngine, get_hailo_engine
from .prey_detector import PreyDetector, get_prey_detector

__all__ = [
    "Detection",
    "BoundingBox",
    "HailoEngine",
    "get_hailo_engine",
    "PreyDetector",
    "get_prey_detector",
]
