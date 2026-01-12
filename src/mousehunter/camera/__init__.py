"""
Camera module for MouseHunter.

Provides:
- CameraService: Dual-stream picamera2 setup for inference and evidence
- CircularBuffer: RAM-based video buffer for pre-event recording
"""

from .camera_service import CameraService, get_camera_service
from .circular_buffer import CircularVideoBuffer

__all__ = ["CameraService", "get_camera_service", "CircularVideoBuffer"]
