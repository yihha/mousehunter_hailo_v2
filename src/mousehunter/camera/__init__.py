"""
Camera module for MouseHunter.

Provides:
- CameraService: Dual-stream picamera2 + hardware H.264 encoder with CircularOutput2
- CircularBuffer: Legacy RAM-based raw frame buffer (kept for fallback/testing)
- VideoEncoder / EvidenceRecorder: Legacy ffmpeg-based encoding (kept for fallback)
- Frame annotation utilities for detection visualization
"""

from .camera_service import CameraService, get_camera_service
from .circular_buffer import CircularVideoBuffer
from .frame_annotator import annotate_frame, annotate_frame_to_jpeg
from .video_encoder import FFMPEG_AVAILABLE, EvidenceRecorder, VideoEncoder

__all__ = [
    "CameraService",
    "get_camera_service",
    "CircularVideoBuffer",
    "VideoEncoder",
    "EvidenceRecorder",
    "FFMPEG_AVAILABLE",
    "annotate_frame",
    "annotate_frame_to_jpeg",
]
