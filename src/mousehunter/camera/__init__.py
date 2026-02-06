"""
Camera module for MouseHunter.

Provides:
- CameraService: Dual-stream picamera2 setup for inference and evidence
- CircularBuffer: RAM-based video buffer for pre-event recording
- VideoEncoder / EvidenceRecorder: H.264 video evidence encoding
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
