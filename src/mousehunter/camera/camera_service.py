"""
Camera Service - Dual Stream PiCamera2 Interface

Implements the "Zero-Copy" vision pipeline using picamera2:
- Stream 1 (lores): 640x640 RGB for AI inference (Hailo-8L)
- Stream 2 (main): 1080p H.264 for evidence buffer

The lores stream is mapped directly to NPU memory space via
the Hailo integration, minimizing CPU involvement.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .circular_buffer import CircularVideoBuffer

logger = logging.getLogger(__name__)

# Conditional import for development without Pi hardware
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import CircularOutput

    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    logger.warning("picamera2 not available - running in simulation mode")


class MockPicamera2:
    """Mock camera for development/testing without hardware."""

    def __init__(self):
        self._running = False
        self._config = None
        logger.info("[MOCK] Picamera2 initialized")

    def create_video_configuration(self, **kwargs):
        self._config = kwargs
        logger.info(f"[MOCK] Video config created: {kwargs}")
        return kwargs

    def configure(self, config):
        self._config = config
        logger.info("[MOCK] Camera configured")

    def start(self):
        self._running = True
        logger.info("[MOCK] Camera started")

    def stop(self):
        self._running = False
        logger.info("[MOCK] Camera stopped")

    def close(self):
        self._running = False
        logger.info("[MOCK] Camera closed")

    def capture_array(self, stream: str = "main") -> np.ndarray:
        """Generate a mock frame."""
        if stream == "lores":
            return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

    def capture_buffer(self, stream: str = "main"):
        return self.capture_array(stream)

    @property
    def started(self) -> bool:
        return self._running


class CameraService:
    """
    Dual-stream camera service for inference and evidence capture.

    Configures PiCamera 3 with:
    - Main stream: 1920x1080 RGB for H.264 encoding (evidence)
    - Lores stream: 640x640 RGB for Hailo-8L inference

    The service runs a background thread that continuously feeds
    frames to the circular buffer while providing low-latency
    access to inference frames.
    """

    def __init__(
        self,
        main_resolution: tuple[int, int] = (1920, 1080),
        inference_resolution: tuple[int, int] = (640, 640),
        framerate: int = 30,
        buffer_seconds: float = 15.0,
        vflip: bool = False,
        hflip: bool = False,
        output_dir: str | Path = "runtime/recordings",
    ):
        """
        Initialize the camera service.

        Args:
            main_resolution: Resolution for evidence stream (1080p default)
            inference_resolution: Resolution for AI stream (640x640 for YOLO)
            framerate: Target framerate
            buffer_seconds: Circular buffer size in seconds
            vflip: Vertical flip
            hflip: Horizontal flip
            output_dir: Directory for saved recordings
        """
        self.main_resolution = main_resolution
        self.inference_resolution = inference_resolution
        self.framerate = framerate
        self.vflip = vflip
        self.hflip = hflip

        # State
        self._started = False
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_inference_frame: np.ndarray | None = None
        self._latest_main_frame: np.ndarray | None = None
        self._frame_timestamp: datetime | None = None
        self._frame_count = 0

        # Circular buffer for evidence
        self.buffer = CircularVideoBuffer(
            buffer_seconds=buffer_seconds,
            framerate=framerate,
            resolution=main_resolution,
            output_dir=output_dir,
        )

        # Frame callbacks (for inference engine)
        self._frame_callbacks: list[Callable[[np.ndarray, datetime], None]] = []

        # Initialize camera
        if PICAMERA_AVAILABLE:
            self._camera = Picamera2()
        else:
            self._camera = MockPicamera2()

        self._configure_camera()

        logger.info(
            f"CameraService initialized: main={main_resolution}, "
            f"inference={inference_resolution}, fps={framerate}"
        )

    def _configure_camera(self) -> None:
        """Configure camera with dual streams."""
        # Create video configuration with both streams
        # Main stream: High-res for evidence
        # Lores stream: Square format for YOLO inference

        if PICAMERA_AVAILABLE:
            config = self._camera.create_video_configuration(
                main={
                    "size": self.main_resolution,
                    "format": "RGB888",
                },
                lores={
                    "size": self.inference_resolution,
                    "format": "RGB888",
                },
                controls={
                    "FrameRate": self.framerate,
                },
                transform=self._get_transform(),
            )
            self._camera.configure(config)
        else:
            # Mock configuration
            self._camera.create_video_configuration(
                main={"size": self.main_resolution},
                lores={"size": self.inference_resolution},
            )

        logger.info("Camera configured with dual streams")

    def _get_transform(self):
        """Get libcamera transform for flip settings."""
        if PICAMERA_AVAILABLE:
            from libcamera import Transform

            return Transform(hflip=self.hflip, vflip=self.vflip)
        return None

    def start(self) -> None:
        """Start the camera and capture thread."""
        if self._started:
            logger.warning("Camera already started")
            return

        self._camera.start()
        self._started = True
        self._stop_event.clear()

        # Start background capture thread
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCaptureThread",
            daemon=True,
        )
        self._capture_thread.start()

        logger.info("Camera started with capture thread")

        # Allow camera to warm up
        time.sleep(0.5)

    def stop(self) -> None:
        """Stop the camera and capture thread."""
        if not self._started:
            return

        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        self._camera.stop()
        self._started = False

        logger.info("Camera stopped")

    def _capture_loop(self) -> None:
        """Background thread for continuous frame capture."""
        logger.info("Capture loop started")
        target_interval = 1.0 / self.framerate

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                timestamp = datetime.now()

                # Capture both streams (synchronized if using Picamera2)
                if PICAMERA_AVAILABLE:
                    request = self._camera.capture_request()
                    try:
                        main_frame = request.make_array("main")
                        inference_frame = request.make_array("lores")
                    finally:
                        # CRITICAL: Must release request to avoid buffer exhaustion
                        request.release()
                else:
                    # Mock capture
                    main_frame = self._camera.capture_array("main")
                    inference_frame = self._camera.capture_array("lores")

                with self._frame_lock:
                    self._latest_main_frame = main_frame
                    self._latest_inference_frame = inference_frame
                    self._frame_timestamp = timestamp
                    self._frame_count += 1

                # Add main frame to circular buffer
                self.buffer.add_frame(main_frame, timestamp)

                # Notify callbacks (for inference)
                for callback in self._frame_callbacks:
                    try:
                        callback(inference_frame, timestamp)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

            except Exception as e:
                logger.error(f"Capture error: {e}")

            # Maintain framerate
            elapsed = time.perf_counter() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info("Capture loop stopped")

    def get_inference_frame(self) -> tuple[np.ndarray | None, datetime | None]:
        """
        Get the latest frame for inference.

        Returns:
            Tuple of (frame, timestamp) or (None, None) if not available
        """
        with self._frame_lock:
            if self._latest_inference_frame is not None:
                return self._latest_inference_frame.copy(), self._frame_timestamp
        return None, None

    def get_main_frame(self) -> tuple[np.ndarray | None, datetime | None]:
        """
        Get the latest high-resolution frame.

        Returns:
            Tuple of (frame, timestamp) or (None, None) if not available
        """
        with self._frame_lock:
            if self._latest_main_frame is not None:
                return self._latest_main_frame.copy(), self._frame_timestamp
        return None, None

    def capture_snapshot_bytes(self, quality: int = 85) -> bytes | None:
        """
        Capture current frame as JPEG bytes.

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG bytes or None if failed
        """
        return self.buffer.get_snapshot_bytes(quality)

    def save_snapshot(self, filename: str | None = None) -> Path | None:
        """
        Save current frame as JPEG file.

        Args:
            filename: Custom filename (auto-generated if None)

        Returns:
            Path to saved file or None
        """
        return self.buffer.save_snapshot(filename)

    def trigger_evidence_save(self, event_name: str = "detection") -> Path | None:
        """
        Trigger saving of the circular buffer.

        Args:
            event_name: Name for the event folder

        Returns:
            Path to saved evidence directory
        """
        return self.buffer.trigger_save(event_name)

    def on_frame(self, callback: Callable[[np.ndarray, datetime], None]) -> None:
        """
        Register callback for new inference frames.

        The callback receives (frame, timestamp) for each captured frame.
        Used by the inference engine for real-time processing.

        Args:
            callback: Function to call with new frames
        """
        self._frame_callbacks.append(callback)
        logger.debug(f"Frame callback registered, total: {len(self._frame_callbacks)}")

    def get_status(self) -> dict:
        """Get camera service status."""
        return {
            "started": self._started,
            "frame_count": self._frame_count,
            "main_resolution": self.main_resolution,
            "inference_resolution": self.inference_resolution,
            "framerate": self.framerate,
            "buffer_status": self.buffer.get_status(),
            "picamera_available": PICAMERA_AVAILABLE,
        }

    def cleanup(self) -> None:
        """Clean up camera resources."""
        self.stop()
        self._camera.close()
        logger.info("Camera resources cleaned up")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Factory function
def _create_default_camera() -> CameraService:
    """Create camera service from config."""
    try:
        from mousehunter.config import camera_config, recording_config

        return CameraService(
            main_resolution=camera_config.main_resolution,
            inference_resolution=camera_config.inference_resolution,
            framerate=camera_config.framerate,
            buffer_seconds=camera_config.buffer_seconds,
            vflip=camera_config.vflip,
            hflip=camera_config.hflip,
            output_dir=recording_config.output_dir,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return CameraService()


# Global instance (lazy)
_camera_instance: CameraService | None = None


def get_camera_service() -> CameraService:
    """Get or create the global camera service."""
    global _camera_instance
    if _camera_instance is None:
        _camera_instance = _create_default_camera()
    return _camera_instance


def test_camera() -> None:
    """Test the camera service."""
    logging.basicConfig(level=logging.INFO)
    print("=== Camera Service Test ===")
    print(f"PiCamera Available: {PICAMERA_AVAILABLE}")

    camera = CameraService(
        main_resolution=(1920, 1080),
        inference_resolution=(640, 640),
        framerate=30,
        buffer_seconds=5.0,
    )

    def frame_handler(frame: np.ndarray, ts: datetime):
        print(f"Frame: {frame.shape} at {ts.strftime('%H:%M:%S.%f')[:-3]}")

    camera.on_frame(frame_handler)

    try:
        with camera:
            print("Camera running... Press Ctrl+C to stop")
            print(f"Status: {camera.get_status()}")

            time.sleep(3)

            # Test snapshot
            snapshot = camera.save_snapshot()
            print(f"Snapshot saved: {snapshot}")

            # Test evidence save
            evidence = camera.trigger_evidence_save("test")
            print(f"Evidence saved: {evidence}")

            time.sleep(2)

    except KeyboardInterrupt:
        print("\nStopping...")

    print("Test complete")


if __name__ == "__main__":
    test_camera()
