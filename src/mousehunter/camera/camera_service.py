"""
Camera Service - Dual Stream PiCamera2 Interface

Implements the "Zero-Copy" vision pipeline using picamera2:
- Stream 1 (lores): 640x640 RGB for AI inference (Hailo-8L)
- Stream 2 (main): 1080p YUV420 -> hardware H.264 -> CircularOutput2

The main stream is hardware-encoded to H.264 and held in a ~19 MB
circular buffer (CircularOutput2), replacing the previous 2.7 GB
raw-frame deque. Evidence is saved as MP4 via PyavOutput.
"""

import logging
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Conditional import for development without Pi hardware
try:
    from picamera2 import Picamera2
    from picamera2.encoders import H264Encoder
    from picamera2.outputs import CircularOutput2

    # PyavOutput for MP4 muxing (available in picamera2 >= 0.3.17)
    try:
        from picamera2.outputs import PyavOutput
        PYAV_AVAILABLE = True
    except ImportError:
        PYAV_AVAILABLE = False
        logger.warning("PyavOutput not available - upgrade picamera2 for MP4 evidence")

    PICAMERA_AVAILABLE = True
except ImportError:
    PICAMERA_AVAILABLE = False
    PYAV_AVAILABLE = False
    logger.warning("picamera2 not available - running in simulation mode")

# PIL for on-demand snapshots
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


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

    def start_recording(self, encoder, output):
        logger.info("[MOCK] Recording started")

    def stop_recording(self):
        logger.info("[MOCK] Recording stopped")

    def capture_array(self, stream: str = "main") -> np.ndarray:
        """Generate a mock frame."""
        if stream == "lores":
            return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        # Main stream is YUV420: shape (height * 3 // 2, width) - 2D array
        # Simulating this keeps mock behavior consistent with real camera
        return np.random.randint(0, 255, (1080 * 3 // 2, 1920), dtype=np.uint8)

    def capture_buffer(self, stream: str = "main"):
        return self.capture_array(stream)

    @property
    def started(self) -> bool:
        return self._running


class CameraService:
    """
    Dual-stream camera service for inference and evidence capture.

    Configures PiCamera 3 with:
    - Main stream: 1920x1080 YUV420 -> H.264 hardware encoder -> CircularOutput2 (~19 MB)
    - Lores stream: 640x640 RGB for Hailo-8L inference

    The capture thread only reads the lores stream for inference.
    The main stream is handled entirely by the hardware encoder pipeline.
    On-demand snapshots use the lores stream (RGB888) since the main stream
    is YUV420 and cannot be directly used with PIL Image.fromarray().
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
        post_roll_seconds: float = 15.0,
        evidence_format: str = "video",
    ):
        self.main_resolution = main_resolution
        self.inference_resolution = inference_resolution
        self.framerate = framerate
        self.buffer_seconds = buffer_seconds
        self.vflip = vflip
        self.hflip = hflip
        self.evidence_format = evidence_format
        self.output_dir = Path(output_dir)
        self.post_roll_seconds = post_roll_seconds

        # State
        self._started = False
        self._capture_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_inference_frame: np.ndarray | None = None
        self._frame_timestamp: datetime | None = None
        self._frame_count = 0

        # Evidence state
        self._evidence_recording = False
        self._evidence_thread: threading.Thread | None = None
        self._on_complete_callbacks: list[Callable[[Path, bool], None]] = []

        # Hardware encoder + circular buffer (Pi only)
        self._h264_encoder = None
        self._circular_output = None

        # Frame callbacks (for inference engine)
        self._frame_callbacks: list[Callable[[np.ndarray, datetime], None]] = []

        # Initialize camera
        if PICAMERA_AVAILABLE:
            self._camera = Picamera2()
        else:
            self._camera = MockPicamera2()

        self._configure_camera()

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"CameraService initialized: main={main_resolution}, "
            f"inference={inference_resolution}, fps={framerate}, "
            f"buffer={buffer_seconds}s, evidence={evidence_format}"
        )

    def _configure_camera(self) -> None:
        """Configure camera with dual streams."""
        if PICAMERA_AVAILABLE:
            config = self._camera.create_video_configuration(
                main={
                    "size": self.main_resolution,
                    # YUV420 for hardware H.264 encoding (not RGB888)
                    "format": "YUV420",
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
            logger.info("Camera configured: main=YUV420 (hw encode), lores=RGB888 (inference)")
        else:
            # Mock configuration
            self._camera.create_video_configuration(
                main={"size": self.main_resolution},
                lores={"size": self.inference_resolution},
            )
            logger.info("Camera configured (mock mode)")

    def _get_transform(self):
        """Get libcamera transform for flip settings."""
        if PICAMERA_AVAILABLE:
            from libcamera import Transform
            return Transform(hflip=self.hflip, vflip=self.vflip)
        return None

    def start(self) -> None:
        """Start the camera, encoder, and capture thread."""
        if self._started:
            logger.warning("Camera already started")
            return

        if PICAMERA_AVAILABLE and PYAV_AVAILABLE:
            # Start hardware H.264 encoder with circular buffer
            buffer_ms = int(self.buffer_seconds * 1000)
            self._h264_encoder = H264Encoder(bitrate=5_000_000, repeat=True)
            self._circular_output = CircularOutput2(buffer_duration_ms=buffer_ms)

            # start_recording starts the camera + encoder together
            self._camera.start_recording(self._h264_encoder, self._circular_output)
            logger.info(
                f"Hardware H.264 encoder started: 5Mbps, "
                f"circular buffer={self.buffer_seconds}s "
                f"(~{5_000_000 * self.buffer_seconds / 8 / 1024 / 1024:.0f}MB)"
            )
        else:
            self._camera.start()
            if PICAMERA_AVAILABLE:
                logger.warning("PyavOutput unavailable, running without H.264 encoder")

        self._started = True
        self._stop_event.clear()

        # Start background capture thread (lores only for inference)
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
        """Stop the camera, encoder, and capture thread."""
        if not self._started:
            return

        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
            self._capture_thread = None

        if PICAMERA_AVAILABLE and self._h264_encoder:
            try:
                self._camera.stop_recording()
            except Exception as e:
                logger.error(f"Error stopping recording: {e}")
            self._h264_encoder = None
            self._circular_output = None
        else:
            self._camera.stop()

        self._started = False
        logger.info("Camera stopped")

    def _capture_loop(self) -> None:
        """
        Background thread for continuous inference frame capture.

        Only captures the lores stream for inference. The main stream
        is handled by the hardware H.264 encoder pipeline.
        """
        logger.info("Capture loop started (lores only)")
        target_interval = 1.0 / self.framerate

        while not self._stop_event.is_set():
            loop_start = time.perf_counter()

            try:
                timestamp = datetime.now()

                # Capture lores stream only (main is handled by H.264 encoder)
                if PICAMERA_AVAILABLE:
                    inference_frame = self._camera.capture_array("lores")
                else:
                    inference_frame = self._camera.capture_array("lores")

                with self._frame_lock:
                    self._latest_inference_frame = inference_frame
                    self._frame_timestamp = timestamp
                    self._frame_count += 1

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
        """Get the latest frame for inference (lores RGB)."""
        with self._frame_lock:
            if self._latest_inference_frame is not None:
                return self._latest_inference_frame.copy(), self._frame_timestamp
        return None, None

    def get_main_frame(self) -> tuple[np.ndarray | None, datetime | None]:
        """
        Get an RGB frame on-demand via capture_array.

        Returns the lores stream (640x640 RGB888) because the main stream
        is YUV420 (required for hardware H.264 encoding) and cannot be
        directly used as RGB with Image.fromarray().
        """
        if not self._started:
            return None, None
        try:
            frame = self._camera.capture_array("lores")
            return frame, datetime.now()
        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None, None

    def capture_snapshot_bytes(self, quality: int = 85) -> bytes | None:
        """Capture current frame as JPEG bytes (from lores RGB stream)."""
        if not self._started:
            return None
        if not PIL_AVAILABLE:
            logger.error("PIL not available for snapshot encoding")
            return None
        try:
            # Use lores stream (RGB888) - main stream is YUV420 for hw encoder
            frame = self._camera.capture_array("lores")
            img = Image.fromarray(frame)
            buf = BytesIO()
            img.save(buf, "JPEG", quality=quality)
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Failed to capture snapshot: {e}")
            return None

    def save_snapshot(self, filename: str | None = None) -> Path | None:
        """Save current frame as JPEG file (from lores RGB stream)."""
        if not self._started or not PIL_AVAILABLE:
            return None
        try:
            # Use lores stream (RGB888) - main stream is YUV420 for hw encoder
            frame = self._camera.capture_array("lores")
            filename = filename or f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = self.output_dir / filename
            img = Image.fromarray(frame)
            img.save(filepath, "JPEG", quality=90)
            logger.info(f"Snapshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def trigger_evidence_save(self, event_name: str = "detection") -> Path | None:
        """
        Trigger saving evidence from the circular H.264 buffer.

        Opens a PyavOutput to write the buffered pre-roll + live post-roll
        to an MP4 file. A background thread handles the post-roll timing
        and calls close_output() when done.

        Args:
            event_name: Name for the event folder

        Returns:
            Path to evidence directory (created immediately)
        """
        evidence_dir = self.output_dir / event_name
        evidence_dir.mkdir(parents=True, exist_ok=True)

        # Guard: skip if already recording evidence
        if self._evidence_recording:
            logger.warning("Evidence recording already in progress, SKIPPING")
            return evidence_dir

        if not PICAMERA_AVAILABLE or not PYAV_AVAILABLE or not self._circular_output:
            logger.warning("CircularOutput2 not available, cannot save video evidence")
            return evidence_dir

        output_path = evidence_dir / "evidence.mp4"

        try:
            # Open output: flushes circular buffer (pre-roll) + continues recording
            self._circular_output.open_output(PyavOutput(str(output_path)))
            self._evidence_recording = True
            logger.info(
                f"Evidence recording started: {output_path} "
                f"(pre-roll={self.buffer_seconds}s, post-roll={self.post_roll_seconds}s)"
            )

            # Background thread to wait for post-roll then close
            self._evidence_thread = threading.Thread(
                target=self._evidence_post_roll,
                args=(evidence_dir, output_path),
                name="EvidencePostRollThread",
                daemon=True,
            )
            self._evidence_thread.start()

        except Exception as e:
            logger.error(f"Failed to start evidence recording: {e}", exc_info=True)
            self._evidence_recording = False

        return evidence_dir

    def _evidence_post_roll(self, evidence_dir: Path, output_path: Path) -> None:
        """Background thread: wait for post-roll duration, then close the output."""
        success = False
        try:
            time.sleep(self.post_roll_seconds)

            if self._circular_output:
                self._circular_output.close_output()
                logger.info(f"Evidence recording complete: {output_path}")

            # Verify the file was created
            if output_path.exists():
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                logger.info(f"Evidence file: {output_path} ({file_size_mb:.1f}MB)")
                success = True
            else:
                logger.error(f"Evidence file not created: {output_path}")

        except Exception as e:
            logger.error(f"Evidence post-roll error: {e}", exc_info=True)
        finally:
            self._evidence_recording = False
            # Fire completion callbacks
            for callback in self._on_complete_callbacks:
                try:
                    callback(evidence_dir, success)
                except Exception as e:
                    logger.error(f"Evidence complete callback error: {e}")

    def on_evidence_complete(self, callback: Callable[[Path, bool], None]) -> None:
        """Register callback for when video evidence encoding completes."""
        self._on_complete_callbacks.append(callback)

    def on_frame(self, callback: Callable[[np.ndarray, datetime], None]) -> None:
        """Register callback for new inference frames."""
        self._frame_callbacks.append(callback)
        logger.debug(f"Frame callback registered, total: {len(self._frame_callbacks)}")

    def get_status(self) -> dict:
        """Get camera service status."""
        buffer_info = {}
        if self._circular_output:
            buffer_info = {
                "type": "CircularOutput2",
                "buffer_seconds": self.buffer_seconds,
                "estimated_mb": 5_000_000 * self.buffer_seconds / 8 / 1024 / 1024,
            }
        return {
            "started": self._started,
            "frame_count": self._frame_count,
            "main_resolution": self.main_resolution,
            "inference_resolution": self.inference_resolution,
            "framerate": self.framerate,
            "buffer_status": buffer_info,
            "evidence_recording": self._evidence_recording,
            "picamera_available": PICAMERA_AVAILABLE,
            "hw_encoder": self._h264_encoder is not None,
        }

    def cleanup(self) -> None:
        """Clean up camera resources."""
        self.stop()
        if PICAMERA_AVAILABLE:
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
            post_roll_seconds=recording_config.post_roll_seconds,
            evidence_format=recording_config.evidence_format,
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
    print(f"PyavOutput Available: {PYAV_AVAILABLE}")

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
            print(f"Evidence triggered: {evidence}")

            # Wait for post-roll
            time.sleep(20)

    except KeyboardInterrupt:
        print("\nStopping...")

    print("Test complete")


if __name__ == "__main__":
    test_camera()
