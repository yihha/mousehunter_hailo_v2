"""
Circular Video Buffer

Implements a ring buffer in RAM to store the last N seconds of video
without continuous SD card writes. When an event triggers, the buffer
is dumped to disk as evidence.

This is critical for capturing the "approach" footage before detection.
"""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)

# Conditional import for PIL
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - frame saving will be limited")


@dataclass
class BufferedFrame:
    """A single frame in the circular buffer."""

    timestamp: datetime
    frame_data: np.ndarray  # RGB or YUV frame
    frame_number: int


class CircularVideoBuffer:
    """
    RAM-based circular buffer for video frames.

    Stores the last N seconds of video frames in memory.
    On trigger, saves frames to disk for evidence without
    interrupting the live stream.

    Zero-copy design: frames are stored as numpy array views
    when possible to minimize memory overhead.
    """

    def __init__(
        self,
        buffer_seconds: float = 15.0,
        framerate: float = 30.0,
        resolution: tuple[int, int] = (1920, 1080),
        output_dir: str | Path = "runtime/recordings",
    ):
        """
        Initialize the circular buffer.

        Args:
            buffer_seconds: How many seconds of video to keep in memory
            framerate: Expected framerate for capacity calculation
            resolution: Frame resolution (for memory estimation)
            output_dir: Directory for saved evidence
        """
        self.buffer_seconds = buffer_seconds
        self.framerate = framerate
        self.resolution = resolution
        self.output_dir = Path(output_dir)

        # Calculate buffer capacity
        self.max_frames = int(buffer_seconds * framerate)

        # Thread-safe deque as ring buffer
        self._buffer: deque[BufferedFrame] = deque(maxlen=self.max_frames)
        self._lock = threading.Lock()
        self._frame_counter = 0

        # State
        self._is_recording = False
        self._trigger_time: datetime | None = None

        # Callbacks
        self._on_save_callbacks: list[Callable[[Path], None]] = []

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Memory estimation
        bytes_per_frame = resolution[0] * resolution[1] * 3  # RGB
        total_mb = (bytes_per_frame * self.max_frames) / (1024 * 1024)
        logger.info(
            f"CircularBuffer initialized: {self.max_frames} frames "
            f"({buffer_seconds}s @ {framerate}fps), ~{total_mb:.1f}MB RAM"
        )

    @property
    def frame_count(self) -> int:
        """Number of frames currently in buffer."""
        return len(self._buffer)

    @property
    def buffer_duration(self) -> float:
        """Actual duration of buffered content (seconds)."""
        return len(self._buffer) / self.framerate if self.framerate > 0 else 0.0

    @property
    def is_full(self) -> bool:
        """Check if buffer has reached capacity."""
        return len(self._buffer) >= self.max_frames

    def add_frame(self, frame: np.ndarray, timestamp: datetime | None = None) -> None:
        """
        Add a frame to the circular buffer.

        Old frames are automatically dropped when buffer is full.

        Args:
            frame: Image frame as numpy array (RGB or BGR)
            timestamp: Frame timestamp (auto-generated if None)
        """
        timestamp = timestamp or datetime.now()

        with self._lock:
            self._frame_counter += 1
            buffered = BufferedFrame(
                timestamp=timestamp,
                frame_data=frame.copy(),  # Copy to ensure we own the data
                frame_number=self._frame_counter,
            )
            self._buffer.append(buffered)

    def get_frames(
        self, last_n: int | None = None, since: datetime | None = None
    ) -> list[BufferedFrame]:
        """
        Get frames from the buffer.

        Args:
            last_n: Get the last N frames (None for all)
            since: Get frames since this timestamp

        Returns:
            List of BufferedFrame objects
        """
        with self._lock:
            frames = list(self._buffer)

        if since:
            frames = [f for f in frames if f.timestamp >= since]

        if last_n:
            frames = frames[-last_n:]

        return frames

    def get_latest_frame(self) -> BufferedFrame | None:
        """Get the most recent frame."""
        with self._lock:
            if self._buffer:
                return self._buffer[-1]
        return None

    def trigger_save(
        self,
        event_name: str = "detection",
        pre_seconds: float | None = None,
    ) -> Path | None:
        """
        Trigger saving of the buffer to disk.

        Saves buffered frames from (now - pre_seconds) to now.

        Args:
            event_name: Name for the saved file
            pre_seconds: Seconds before trigger to include (default: all buffered)

        Returns:
            Path to saved directory, or None if failed
        """
        self._trigger_time = datetime.now()
        pre_seconds = pre_seconds or self.buffer_seconds
        since = self._trigger_time - timedelta(seconds=pre_seconds)

        # Get pre-event frames
        pre_frames = self.get_frames(since=since)

        if not pre_frames:
            logger.warning("No frames in buffer to save")
            return None

        # Create output directory for this event
        timestamp_str = self._trigger_time.strftime("%Y%m%d_%H%M%S")
        event_dir = self.output_dir / f"{event_name}_{timestamp_str}"
        event_dir.mkdir(parents=True, exist_ok=True)

        # Save frames
        saved_count = self._save_frames_to_dir(pre_frames, event_dir, "pre")

        logger.info(f"Saved {saved_count} pre-event frames to {event_dir}")

        # Notify callbacks
        for callback in self._on_save_callbacks:
            try:
                callback(event_dir)
            except Exception as e:
                logger.error(f"Save callback error: {e}")

        return event_dir

    def _save_frames_to_dir(
        self, frames: list[BufferedFrame], output_dir: Path, prefix: str
    ) -> int:
        """Save frames to a directory as images."""
        if not PIL_AVAILABLE:
            logger.error("PIL not available, cannot save frames")
            return 0

        saved = 0
        for i, frame in enumerate(frames):
            try:
                filename = f"{prefix}_{i:04d}_{frame.timestamp.strftime('%H%M%S_%f')}.jpg"
                filepath = output_dir / filename

                # Convert numpy array to PIL Image and save
                img = Image.fromarray(frame.frame_data)
                img.save(filepath, "JPEG", quality=85)
                saved += 1

            except Exception as e:
                logger.error(f"Failed to save frame {i}: {e}")

        return saved

    def save_snapshot(self, filename: str | None = None) -> Path | None:
        """
        Save the latest frame as a snapshot.

        Args:
            filename: Custom filename (auto-generated if None)

        Returns:
            Path to saved image, or None if failed
        """
        frame = self.get_latest_frame()
        if not frame:
            logger.warning("No frame available for snapshot")
            return None

        if not PIL_AVAILABLE:
            logger.error("PIL not available, cannot save snapshot")
            return None

        filename = filename or f"snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        filepath = self.output_dir / filename

        try:
            img = Image.fromarray(frame.frame_data)
            img.save(filepath, "JPEG", quality=90)
            logger.info(f"Snapshot saved: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")
            return None

    def get_snapshot_bytes(self, quality: int = 85) -> bytes | None:
        """
        Get the latest frame as JPEG bytes.

        Args:
            quality: JPEG quality (1-100)

        Returns:
            JPEG image as bytes, or None if failed
        """
        frame = self.get_latest_frame()
        if not frame:
            return None

        if not PIL_AVAILABLE:
            logger.error("PIL not available")
            return None

        try:
            from io import BytesIO

            img = Image.fromarray(frame.frame_data)
            buffer = BytesIO()
            img.save(buffer, "JPEG", quality=quality)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to encode frame: {e}")
            return None

    def on_save(self, callback: Callable[[Path], None]) -> None:
        """Register callback for when buffer is saved."""
        self._on_save_callbacks.append(callback)

    def clear(self) -> None:
        """Clear all frames from the buffer."""
        with self._lock:
            self._buffer.clear()
            logger.info("Buffer cleared")

    def get_status(self) -> dict:
        """Get buffer status."""
        return {
            "frame_count": self.frame_count,
            "max_frames": self.max_frames,
            "buffer_duration_seconds": self.buffer_duration,
            "buffer_capacity_seconds": self.buffer_seconds,
            "is_full": self.is_full,
            "resolution": self.resolution,
            "framerate": self.framerate,
            "output_dir": str(self.output_dir),
        }


def test_buffer() -> None:
    """Test the circular buffer."""
    logging.basicConfig(level=logging.INFO)
    print("=== Circular Buffer Test ===")

    buffer = CircularVideoBuffer(
        buffer_seconds=5.0,
        framerate=10.0,
        resolution=(640, 480),
    )

    # Generate test frames
    print("Adding test frames...")
    for i in range(60):  # 6 seconds at 10fps
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        buffer.add_frame(frame)
        time.sleep(0.1)

    print(f"Status: {buffer.get_status()}")

    # Test snapshot
    snapshot = buffer.save_snapshot()
    print(f"Snapshot saved: {snapshot}")

    # Test trigger save
    event_dir = buffer.trigger_save("test_event")
    print(f"Event saved to: {event_dir}")


if __name__ == "__main__":
    test_buffer()
