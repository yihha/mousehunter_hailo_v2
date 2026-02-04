"""
Training Data Capture Module - Collect images for YOLO model improvement.

Captures images in three modes:
1. Periodic: Every N minutes regardless of detections
2. Cat-only: When cat detected without prey (reduces false positives)
3. Near-miss: When VERIFYING resets to IDLE (hard negatives)

Images are saved with detection metadata for semi-automated labeling.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from mousehunter.inference.detection import Detection

logger = logging.getLogger(__name__)


@dataclass
class CaptureMetadata:
    """Metadata for a training data capture."""

    timestamp: datetime
    capture_type: str  # "periodic", "cat_only", "near_miss"
    detections: list[dict] = field(default_factory=list)
    accumulated_score: float = 0.0
    detection_state: str = "IDLE"

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "capture_type": self.capture_type,
            "detections": self.detections,
            "accumulated_score": self.accumulated_score,
            "detection_state": self.detection_state,
        }


class TrainingDataCapture:
    """
    Captures and manages training data for model improvement.

    Coordinates with cloud storage for uploads and tracks daily limits.
    """

    def __init__(
        self,
        local_dir: str | Path = "runtime/training_data",
        remote_path: str = "MouseHunter/training",
        periodic_interval_minutes: int = 30,
        capture_cat_only: bool = True,
        cat_only_delay_seconds: float = 2.0,
        capture_near_miss: bool = True,
        include_detections_json: bool = True,
        max_images_per_day: int = 100,
        use_inference_resolution: bool = True,
        enabled: bool = True,
    ):
        """
        Initialize training data capture.

        Args:
            local_dir: Directory for local storage before upload
            remote_path: Path on cloud storage for training data
            periodic_interval_minutes: Interval for periodic captures (0 to disable)
            capture_cat_only: Capture when cat detected without prey
            cat_only_delay_seconds: Wait time before capturing cat-only
            capture_near_miss: Capture when VERIFYING resets (almost triggered)
            include_detections_json: Save detection metadata with images
            max_images_per_day: Daily capture limit (0 for unlimited)
            use_inference_resolution: Save at 640x640 instead of full res
            enabled: Enable training data capture
        """
        self.local_dir = Path(local_dir)
        self.remote_path = remote_path
        self.periodic_interval_minutes = periodic_interval_minutes
        self.cat_only_enabled = capture_cat_only
        self.cat_only_delay_seconds = cat_only_delay_seconds
        self.near_miss_enabled = capture_near_miss
        self.include_detections_json = include_detections_json
        self.max_images_per_day = max_images_per_day
        self.use_inference_resolution = use_inference_resolution
        self.enabled = enabled

        # Daily counter
        self._capture_date: date | None = None
        self._daily_count: int = 0

        # Cat-only tracking
        self._cat_detected_time: float | None = None
        self._last_cat_only_capture: float = 0
        self._cat_only_cooldown = 60.0  # Minimum seconds between cat-only captures

        # Near-miss tracking
        self._last_near_miss_capture: float = 0
        self._near_miss_cooldown = 30.0  # Minimum seconds between near-miss captures

        # Periodic tracking
        self._last_periodic_capture: float = 0

        # Cloud storage reference (set externally)
        self._cloud_storage = None

        # Ensure local directories exist
        self._ensure_dirs()

        logger.info(
            f"TrainingDataCapture initialized: enabled={enabled}, "
            f"periodic={periodic_interval_minutes}min, cat_only={self.cat_only_enabled}, "
            f"near_miss={self.near_miss_enabled}, max/day={max_images_per_day}"
        )

    def _ensure_dirs(self) -> None:
        """Create local directories for each capture type."""
        if not self.enabled:
            return

        for subdir in ["periodic", "cat_only", "near_miss"]:
            (self.local_dir / subdir).mkdir(parents=True, exist_ok=True)

    def set_cloud_storage(self, cloud_storage) -> None:
        """Set cloud storage instance for uploads."""
        self._cloud_storage = cloud_storage

    def _check_daily_limit(self) -> bool:
        """Check if daily capture limit reached. Returns True if can capture."""
        if self.max_images_per_day <= 0:
            return True  # Unlimited

        today = date.today()
        if self._capture_date != today:
            # New day, reset counter
            self._capture_date = today
            self._daily_count = 0

        return self._daily_count < self.max_images_per_day

    def _increment_daily_count(self) -> None:
        """Increment daily capture counter."""
        today = date.today()
        if self._capture_date != today:
            self._capture_date = today
            self._daily_count = 0
        self._daily_count += 1

    def _save_image(
        self,
        frame: np.ndarray,
        capture_type: str,
        metadata: CaptureMetadata,
    ) -> Path | None:
        """
        Save image and metadata to local storage.

        Args:
            frame: Image frame (RGB numpy array)
            capture_type: Type of capture ("periodic", "cat_only", "near_miss")
            metadata: Capture metadata

        Returns:
            Path to saved image or None if failed
        """
        if not self.enabled:
            return None

        if not self._check_daily_limit():
            logger.debug(f"Daily limit reached ({self.max_images_per_day}), skipping capture")
            return None

        try:
            # Generate filename with timestamp
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{capture_type}_{timestamp_str}"

            # Determine output directory
            output_dir = self.local_dir / capture_type
            output_dir.mkdir(parents=True, exist_ok=True)

            # Convert and save image
            img = Image.fromarray(frame)

            # Resize to inference resolution if configured and not already correct size
            if self.use_inference_resolution and img.size != (640, 640):
                img = img.resize((640, 640), Image.Resampling.LANCZOS)

            image_path = output_dir / f"{filename}.jpg"
            img.save(image_path, "JPEG", quality=90)

            # Save metadata if configured
            if self.include_detections_json:
                metadata_path = output_dir / f"{filename}.json"
                with open(metadata_path, "w") as f:
                    json.dump(metadata.to_dict(), f, indent=2)

            self._increment_daily_count()

            logger.info(
                f"Training data captured: {capture_type}/{filename}.jpg "
                f"({self._daily_count}/{self.max_images_per_day or 'unlimited'} today)"
            )

            return image_path

        except Exception as e:
            logger.error(f"Failed to save training image: {e}")
            return None

    async def _upload_image(self, image_path: Path, capture_type: str) -> bool:
        """Upload image to cloud storage."""
        if not self._cloud_storage or not self._cloud_storage.enabled:
            return False

        try:
            # Construct remote path with date structure
            now = datetime.now()
            remote_full_path = (
                f"{self._cloud_storage.rclone_remote}:{self.remote_path}/"
                f"{capture_type}/{now.year}/{now.month:02d}/{now.day:02d}/"
            )

            # Upload using rclone
            success, output = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: self._cloud_storage._run_rclone([
                    "copy",
                    str(image_path),
                    remote_full_path,
                ])
            )

            if success:
                logger.debug(f"Uploaded training image: {image_path.name}")

                # Also upload metadata if exists
                metadata_path = image_path.with_suffix(".json")
                if metadata_path.exists():
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: self._cloud_storage._run_rclone([
                            "copy",
                            str(metadata_path),
                            remote_full_path,
                        ])
                    )
            else:
                logger.warning(f"Failed to upload training image: {output}")

            return success

        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False

    # ==================== Capture Methods ====================

    async def capture_periodic(
        self,
        frame: np.ndarray,
        detections: list["Detection"] | None = None,
    ) -> Path | None:
        """
        Capture periodic training image.

        Args:
            frame: Current camera frame
            detections: Current detections (if any)

        Returns:
            Path to saved image or None
        """
        if not self.enabled or self.periodic_interval_minutes <= 0:
            return None

        now = time.time()
        interval_seconds = self.periodic_interval_minutes * 60

        if now - self._last_periodic_capture < interval_seconds:
            return None

        self._last_periodic_capture = now

        # Build metadata
        detection_dicts = []
        if detections:
            detection_dicts = [d.to_dict() for d in detections]

        metadata = CaptureMetadata(
            timestamp=datetime.now(),
            capture_type="periodic",
            detections=detection_dicts,
        )

        # Save locally
        image_path = self._save_image(frame, "periodic", metadata)

        # Upload asynchronously
        if image_path:
            asyncio.create_task(self._upload_image(image_path, "periodic"))

        return image_path

    async def capture_cat_only(
        self,
        frame: np.ndarray,
        cat_detection: "Detection",
        all_detections: list["Detection"] | None = None,
    ) -> Path | None:
        """
        Capture cat-only image (cat detected, no prey).

        Args:
            frame: Current camera frame
            cat_detection: The cat detection
            all_detections: All detections in frame

        Returns:
            Path to saved image or None
        """
        if not self.enabled or not self.cat_only_enabled:
            return None

        now = time.time()

        # Check cooldown
        if now - self._last_cat_only_capture < self._cat_only_cooldown:
            return None

        self._last_cat_only_capture = now

        # Build metadata
        detection_dicts = []
        if all_detections:
            detection_dicts = [d.to_dict() for d in all_detections]

        metadata = CaptureMetadata(
            timestamp=datetime.now(),
            capture_type="cat_only",
            detections=detection_dicts,
            detection_state="MONITORING",
        )

        # Save locally
        image_path = self._save_image(frame, "cat_only", metadata)

        # Upload asynchronously
        if image_path:
            asyncio.create_task(self._upload_image(image_path, "cat_only"))

            # Log with confidence if available
            if cat_detection and hasattr(cat_detection, 'confidence'):
                logger.info(
                    f"Cat-only capture: cat confidence={cat_detection.confidence:.2f}"
                )
            else:
                logger.info("Cat-only capture saved")

        return image_path

    async def capture_near_miss(
        self,
        frame: np.ndarray,
        accumulated_score: float,
        all_detections: list["Detection"] | None = None,
    ) -> Path | None:
        """
        Capture near-miss image (VERIFYING that reset to IDLE).

        Args:
            frame: Current camera frame
            accumulated_score: Score that was accumulated before reset
            all_detections: All detections in frame

        Returns:
            Path to saved image or None
        """
        if not self.enabled or not self.near_miss_enabled:
            return None

        now = time.time()

        # Check cooldown
        if now - self._last_near_miss_capture < self._near_miss_cooldown:
            return None

        self._last_near_miss_capture = now

        # Build metadata
        detection_dicts = []
        if all_detections:
            detection_dicts = [d.to_dict() for d in all_detections]

        metadata = CaptureMetadata(
            timestamp=datetime.now(),
            capture_type="near_miss",
            detections=detection_dicts,
            accumulated_score=accumulated_score,
            detection_state="VERIFYING->IDLE",
        )

        # Save locally
        image_path = self._save_image(frame, "near_miss", metadata)

        # Upload asynchronously
        if image_path:
            asyncio.create_task(self._upload_image(image_path, "near_miss"))

        logger.info(
            f"Near-miss capture: accumulated_score={accumulated_score:.2f}"
        )

        return image_path

    # ==================== State Tracking ====================

    def on_cat_detected(self) -> None:
        """Called when cat is first detected (IDLE -> MONITORING)."""
        if self._cat_detected_time is None:
            self._cat_detected_time = time.time()

    def on_cat_lost(self) -> None:
        """Called when cat is lost."""
        self._cat_detected_time = None

    def should_capture_cat_only(self) -> bool:
        """Check if we should capture a cat-only image."""
        if not self.enabled or not self.cat_only_enabled:
            return False

        if self._cat_detected_time is None:
            return False

        # Check if cat has been present long enough
        elapsed = time.time() - self._cat_detected_time
        if elapsed < self.cat_only_delay_seconds:
            return False

        # Check cooldown
        if time.time() - self._last_cat_only_capture < self._cat_only_cooldown:
            return False

        return self._check_daily_limit()

    # ==================== Cleanup ====================

    def cleanup_local(self, max_age_days: int = 7) -> int:
        """
        Delete local training data older than max_age_days.

        Returns:
            Number of files deleted
        """
        if not self.local_dir.exists():
            return 0

        deleted = 0
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

        for capture_type in ["periodic", "cat_only", "near_miss"]:
            type_dir = self.local_dir / capture_type
            if not type_dir.exists():
                continue

            for file in type_dir.iterdir():
                if file.stat().st_mtime < cutoff:
                    try:
                        file.unlink()
                        deleted += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {file}: {e}")

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old training files")

        return deleted

    def get_status(self) -> dict[str, Any]:
        """Get training data capture status."""
        # Count local files
        counts = {"periodic": 0, "cat_only": 0, "near_miss": 0}
        if self.local_dir.exists():
            for capture_type in counts:
                type_dir = self.local_dir / capture_type
                if type_dir.exists():
                    counts[capture_type] = len(list(type_dir.glob("*.jpg")))

        return {
            "enabled": self.enabled,
            "daily_count": self._daily_count,
            "max_per_day": self.max_images_per_day,
            "periodic_interval_minutes": self.periodic_interval_minutes,
            "capture_cat_only": self.cat_only_enabled,
            "capture_near_miss": self.near_miss_enabled,
            "local_counts": counts,
            "local_dir": str(self.local_dir),
            "remote_path": self.remote_path,
        }


# ==================== Factory Functions ====================

_training_capture_instance: TrainingDataCapture | None = None


def _create_default_training_capture() -> TrainingDataCapture:
    """Create training data capture from config."""
    try:
        from mousehunter.config import training_data_config

        return TrainingDataCapture(
            local_dir=training_data_config.local_dir,
            remote_path=training_data_config.remote_path,
            periodic_interval_minutes=training_data_config.periodic_interval_minutes,
            capture_cat_only=training_data_config.capture_cat_only,
            cat_only_delay_seconds=training_data_config.cat_only_delay_seconds,
            capture_near_miss=training_data_config.capture_near_miss,
            include_detections_json=training_data_config.include_detections_json,
            max_images_per_day=training_data_config.max_images_per_day,
            use_inference_resolution=training_data_config.use_inference_resolution,
            enabled=training_data_config.enabled,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return TrainingDataCapture(enabled=False)


def get_training_capture() -> TrainingDataCapture:
    """Get or create the global training data capture instance."""
    global _training_capture_instance
    if _training_capture_instance is None:
        _training_capture_instance = _create_default_training_capture()
    return _training_capture_instance


# ==================== CLI Test ====================

async def test_training_capture() -> None:
    """Test training data capture (CLI entry point)."""
    logging.basicConfig(level=logging.INFO)
    print("=== Training Data Capture Test ===")

    capture = _create_default_training_capture()

    print(f"\nConfiguration:")
    status = capture.get_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    if not capture.enabled:
        print("\nTraining data capture is disabled.")
        print("Enable it in config/config.json: training_data.enabled = true")
        return

    # Test with dummy frame
    print("\nTesting periodic capture with dummy frame...")
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Force capture by resetting last capture time
    capture._last_periodic_capture = 0

    path = await capture.capture_periodic(dummy_frame)
    if path:
        print(f"  Saved: {path}")
    else:
        print("  No capture (check daily limit)")

    print(f"\nStatus after test:")
    status = capture.get_status()
    print(f"  Daily count: {status['daily_count']}/{status['max_per_day']}")
    print(f"  Local counts: {status['local_counts']}")

    print("\nTest complete")


def test_training_capture_sync() -> None:
    """Synchronous wrapper for test."""
    asyncio.run(test_training_capture())


if __name__ == "__main__":
    test_training_capture_sync()
