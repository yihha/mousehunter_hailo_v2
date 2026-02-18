"""
Configuration management for MouseHunter using Pydantic settings.

Loads configuration from:
1. .env file (if present)
2. config/config.json (defaults)
3. Environment variables (override with MOUSEHUNTER_ prefix)
"""

import json
import logging
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RUNTIME_DIR = PROJECT_ROOT / "runtime"

# Load .env file from project root (if exists)
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    load_dotenv(_env_file)
    logger.debug(f"Loaded environment from {_env_file}")


def load_json_config() -> dict[str, Any]:
    """Load configuration from config.json file."""
    config_file = CONFIG_DIR / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            return json.load(f)
    return {}


_json_config = load_json_config()


class TelegramConfig(BaseSettings):
    """Telegram bot configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_TELEGRAM_"}

    bot_token: str = Field(
        default=_json_config.get("telegram", {}).get("bot_token", ""),
        description="Telegram bot token from @BotFather",
    )
    chat_id: str = Field(
        default=_json_config.get("telegram", {}).get("chat_id", ""),
        description="Telegram chat/group ID for notifications",
    )
    enabled: bool = Field(
        default=_json_config.get("telegram", {}).get("enabled", True),
        description="Enable Telegram notifications",
    )
    notify_on_cat_detected: bool = Field(
        default=_json_config.get("telegram", {}).get("notify_on_cat_detected", True),
        description="Send notification when cat is detected",
    )
    cat_notification_cooldown_minutes: int = Field(
        default=_json_config.get("telegram", {}).get("cat_notification_cooldown_minutes", 5),
        description="Minimum minutes between cat detection notifications",
    )


class CameraConfig(BaseSettings):
    """PiCamera 3 configuration for dual-stream setup."""

    model_config = {"env_prefix": "MOUSEHUNTER_CAMERA_"}

    main_resolution: tuple[int, int] = Field(
        default=tuple(_json_config.get("camera", {}).get("main_resolution", [1920, 1080])),
        description="Main stream resolution for H.264 encoding (evidence)",
    )
    inference_resolution: tuple[int, int] = Field(
        default=tuple(_json_config.get("camera", {}).get("inference_resolution", [640, 640])),
        description="Low-res stream for AI inference (square for YOLO)",
    )
    framerate: int = Field(
        default=_json_config.get("camera", {}).get("framerate", 30),
        description="Camera framerate",
    )
    buffer_seconds: int = Field(
        default=_json_config.get("camera", {}).get("buffer_seconds", 15),
        description="Circular buffer size in seconds",
    )
    vflip: bool = Field(
        default=_json_config.get("camera", {}).get("vflip", False),
        description="Vertical flip",
    )
    hflip: bool = Field(
        default=_json_config.get("camera", {}).get("hflip", False),
        description="Horizontal flip",
    )

    @field_validator("main_resolution", "inference_resolution", mode="before")
    @classmethod
    def parse_resolution(cls, v):
        if isinstance(v, list):
            return tuple(v)
        return v

    @field_validator("framerate")
    @classmethod
    def validate_framerate(cls, v):
        if v < 1 or v > 120:
            raise ValueError(f"framerate must be between 1 and 120, got {v}")
        return v

    @field_validator("buffer_seconds")
    @classmethod
    def validate_buffer_seconds(cls, v):
        if v < 1 or v > 300:
            raise ValueError(f"buffer_seconds must be between 1 and 300, got {v}")
        return v


class InferenceConfig(BaseSettings):
    """Hailo-8L inference engine configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_INFERENCE_"}

    model_path: str = Field(
        default=_json_config.get("inference", {}).get(
            "model_path", str(PROJECT_ROOT / "models" / "yolov8n_catprey.hef")
        ),
        description="Path to compiled Hailo HEF model",
    )
    classes: dict[str, str] = Field(
        default=_json_config.get("inference", {}).get(
            "classes", {"0": "cat", "1": "rodent"}
        ),
        description="Class ID to name mapping. v3 Custom: 0=cat, 1=rodent. COCO: 15=cat, 14=bird",
    )

    # DFL reg_max (must match training configuration)
    reg_max: int = Field(
        default=_json_config.get("inference", {}).get("reg_max", 8),
        description="DFL reg_max (number of bins per coordinate). v3 uses 8 for better INT8 quantization.",
    )

    # Engine-level confidence threshold (coarse filter for performance)
    engine_confidence_threshold: float = Field(
        default=_json_config.get("inference", {}).get("engine_confidence_threshold", 0.10),
        description="Engine-level threshold for YOLO postprocessing. Low value (0.10) to let "
        "weak detections through to the score accumulation system.",
    )

    # Per-class confidence thresholds
    thresholds: dict[str, float] = Field(
        default=_json_config.get("inference", {}).get(
            "thresholds", {"cat": 0.50, "rodent": 0.15}
        ),
        description="Per-class confidence thresholds",
    )

    # Spatial validation settings
    spatial_validation_enabled: bool = Field(
        default=_json_config.get("inference", {}).get("spatial_validation", {}).get(
            "enabled", True
        ),
        description="Require prey to be spatially near cat",
    )
    box_expansion: float = Field(
        default=_json_config.get("inference", {}).get("spatial_validation", {}).get(
            "box_expansion", 0.25
        ),
        description="Expand cat box by this factor for intersection check",
    )

    # Prey confirmation (time-based score accumulation)
    prey_confirmation_mode: str = Field(
        default=_json_config.get("inference", {}).get("prey_confirmation", {}).get(
            "mode", "score_accumulation"
        ),
        description="Confirmation mode: 'score_accumulation' or 'frame_count' (legacy)",
    )
    prey_window_seconds: float = Field(
        default=_json_config.get("inference", {}).get("prey_confirmation", {}).get(
            "window_seconds", 5.0
        ),
        description="Time window in seconds for prey score accumulation",
    )
    prey_score_threshold: float = Field(
        default=_json_config.get("inference", {}).get("prey_confirmation", {}).get(
            "score_threshold", 0.9
        ),
        description="Accumulated score threshold to confirm prey",
    )
    prey_min_detection_score: float = Field(
        default=_json_config.get("inference", {}).get("prey_confirmation", {}).get(
            "min_detection_score", 0.20
        ),
        description="Minimum detection score to count towards accumulation",
    )
    prey_min_detection_count: int = Field(
        default=_json_config.get("inference", {}).get("prey_confirmation", {}).get(
            "min_detection_count", 3
        ),
        description="Minimum number of separate frames with prey detected to confirm "
        "(prevents single high-confidence noise frame from triggering)",
    )
    reset_on_cat_lost_seconds: float = Field(
        default=_json_config.get("inference", {}).get("prey_confirmation", {}).get(
            "reset_on_cat_lost_seconds", 5.0
        ),
        description="Reset prey monitoring if no cat detected for this duration",
    )

    # Two-stage zoom detection
    zoom_detection_enabled: bool = Field(
        default=_json_config.get("inference", {}).get("zoom_detection", {}).get(
            "enabled", False
        ),
        description="Enable two-stage zoom detection: crop cat region from 1080p main stream "
        "and re-run inference for better prey detection at distance",
    )
    zoom_crop_padding: float = Field(
        default=_json_config.get("inference", {}).get("zoom_detection", {}).get(
            "crop_padding", 0.5
        ),
        description="Padding factor around cat bbox for zoom crop (0.5 = 50% expansion each side)",
    )

    # Legacy temporal smoothing (frame-based rolling window)
    window_size: int = Field(
        default=_json_config.get("inference", {}).get("temporal_smoothing", {}).get(
            "window_size", 5
        ),
        description="Legacy: Rolling window size for detection smoothing",
    )
    trigger_count: int = Field(
        default=_json_config.get("inference", {}).get("temporal_smoothing", {}).get(
            "trigger_count", 3
        ),
        description="Legacy: Number of positive detections in window to trigger",
    )

    @field_validator("thresholds")
    @classmethod
    def validate_thresholds(cls, v):
        for class_name, threshold in v.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"threshold for {class_name} must be 0.0-1.0, got {threshold}")
        return v

    @field_validator("box_expansion")
    @classmethod
    def validate_box_expansion(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError(f"box_expansion must be 0.0-2.0, got {v}")
        return v

    @field_validator("window_size")
    @classmethod
    def validate_window_size(cls, v):
        if v < 1 or v > 30:
            raise ValueError(f"window_size must be 1-30, got {v}")
        return v

    @field_validator("trigger_count")
    @classmethod
    def validate_trigger_count(cls, v, info):
        if v < 1:
            raise ValueError(f"trigger_count must be >= 1, got {v}")
        # Note: Can't validate against window_size here as it may not be set yet
        # This is validated at runtime in PreyDetector
        return v

    # Legacy fields for backwards compatibility
    @property
    def confidence_threshold(self) -> float:
        """Engine-level confidence threshold for YOLO postprocessing."""
        return self.engine_confidence_threshold

    @property
    def consecutive_frames_required(self) -> int:
        """Legacy: map to trigger_count."""
        return self.trigger_count

    def get_threshold(self, class_name: str) -> float:
        """Get confidence threshold for a specific class."""
        return self.thresholds.get(class_name.lower(), 0.5)


class JammerConfig(BaseSettings):
    """RFID jammer (relay) configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_JAMMER_"}

    gpio_pin: int = Field(
        default=_json_config.get("jammer", {}).get("gpio_pin", 17),
        description="BCM GPIO pin for relay control",
    )
    active_high: bool = Field(
        default=_json_config.get("jammer", {}).get("active_high", True),
        description="True if HIGH activates relay (normally open)",
    )
    lockdown_duration_seconds: int = Field(
        default=_json_config.get("jammer", {}).get("lockdown_duration_seconds", 300),
        description="Duration to keep jammer active (5 minutes default)",
    )
    cooldown_duration_seconds: int = Field(
        default=_json_config.get("jammer", {}).get("cooldown_duration_seconds", 30),
        description="Cooldown period before returning to IDLE",
    )


class AudioConfig(BaseSettings):
    """Audio deterrent configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_AUDIO_"}

    enabled: bool = Field(
        default=_json_config.get("audio", {}).get("enabled", True),
        description="Enable audio deterrent",
    )
    sound_file: str = Field(
        default=_json_config.get("audio", {}).get(
            "sound_file", str(PROJECT_ROOT / "sounds" / "hawk_screech.wav")
        ),
        description="Path to deterrent sound file",
    )
    volume: float = Field(
        default=_json_config.get("audio", {}).get("volume", 0.8),
        description="Playback volume (0.0 to 1.0)",
    )


class APIConfig(BaseSettings):
    """FastAPI server configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_API_"}

    enabled: bool = Field(
        default=_json_config.get("api", {}).get("enabled", True),
        description="Enable REST API server",
    )
    host: str = Field(
        default=_json_config.get("api", {}).get("host", "0.0.0.0"),
        description="API server bind host",
    )
    port: int = Field(
        default=_json_config.get("api", {}).get("port", 8080),
        description="API server port",
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_LOGGING_"}

    level: str = Field(
        default=_json_config.get("logging", {}).get("level", "INFO"),
        description="Log level",
    )
    file: str = Field(
        default=_json_config.get("logging", {}).get(
            "file", str(RUNTIME_DIR / "logs" / "mousehunter.log")
        ),
        description="Log file path",
    )


class RecordingConfig(BaseSettings):
    """Video recording and evidence configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_RECORDING_"}

    output_dir: str = Field(
        default=_json_config.get("recording", {}).get(
            "output_dir", str(RUNTIME_DIR / "recordings")
        ),
        description="Directory for video recordings",
    )
    evidence_dir: str = Field(
        default=_json_config.get("recording", {}).get(
            "evidence_dir", str(RUNTIME_DIR / "evidence")
        ),
        description="Directory for detection evidence (images)",
    )
    max_age_days: int = Field(
        default=_json_config.get("recording", {}).get("max_age_days", 7),
        description="Maximum age of recordings before cleanup",
    )
    post_roll_seconds: float = Field(
        default=_json_config.get("recording", {}).get("post_roll_seconds", 15.0),
        description="Seconds of post-event footage to capture for evidence",
    )
    evidence_format: str = Field(
        default=_json_config.get("recording", {}).get("evidence_format", "video"),
        description="Evidence format: 'video' for H.264 MP4, 'frames' for legacy JPEGs",
    )


class TrainingDataConfig(BaseSettings):
    """Training data capture configuration for model improvement."""

    model_config = {"env_prefix": "MOUSEHUNTER_TRAINING_"}

    enabled: bool = Field(
        default=_json_config.get("training_data", {}).get("enabled", False),
        description="Enable training data capture",
    )
    periodic_interval_minutes: int = Field(
        default=_json_config.get("training_data", {}).get("periodic_interval_minutes", 30),
        description="Interval for periodic snapshots (0 to disable)",
    )
    capture_cat_only: bool = Field(
        default=_json_config.get("training_data", {}).get("capture_cat_only", True),
        description="Capture when cat detected without prey",
    )
    cat_only_delay_seconds: float = Field(
        default=_json_config.get("training_data", {}).get("cat_only_delay_seconds", 2.0),
        description="Delay before capturing cat-only (to ensure stable detection)",
    )
    capture_near_miss: bool = Field(
        default=_json_config.get("training_data", {}).get("capture_near_miss", True),
        description="Capture when VERIFYING resets to IDLE (almost triggered)",
    )
    include_detections_json: bool = Field(
        default=_json_config.get("training_data", {}).get("include_detections_json", True),
        description="Include detection metadata (bboxes, confidences) with images",
    )
    max_images_per_day: int = Field(
        default=_json_config.get("training_data", {}).get("max_images_per_day", 100),
        description="Maximum training images to capture per day (0 for unlimited)",
    )
    use_inference_resolution: bool = Field(
        default=_json_config.get("training_data", {}).get("use_inference_resolution", True),
        description="Save at inference resolution (640x640) instead of full resolution",
    )
    remote_path: str = Field(
        default=_json_config.get("training_data", {}).get("remote_path", "MouseHunter/training"),
        description="Remote path for training data (under cloud storage)",
    )
    local_dir: str = Field(
        default=_json_config.get("training_data", {}).get(
            "local_dir", str(RUNTIME_DIR / "training_data")
        ),
        description="Local directory for training data before upload",
    )


class CloudStorageConfig(BaseSettings):
    """Cloud storage configuration for long-term archival via rclone."""

    model_config = {"env_prefix": "MOUSEHUNTER_CLOUD_"}

    enabled: bool = Field(
        default=_json_config.get("cloud_storage", {}).get("enabled", False),
        description="Enable cloud storage uploads",
    )
    rclone_remote: str = Field(
        default=_json_config.get("cloud_storage", {}).get("rclone_remote", ""),
        description="rclone remote name (e.g., 'gdrive', 'dropbox')",
    )
    remote_path: str = Field(
        default=_json_config.get("cloud_storage", {}).get("remote_path", "MouseHunter"),
        description="Base path on remote storage",
    )
    upload_after_detection: bool = Field(
        default=_json_config.get("cloud_storage", {}).get("upload_after_detection", True),
        description="Automatically upload evidence after each detection",
    )
    sync_interval_minutes: int = Field(
        default=_json_config.get("cloud_storage", {}).get("sync_interval_minutes", 60),
        description="Periodic sync interval (0 to disable)",
    )
    delete_local_after_upload: bool = Field(
        default=_json_config.get("cloud_storage", {}).get("delete_local_after_upload", False),
        description="Delete local files after successful upload",
    )
    retention_days_cloud: int = Field(
        default=_json_config.get("cloud_storage", {}).get("retention_days_cloud", 365),
        description="Days to keep files in cloud storage (0 for forever)",
    )
    bandwidth_limit: str = Field(
        default=_json_config.get("cloud_storage", {}).get("bandwidth_limit", ""),
        description="Bandwidth limit for uploads (e.g., '1M' for 1MB/s, empty for unlimited)",
    )


# Global configuration instances
telegram_config = TelegramConfig()
camera_config = CameraConfig()
inference_config = InferenceConfig()
jammer_config = JammerConfig()
audio_config = AudioConfig()
api_config = APIConfig()
logging_config = LoggingConfig()
recording_config = RecordingConfig()
cloud_storage_config = CloudStorageConfig()
training_data_config = TrainingDataConfig()


def setup_logging() -> None:
    """Configure logging for the application with log rotation."""
    from logging.handlers import RotatingFileHandler

    log_dir = Path(logging_config.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # Use RotatingFileHandler to prevent disk fill (10MB max, keep 5 backups)
    file_handler = RotatingFileHandler(
        logging_config.file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, logging_config.level.upper()))
    root_logger.addHandler(file_handler)
    root_logger.addHandler(logging.StreamHandler())

    logger.info(
        f"Logging configured: level={logging_config.level}, "
        f"file={logging_config.file} (rotating, 10MB max, 5 backups)"
    )


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they don't exist."""
    dirs = [
        Path(recording_config.output_dir),
        Path(recording_config.evidence_dir),
        Path(logging_config.file).parent,
        Path(training_data_config.local_dir),
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {d}")
