"""
Configuration management for MouseHunter using Pydantic settings.

Loads configuration from:
1. config/config.json (defaults)
2. Environment variables (override with MOUSEHUNTER_ prefix)
"""

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
RUNTIME_DIR = PROJECT_ROOT / "runtime"


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


class InferenceConfig(BaseSettings):
    """Hailo-8L inference engine configuration."""

    model_config = {"env_prefix": "MOUSEHUNTER_INFERENCE_"}

    model_path: str = Field(
        default=_json_config.get("inference", {}).get(
            "model_path", str(PROJECT_ROOT / "models" / "yolov8n_catprey.hef")
        ),
        description="Path to compiled Hailo HEF model",
    )
    confidence_threshold: float = Field(
        default=_json_config.get("inference", {}).get("confidence_threshold", 0.60),
        description="Minimum confidence for prey detection trigger",
    )
    consecutive_frames_required: int = Field(
        default=_json_config.get("inference", {}).get("consecutive_frames_required", 3),
        description="Consecutive frames with prey detection to trigger lockdown",
    )
    classes: dict[str, str] = Field(
        default=_json_config.get("inference", {}).get("classes", {"0": "cat", "1": "prey"}),
        description="Class ID to name mapping",
    )

    @field_validator("confidence_threshold")
    @classmethod
    def validate_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1")
        return v


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


# Global configuration instances
telegram_config = TelegramConfig()
camera_config = CameraConfig()
inference_config = InferenceConfig()
jammer_config = JammerConfig()
audio_config = AudioConfig()
api_config = APIConfig()
logging_config = LoggingConfig()
recording_config = RecordingConfig()


def setup_logging() -> None:
    """Configure logging for the application."""
    log_dir = Path(logging_config.file).parent
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, logging_config.level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logging_config.file),
            logging.StreamHandler(),
        ],
    )
    logger.info(f"Logging configured: level={logging_config.level}, file={logging_config.file}")


def ensure_runtime_dirs() -> None:
    """Create runtime directories if they don't exist."""
    dirs = [
        Path(recording_config.output_dir),
        Path(recording_config.evidence_dir),
        Path(logging_config.file).parent,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory exists: {d}")
