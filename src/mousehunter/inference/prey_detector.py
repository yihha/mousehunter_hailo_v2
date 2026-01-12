"""
Prey Detector - High-level prey detection with debouncing

Implements the detection logic with:
- Consecutive frame requirement (debouncing)
- State tracking for reliable prey detection
- Integration with camera and inference engine
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Callable

import numpy as np

from .detection import Detection, DetectionFrame
from .hailo_engine import HailoEngine

logger = logging.getLogger(__name__)


class DetectionState(Enum):
    """States for the prey detection state machine."""

    IDLE = auto()  # No detection, monitoring
    VERIFYING = auto()  # Potential detection, verifying with consecutive frames
    CONFIRMED = auto()  # Prey confirmed, trigger actions


@dataclass
class PreyDetectionEvent:
    """Event data when prey is confirmed."""

    timestamp: datetime
    consecutive_count: int
    confidence: float
    detection: Detection
    frame: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "consecutive_count": self.consecutive_count,
            "confidence": self.confidence,
            "detection": self.detection.to_dict(),
        }


class PreyDetector:
    """
    High-level prey detector with debouncing.

    Monitors inference results and triggers only when prey is
    detected in N consecutive frames, preventing false positives
    from shadows, reflections, or single-frame anomalies.
    """

    def __init__(
        self,
        engine: HailoEngine | None = None,
        consecutive_frames_required: int = 3,
        confidence_threshold: float = 0.6,
        prey_class_name: str = "prey",
    ):
        """
        Initialize the prey detector.

        Args:
            engine: Hailo inference engine (created if not provided)
            consecutive_frames_required: Frames needed to confirm prey
            confidence_threshold: Minimum confidence for prey class
            prey_class_name: Name of the prey class in model
        """
        self.consecutive_frames_required = consecutive_frames_required
        self.confidence_threshold = confidence_threshold
        self.prey_class_name = prey_class_name.lower()

        # Inference engine
        if engine is None:
            from .hailo_engine import get_hailo_engine

            engine = get_hailo_engine()
        self.engine = engine

        # State
        self._state = DetectionState.IDLE
        self._consecutive_count = 0
        self._last_detection: Detection | None = None
        self._lock = threading.Lock()

        # Detection history (for analysis)
        self._history: deque[DetectionFrame] = deque(maxlen=100)

        # Callbacks
        self._on_prey_confirmed_callbacks: list[Callable[[PreyDetectionEvent], None]] = []
        self._on_state_change_callbacks: list[Callable[[DetectionState], None]] = []

        logger.info(
            f"PreyDetector initialized: consecutive_frames={consecutive_frames_required}, "
            f"threshold={confidence_threshold}"
        )

    @property
    def state(self) -> DetectionState:
        """Get current detection state."""
        return self._state

    @property
    def consecutive_count(self) -> int:
        """Get current consecutive detection count."""
        return self._consecutive_count

    def process_frame(
        self, frame: np.ndarray, timestamp: datetime | None = None
    ) -> DetectionFrame:
        """
        Process a single frame for prey detection.

        Args:
            frame: Input frame (RGB)
            timestamp: Frame timestamp

        Returns:
            DetectionFrame with inference results
        """
        timestamp = timestamp or datetime.now()

        # Run inference
        result = self.engine.infer(frame)

        # Add to history
        self._history.append(result)

        # Update state machine
        self._update_state(result, frame)

        return result

    def _update_state(self, result: DetectionFrame, frame: np.ndarray) -> None:
        """Update the detection state machine."""
        with self._lock:
            # Check for prey in this frame
            prey_detection = self._get_prey_detection(result)

            if prey_detection:
                self._consecutive_count += 1
                self._last_detection = prey_detection

                logger.debug(
                    f"Prey detected: {prey_detection.confidence:.2f}, "
                    f"consecutive={self._consecutive_count}/{self.consecutive_frames_required}"
                )

                if self._state == DetectionState.IDLE:
                    self._transition_to(DetectionState.VERIFYING)

                if self._consecutive_count >= self.consecutive_frames_required:
                    if self._state != DetectionState.CONFIRMED:
                        self._confirm_prey(frame)
            else:
                # No prey detected - reset
                if self._consecutive_count > 0:
                    logger.debug(f"Detection broken at {self._consecutive_count} frames")
                self._consecutive_count = 0
                self._last_detection = None

                if self._state == DetectionState.VERIFYING:
                    self._transition_to(DetectionState.IDLE)

    def _get_prey_detection(self, result: DetectionFrame) -> Detection | None:
        """Get prey detection if above threshold."""
        for detection in result.detections:
            if (
                detection.class_name.lower() == self.prey_class_name
                and detection.confidence >= self.confidence_threshold
            ):
                return detection
        return None

    def _transition_to(self, new_state: DetectionState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        logger.info(f"State transition: {old_state.name} -> {new_state.name}")

        for callback in self._on_state_change_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    def _confirm_prey(self, frame: np.ndarray) -> None:
        """Confirm prey detection and trigger callbacks."""
        self._transition_to(DetectionState.CONFIRMED)

        event = PreyDetectionEvent(
            timestamp=datetime.now(),
            consecutive_count=self._consecutive_count,
            confidence=self._last_detection.confidence if self._last_detection else 0.0,
            detection=self._last_detection,
            frame=frame.copy(),
        )

        logger.warning(
            f"PREY CONFIRMED after {self._consecutive_count} frames! "
            f"Confidence: {event.confidence:.2f}"
        )

        # Notify callbacks
        for callback in self._on_prey_confirmed_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Prey confirmed callback error: {e}")

    def reset(self) -> None:
        """Reset detector to IDLE state."""
        with self._lock:
            self._consecutive_count = 0
            self._last_detection = None
            if self._state != DetectionState.IDLE:
                self._transition_to(DetectionState.IDLE)
            logger.info("PreyDetector reset to IDLE")

    def on_prey_confirmed(self, callback: Callable[[PreyDetectionEvent], None]) -> None:
        """Register callback for when prey is confirmed."""
        self._on_prey_confirmed_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[DetectionState], None]) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get detector status."""
        return {
            "state": self._state.name,
            "consecutive_count": self._consecutive_count,
            "consecutive_required": self.consecutive_frames_required,
            "confidence_threshold": self.confidence_threshold,
            "prey_class": self.prey_class_name,
            "history_size": len(self._history),
            "last_detection": self._last_detection.to_dict() if self._last_detection else None,
            "engine_status": self.engine.get_status(),
        }

    def get_recent_detections(self, count: int = 10) -> list[DetectionFrame]:
        """Get recent detection frames."""
        return list(self._history)[-count:]

    def cleanup(self) -> None:
        """Clean up resources."""
        self.engine.cleanup()
        logger.info("PreyDetector cleaned up")


# Factory function
def _create_default_detector() -> PreyDetector:
    """Create detector from config."""
    try:
        from mousehunter.config import inference_config

        return PreyDetector(
            consecutive_frames_required=inference_config.consecutive_frames_required,
            confidence_threshold=inference_config.confidence_threshold,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return PreyDetector()


# Global instance (lazy)
_detector_instance: PreyDetector | None = None


def get_prey_detector() -> PreyDetector:
    """Get or create the global prey detector."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = _create_default_detector()
    return _detector_instance


def test_detector() -> None:
    """Test the prey detector."""
    logging.basicConfig(level=logging.DEBUG)
    print("=== Prey Detector Test ===")

    detector = PreyDetector(consecutive_frames_required=3, confidence_threshold=0.5)

    def on_prey(event: PreyDetectionEvent):
        print(f"PREY ALERT! {event}")

    def on_state(state: DetectionState):
        print(f"State: {state.name}")

    detector.on_prey_confirmed(on_prey)
    detector.on_state_change(on_state)

    print("Processing test frames...")
    for i in range(20):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)
        print(f"Frame {i+1}: {len(result.detections)} detections, state={detector.state.name}")

    print(f"\nStatus: {detector.get_status()}")
    detector.cleanup()


if __name__ == "__main__":
    test_detector()
