"""
Prey Detector - Hierarchical spatial prey detection with temporal smoothing

Implements the "Cat as Anchor" detection strategy:
1. Cat must be detected (high confidence anchor)
2. Prey (rodent) must be detected (v3: bird removed for better quantization)
3. Prey must spatially intersect with cat (carrying validation)
4. Uses rolling window (3-of-5) instead of consecutive frames

This approach dramatically reduces false positives from:
- Random objects detected as "rodent" elsewhere in frame
- Single-frame detection glitches
- Shadows, toys, or other non-prey objects
"""

import logging
import threading
from collections import deque
from dataclasses import dataclass, field
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
    VERIFYING = auto()  # Potential detection, accumulating evidence
    CONFIRMED = auto()  # Prey confirmed, trigger actions


@dataclass
class SpatialMatch:
    """A validated cat+prey spatial match."""

    cat: Detection
    prey: Detection
    intersection_type: str  # "overlap" or "proximity"

    @property
    def confidence(self) -> float:
        """Combined confidence score."""
        return (self.cat.confidence + self.prey.confidence) / 2


@dataclass
class PreyDetectionEvent:
    """Event data when prey is confirmed."""

    timestamp: datetime
    detections_in_window: int
    window_size: int
    confidence: float
    cat_detection: Detection
    prey_detection: Detection
    frame: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "detections_in_window": self.detections_in_window,
            "window_size": self.window_size,
            "confidence": self.confidence,
            "cat": self.cat_detection.to_dict(),
            "prey": self.prey_detection.to_dict(),
        }


@dataclass
class FrameResult:
    """Result of processing a single frame."""

    has_cat: bool = False
    has_valid_prey: bool = False
    cat_detection: Detection | None = None
    prey_detection: Detection | None = None
    match: SpatialMatch | None = None
    all_detections: list[Detection] = field(default_factory=list)


class PreyDetector:
    """
    Hierarchical spatial prey detector with temporal smoothing.

    Detection strategy:
    1. Anchor: Detect cat with high confidence (98% mAP)
    2. Filter: Detect prey (rodent) with class-specific thresholds
    3. Validate: Prey must spatially intersect expanded cat bounding box
    4. Smooth: Use rolling window (3-of-5) to filter temporal noise
    """

    # Prey classes to detect (v3: rodent only, bird removed for better quantization)
    PREY_CLASSES = {"rodent"}

    def __init__(
        self,
        engine: HailoEngine | None = None,
        thresholds: dict[str, float] | None = None,
        window_size: int = 5,
        trigger_count: int = 3,
        spatial_validation_enabled: bool = True,
        box_expansion: float = 0.25,
    ):
        """
        Initialize the prey detector.

        Args:
            engine: Hailo inference engine (created if not provided)
            thresholds: Per-class confidence thresholds
            window_size: Rolling window size for temporal smoothing
            trigger_count: Positive detections required in window to trigger
            spatial_validation_enabled: Require prey near cat
            box_expansion: Expand cat box by this factor for intersection
        """
        self.thresholds = thresholds or {"cat": 0.55, "rodent": 0.45}
        self.window_size = window_size
        self.trigger_count = trigger_count
        self.spatial_validation_enabled = spatial_validation_enabled
        self.box_expansion = box_expansion

        # Inference engine
        if engine is None:
            from .hailo_engine import get_hailo_engine

            engine = get_hailo_engine()
        self.engine = engine

        # State
        self._state = DetectionState.IDLE
        self._lock = threading.Lock()

        # Rolling window for temporal smoothing
        # Stores True/False for each frame (prey detected or not)
        self._detection_window: deque[bool] = deque(maxlen=window_size)

        # Store last valid match for event reporting
        self._last_match: SpatialMatch | None = None

        # Detection history (for analysis/debugging)
        self._history: deque[DetectionFrame] = deque(maxlen=100)

        # Callbacks
        self._on_prey_confirmed_callbacks: list[Callable[[PreyDetectionEvent], None]] = []
        self._on_state_change_callbacks: list[Callable[[DetectionState], None]] = []

        logger.info(
            f"PreyDetector initialized: thresholds={thresholds}, "
            f"window={window_size}, trigger={trigger_count}, "
            f"spatial={spatial_validation_enabled}, expansion={box_expansion}"
        )

    @property
    def state(self) -> DetectionState:
        """Get current detection state."""
        return self._state

    @property
    def positive_count(self) -> int:
        """Get count of positive detections in current window."""
        return sum(self._detection_window)

    @property
    def window_fill(self) -> int:
        """Get number of frames in window."""
        return len(self._detection_window)

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

        # Evaluate frame with spatial logic
        frame_result = self._evaluate_frame(result)

        # Update rolling window and state machine
        self._update_state(frame_result, frame)

        return result

    def _evaluate_frame(self, result: DetectionFrame) -> FrameResult:
        """
        Evaluate a frame using hierarchical spatial logic.

        Returns:
            FrameResult with validation details
        """
        frame_result = FrameResult(all_detections=result.detections)

        # Step 1: Find cat (the anchor)
        cat = self._get_best_detection(result.detections, "cat")
        if cat:
            frame_result.has_cat = True
            frame_result.cat_detection = cat

        # Step 2: If no cat, no valid prey detection possible
        if not frame_result.has_cat:
            return frame_result

        # Step 3: Find prey and validate spatially
        for prey_class in self.PREY_CLASSES:
            prey = self._get_best_detection(result.detections, prey_class)
            if prey is None:
                continue

            # Step 4: Spatial validation (if enabled)
            if self.spatial_validation_enabled:
                match = self._check_spatial_match(cat, prey)
                if match:
                    frame_result.has_valid_prey = True
                    frame_result.prey_detection = prey
                    frame_result.match = match
                    logger.debug(
                        f"Valid prey: {prey.class_name} ({prey.confidence:.2f}) "
                        f"intersects cat via {match.intersection_type}"
                    )
                    break  # Found valid prey, stop searching
            else:
                # No spatial validation - any prey with cat is valid
                frame_result.has_valid_prey = True
                frame_result.prey_detection = prey
                frame_result.match = SpatialMatch(
                    cat=cat, prey=prey, intersection_type="disabled"
                )
                break

        return frame_result

    def _get_best_detection(
        self, detections: list[Detection], class_name: str
    ) -> Detection | None:
        """Get highest confidence detection of a class above threshold."""
        threshold = self.thresholds.get(class_name.lower(), 0.5)
        candidates = [
            d
            for d in detections
            if d.class_name.lower() == class_name.lower()
            and d.confidence >= threshold
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    def _check_spatial_match(
        self, cat: Detection, prey: Detection
    ) -> SpatialMatch | None:
        """
        Check if prey spatially intersects with cat.

        Uses relaxed intersection:
        1. Direct overlap between boxes
        2. Prey center within expanded cat box

        Args:
            cat: Cat detection
            prey: Prey detection

        Returns:
            SpatialMatch if valid, None otherwise
        """
        cat_box = cat.bbox
        prey_box = prey.bbox

        # Check 1: Direct intersection
        if cat_box.intersects(prey_box):
            return SpatialMatch(cat=cat, prey=prey, intersection_type="overlap")

        # Check 2: Prey center within expanded cat box
        expanded_cat = cat_box.expanded(self.box_expansion)
        if expanded_cat.contains_point(prey_box.center_x, prey_box.center_y):
            return SpatialMatch(cat=cat, prey=prey, intersection_type="proximity")

        return None

    def _update_state(self, frame_result: FrameResult, frame: np.ndarray) -> None:
        """Update detection state based on frame result and rolling window."""
        # Collect callbacks to invoke outside the lock (prevents deadlock)
        pending_callbacks: list[tuple[Callable, tuple]] = []

        with self._lock:
            # Add result to rolling window
            self._detection_window.append(frame_result.has_valid_prey)

            # Store last match for event creation
            if frame_result.match:
                self._last_match = frame_result.match

            # Count positive detections in window
            positive_count = sum(self._detection_window)

            logger.debug(
                f"Window: {positive_count}/{len(self._detection_window)} "
                f"(need {self.trigger_count}), "
                f"cat={frame_result.has_cat}, prey={frame_result.has_valid_prey}"
            )

            # State transitions based on window
            if positive_count >= self.trigger_count:
                # Enough evidence - confirm prey
                if self._state != DetectionState.CONFIRMED:
                    # Prepare prey confirmation (callbacks invoked after lock release)
                    event = self._prepare_prey_confirmation(frame, positive_count)
                    if event:
                        pending_callbacks.extend(
                            [(cb, (event,)) for cb in self._on_prey_confirmed_callbacks]
                        )
            elif positive_count > 0:
                # Some evidence - verifying
                if self._state == DetectionState.IDLE:
                    self._state = DetectionState.VERIFYING
                    logger.info(f"State transition: IDLE -> VERIFYING")
                    pending_callbacks.extend(
                        [(cb, (DetectionState.VERIFYING,)) for cb in self._on_state_change_callbacks]
                    )
            else:
                # No evidence - idle
                if self._state == DetectionState.VERIFYING:
                    self._state = DetectionState.IDLE
                    self._last_match = None
                    logger.info(f"State transition: VERIFYING -> IDLE")
                    pending_callbacks.extend(
                        [(cb, (DetectionState.IDLE,)) for cb in self._on_state_change_callbacks]
                    )

        # Invoke callbacks OUTSIDE the lock to prevent deadlock
        for callback, args in pending_callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _transition_to(self, new_state: DetectionState) -> list[tuple[Callable, tuple]]:
        """
        Transition to a new state.

        Returns list of (callback, args) tuples to be invoked OUTSIDE the lock.
        """
        old_state = self._state
        self._state = new_state
        logger.info(f"State transition: {old_state.name} -> {new_state.name}")

        # Return callbacks to be invoked outside the lock
        return [(cb, (new_state,)) for cb in self._on_state_change_callbacks]

    def _prepare_prey_confirmation(
        self, frame: np.ndarray, positive_count: int
    ) -> PreyDetectionEvent | None:
        """
        Prepare prey confirmation event (called while holding lock).

        Returns the event to be passed to callbacks OUTSIDE the lock.
        """
        self._state = DetectionState.CONFIRMED
        logger.info(f"State transition: {self._state.name} -> CONFIRMED")

        if self._last_match is None:
            logger.error("Prey confirmed but no match stored")
            return None

        event = PreyDetectionEvent(
            timestamp=datetime.now(),
            detections_in_window=positive_count,
            window_size=self.window_size,
            confidence=self._last_match.confidence,
            cat_detection=self._last_match.cat,
            prey_detection=self._last_match.prey,
            frame=frame.copy(),
        )

        logger.warning(
            f"PREY CONFIRMED! {positive_count}/{self.window_size} frames, "
            f"prey={self._last_match.prey.class_name} "
            f"({self._last_match.prey.confidence:.2f}), "
            f"cat=({self._last_match.cat.confidence:.2f})"
        )

        return event

    def reset(self) -> None:
        """Reset detector to IDLE state."""
        pending_callbacks: list[tuple[Callable, tuple]] = []

        with self._lock:
            self._detection_window.clear()
            self._last_match = None
            if self._state != DetectionState.IDLE:
                old_state = self._state
                self._state = DetectionState.IDLE
                logger.info(f"State transition: {old_state.name} -> IDLE (reset)")
                pending_callbacks = [
                    (cb, (DetectionState.IDLE,)) for cb in self._on_state_change_callbacks
                ]
            logger.info("PreyDetector reset to IDLE")

        # Invoke callbacks OUTSIDE the lock
        for callback, args in pending_callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Reset callback error: {e}")

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
            "window_fill": self.window_fill,
            "positive_count": self.positive_count,
            "window_size": self.window_size,
            "trigger_count": self.trigger_count,
            "thresholds": self.thresholds,
            "spatial_validation": self.spatial_validation_enabled,
            "box_expansion": self.box_expansion,
            "history_size": len(self._history),
            "last_match": (
                {
                    "cat": self._last_match.cat.to_dict(),
                    "prey": self._last_match.prey.to_dict(),
                    "type": self._last_match.intersection_type,
                }
                if self._last_match
                else None
            ),
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
            thresholds=inference_config.thresholds,
            window_size=inference_config.window_size,
            trigger_count=inference_config.trigger_count,
            spatial_validation_enabled=inference_config.spatial_validation_enabled,
            box_expansion=inference_config.box_expansion,
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
    """Test the prey detector with simulated detections."""
    logging.basicConfig(level=logging.DEBUG)
    print("=== Prey Detector Test (Spatial + Rolling Window) ===")

    detector = PreyDetector(
        thresholds={"cat": 0.55, "rodent": 0.45},
        window_size=5,
        trigger_count=3,
        spatial_validation_enabled=True,
        box_expansion=0.25,
    )

    def on_prey(event: PreyDetectionEvent):
        print(f"\n*** PREY ALERT! ***")
        print(f"  Prey: {event.prey_detection.class_name} ({event.confidence:.2f})")
        print(f"  Window: {event.detections_in_window}/{event.window_size}")

    def on_state(state: DetectionState):
        print(f"State -> {state.name}")

    detector.on_prey_confirmed(on_prey)
    detector.on_state_change(on_state)

    print("\nProcessing test frames...")
    for i in range(20):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)
        print(
            f"Frame {i+1}: {len(result.detections)} detections, "
            f"window={detector.positive_count}/{detector.window_fill}, "
            f"state={detector.state.name}"
        )

    print(f"\nFinal Status:")
    status = detector.get_status()
    for key, value in status.items():
        if key != "engine_status":
            print(f"  {key}: {value}")

    detector.cleanup()


if __name__ == "__main__":
    test_detector()
