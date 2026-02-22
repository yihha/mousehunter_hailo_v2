"""
Prey Detector - Hierarchical spatial prey detection with time-based score accumulation

Implements the "Cat as Anchor" detection strategy:
1. Cat must be detected (high confidence anchor)
2. Prey (rodent) must be detected with spatial validation
3. Uses time-based score accumulation instead of frame counting
   - More robust for sporadic/inconsistent detections
   - Based on radar tracking best practices (score-based confirmation)

Detection flow:
- Cat detected → Start monitoring window
- Each rodent detection → Add confidence to accumulated score
- Score >= threshold within window → CONFIRM PREY
- No cat for N seconds → Reset monitoring
"""

import logging
import threading
import time
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

    IDLE = auto()  # No cat detected, not monitoring
    MONITORING = auto()  # Cat detected, monitoring for prey
    VERIFYING = auto()  # Prey detected, accumulating score
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
class PreyScoreEntry:
    """A single prey detection with timestamp for score accumulation."""
    timestamp: float  # time.time()
    confidence: float
    detection: Detection


@dataclass
class PreyDetectionEvent:
    """Event data when prey is confirmed."""

    timestamp: datetime
    accumulated_score: float
    score_threshold: float
    detection_count: int
    window_seconds: float
    confidence: float
    cat_detection: Detection
    prey_detection: Detection
    frame: np.ndarray | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "accumulated_score": self.accumulated_score,
            "score_threshold": self.score_threshold,
            "detection_count": self.detection_count,
            "window_seconds": self.window_seconds,
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
    Hierarchical spatial prey detector with time-based score accumulation.

    Detection strategy:
    1. Anchor: Detect cat with high confidence
    2. Filter: Detect prey (rodent) with class-specific thresholds
    3. Validate: Prey must spatially intersect expanded cat bounding box
    4. Confirm: Accumulate prey confidence scores over time window
       - Triggers when accumulated score >= threshold
       - More robust for sporadic detections than frame counting
    """

    # Prey classes to detect (v3: rodent only)
    PREY_CLASSES = {"rodent"}

    def __init__(
        self,
        engine: HailoEngine | None = None,
        thresholds: dict[str, float] | None = None,
        spatial_validation_enabled: bool = True,
        box_expansion: float = 0.25,
        # Time-based score accumulation (new approach)
        prey_confirmation_mode: str = "score_accumulation",
        prey_window_seconds: float = 5.0,
        prey_score_threshold: float = 0.9,
        prey_min_detection_score: float = 0.20,
        prey_min_detection_count: int = 3,
        reset_on_cat_lost_seconds: float = 5.0,
        # Legacy frame-based (kept for backwards compatibility)
        window_size: int = 5,
        trigger_count: int = 3,
    ):
        """
        Initialize the prey detector.

        Args:
            engine: Hailo inference engine (created if not provided)
            thresholds: Per-class confidence thresholds
            spatial_validation_enabled: Require prey near cat
            box_expansion: Expand cat box by this factor for intersection

            # Score accumulation parameters (recommended)
            prey_confirmation_mode: "score_accumulation" or "frame_count" (legacy)
            prey_window_seconds: Time window for score accumulation
            prey_score_threshold: Accumulated score needed to confirm
            prey_min_detection_score: Minimum detection score to count
            prey_min_detection_count: Minimum separate frames with prey to confirm
            reset_on_cat_lost_seconds: Reset if no cat for this duration

            # Legacy parameters (for backwards compatibility)
            window_size: Rolling window size (legacy frame_count mode)
            trigger_count: Positive detections required (legacy frame_count mode)
        """
        self.thresholds = thresholds or {"cat": 0.60, "rodent": 0.25}
        self.spatial_validation_enabled = spatial_validation_enabled
        self.box_expansion = box_expansion

        # Score accumulation config
        self.confirmation_mode = prey_confirmation_mode
        self.prey_window_seconds = prey_window_seconds
        self.prey_score_threshold = prey_score_threshold
        self.prey_min_detection_score = prey_min_detection_score
        self.prey_min_detection_count = prey_min_detection_count
        self.reset_on_cat_lost_seconds = reset_on_cat_lost_seconds

        # Legacy config (for frame_count mode)
        self.window_size = window_size
        self.trigger_count = trigger_count

        # Inference engine
        if engine is None:
            from .hailo_engine import get_hailo_engine
            engine = get_hailo_engine()
        self.engine = engine

        # State
        self._state = DetectionState.IDLE
        self._lock = threading.Lock()

        # Time-based score accumulation
        self._prey_scores: list[PreyScoreEntry] = []  # Recent prey detections with timestamps
        self._last_cat_time: float | None = None  # Last time cat was detected
        self._monitoring_start_time: float | None = None  # When monitoring started

        # Legacy: Rolling window for frame-based mode
        self._detection_window: deque[bool] = deque(maxlen=window_size)

        # Store last valid match for event reporting
        self._last_match: SpatialMatch | None = None

        # Detection history (for analysis/debugging)
        self._history: deque[DetectionFrame] = deque(maxlen=100)

        # Callbacks
        self._on_prey_confirmed_callbacks: list[Callable[[PreyDetectionEvent], None]] = []
        self._on_state_change_callbacks: list[Callable[[DetectionState], None]] = []
        # Training data callbacks
        self._on_cat_only_callbacks: list[Callable[[Detection, list[Detection], np.ndarray], None]] = []
        self._on_near_miss_callbacks: list[Callable[[float, list[Detection], np.ndarray], None]] = []
        self._last_cat_only_callback_time: float = 0
        self._cat_only_callback_cooldown: float = 5.0  # Minimum seconds between cat-only callbacks

        logger.info(
            f"PreyDetector initialized: mode={prey_confirmation_mode}, "
            f"thresholds={thresholds}, window={prey_window_seconds}s, "
            f"score_threshold={prey_score_threshold}, "
            f"min_score={prey_min_detection_score}, min_count={prey_min_detection_count}, "
            f"spatial={spatial_validation_enabled}, expansion={box_expansion}"
        )

    @property
    def state(self) -> DetectionState:
        """Get current detection state."""
        return self._state

    @property
    def accumulated_score(self) -> float:
        """Get current accumulated prey score within window."""
        now = time.time()
        valid_scores = [
            entry.confidence for entry in self._prey_scores
            if now - entry.timestamp < self.prey_window_seconds
        ]
        return sum(valid_scores)

    @property
    def detection_count_in_window(self) -> int:
        """Get count of prey detections in current window."""
        now = time.time()
        return sum(
            1 for entry in self._prey_scores
            if now - entry.timestamp < self.prey_window_seconds
        )

    @property
    def positive_count(self) -> int:
        """Legacy: Get count of positive detections in frame window."""
        return sum(self._detection_window)

    @property
    def window_fill(self) -> int:
        """Legacy: Get number of frames in window."""
        return len(self._detection_window)

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: datetime | None = None,
        zoom_frame_provider: Callable[[Detection], np.ndarray | None] | None = None,
    ) -> DetectionFrame:
        """
        Process a single frame for prey detection.

        Implements two-stage detection:
        1. Stage 1: Run inference on the lores (640x640) frame
        2. Stage 2 (zoom): If cat found but no prey, crop from high-res main
           stream around the cat, resize to 640x640, and re-run inference.
           Any prey found in the zoomed crop is valid (no spatial check needed
           since the crop IS the cat region).

        Args:
            frame: Input frame (RGB, typically 640x640 lores)
            timestamp: Frame timestamp
            zoom_frame_provider: Optional callback that takes a cat Detection
                and returns a 640x640 RGB crop from the main stream, or None.

        Returns:
            DetectionFrame with inference results
        """
        timestamp = timestamp or datetime.now()

        # Stage 1: Run inference on lores frame
        result = self.engine.infer(frame)

        # Add to history
        self._history.append(result)

        # Evaluate frame with spatial logic
        frame_result = self._evaluate_frame(result)

        # Stage 2: If cat found but no prey, try zoom detection
        if (frame_result.has_cat and not frame_result.has_valid_prey
                and zoom_frame_provider is not None
                and frame_result.cat_detection is not None):
            try:
                zoom_frame = zoom_frame_provider(frame_result.cat_detection)
                if zoom_frame is not None:
                    zoom_result = self.engine.infer(zoom_frame)

                    # Require cat confirmation in zoom frame — if zooming into
                    # the supposed cat region doesn't show a cat, the original
                    # detection was likely a false positive (e.g. human legs)
                    zoom_cat = self._get_best_detection(zoom_result.detections, "cat")
                    if zoom_cat is None:
                        logger.debug(
                            "Zoom: no cat confirmed in zoomed crop, "
                            "skipping prey search (likely false positive)"
                        )
                    else:
                        # Find prey in zoomed crop with spatial validation
                        for prey_class in self.PREY_CLASSES:
                            prey = self._get_best_detection(zoom_result.detections, prey_class)
                            if prey is None:
                                continue

                            # Apply spatial validation within zoom frame
                            if self.spatial_validation_enabled:
                                zoom_match = self._check_spatial_match(zoom_cat, prey)
                                if zoom_match is None:
                                    logger.debug(
                                        f"Zoom: {prey.class_name} ({prey.confidence:.2f}) "
                                        f"not spatially near cat in zoom crop, skipping"
                                    )
                                    continue

                            frame_result.has_valid_prey = True
                            frame_result.prey_detection = prey
                            frame_result.match = SpatialMatch(
                                cat=frame_result.cat_detection,
                                prey=prey,
                                intersection_type="zoom",
                            )
                            logger.info(
                                f"Zoom detection: {prey.class_name} ({prey.confidence:.2f}) "
                                f"found in zoomed crop around cat "
                                f"(zoom cat: {zoom_cat.confidence:.2f})"
                            )
                            break
            except Exception as e:
                logger.error(f"Zoom detection error: {e}")

        # Update state based on confirmation mode
        if self.confirmation_mode == "score_accumulation":
            self._update_state_score_accumulation(frame_result, frame)
        else:
            self._update_state_frame_count(frame_result, frame)

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

    def _update_state_score_accumulation(
        self, frame_result: FrameResult, frame: np.ndarray
    ) -> None:
        """Update detection state using time-based score accumulation."""
        now = time.time()
        pending_callbacks: list[tuple[Callable, tuple]] = []

        with self._lock:
            # Clean up old prey scores outside window
            self._prey_scores = [
                entry for entry in self._prey_scores
                if now - entry.timestamp < self.prey_window_seconds
            ]

            # Handle cat presence
            if frame_result.has_cat:
                self._last_cat_time = now

                # Start monitoring if not already
                if self._state == DetectionState.IDLE:
                    self._state = DetectionState.MONITORING
                    self._monitoring_start_time = now
                    logger.info("State: IDLE -> MONITORING (cat detected)")
                    pending_callbacks.extend(
                        [(cb, (DetectionState.MONITORING,)) for cb in self._on_state_change_callbacks]
                    )

                # Emit cat_only callback if cat detected without valid prey (for training data)
                if not frame_result.has_valid_prey:
                    if (self._state == DetectionState.MONITORING and
                        self._monitoring_start_time and
                        now - self._monitoring_start_time >= 2.0 and  # Cat present for 2+ seconds
                        now - self._last_cat_only_callback_time >= self._cat_only_callback_cooldown):
                        self._last_cat_only_callback_time = now
                        cat_det = frame_result.cat_detection
                        all_dets = frame_result.all_detections
                        pending_callbacks.extend(
                            [(cb, (cat_det, all_dets, frame)) for cb in self._on_cat_only_callbacks]
                        )

                # Handle prey detection
                if frame_result.has_valid_prey and frame_result.prey_detection:
                    prey = frame_result.prey_detection

                    # Only count if above minimum score
                    if prey.confidence >= self.prey_min_detection_score:
                        # Add to score accumulation
                        self._prey_scores.append(PreyScoreEntry(
                            timestamp=now,
                            confidence=prey.confidence,
                            detection=prey,
                        ))

                        # Store match
                        if frame_result.match:
                            self._last_match = frame_result.match

                        # Calculate accumulated score
                        accumulated = sum(e.confidence for e in self._prey_scores)
                        detection_count = len(self._prey_scores)

                        logger.debug(
                            f"Prey score: +{prey.confidence:.2f}, "
                            f"accumulated={accumulated:.2f}/{self.prey_score_threshold}, "
                            f"count={detection_count}/{self.prey_min_detection_count}"
                        )

                        # Transition to VERIFYING if we have any score
                        if self._state == DetectionState.MONITORING:
                            self._state = DetectionState.VERIFYING
                            logger.info("State: MONITORING -> VERIFYING (prey detected)")
                            pending_callbacks.extend(
                                [(cb, (DetectionState.VERIFYING,)) for cb in self._on_state_change_callbacks]
                            )

                        # Check if both thresholds reached (score AND count)
                        if (accumulated >= self.prey_score_threshold
                                and detection_count >= self.prey_min_detection_count):
                            if self._state != DetectionState.CONFIRMED:
                                event = self._create_confirmation_event(
                                    frame, accumulated, detection_count
                                )
                                if event:
                                    pending_callbacks.extend(
                                        [(cb, (event,)) for cb in self._on_prey_confirmed_callbacks]
                                    )

            else:
                # No cat detected
                if self._last_cat_time is not None:
                    time_since_cat = now - self._last_cat_time

                    # Reset if cat lost for too long
                    if time_since_cat >= self.reset_on_cat_lost_seconds:
                        if self._state != DetectionState.IDLE:
                            old_state = self._state
                            accumulated_before_reset = sum(e.confidence for e in self._prey_scores)

                            self._state = DetectionState.IDLE
                            self._prey_scores.clear()
                            self._last_match = None
                            self._monitoring_start_time = None
                            logger.info(
                                f"State: {old_state.name} -> IDLE "
                                f"(cat lost for {time_since_cat:.1f}s)"
                            )
                            pending_callbacks.extend(
                                [(cb, (DetectionState.IDLE,)) for cb in self._on_state_change_callbacks]
                            )

                            # Emit near_miss callback if was VERIFYING with accumulated score
                            if old_state == DetectionState.VERIFYING and accumulated_before_reset > 0:
                                all_dets = frame_result.all_detections if frame_result else []
                                pending_callbacks.extend(
                                    [(cb, (accumulated_before_reset, all_dets, frame))
                                     for cb in self._on_near_miss_callbacks]
                                )

        # Invoke callbacks OUTSIDE the lock
        for callback, args in pending_callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback {getattr(callback, '__name__', callback)} error: {e}", exc_info=True)

    def _update_state_frame_count(
        self, frame_result: FrameResult, frame: np.ndarray
    ) -> None:
        """Legacy: Update detection state using frame-based counting."""
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
                if self._state != DetectionState.CONFIRMED:
                    event = self._create_legacy_confirmation_event(frame, positive_count)
                    if event:
                        pending_callbacks.extend(
                            [(cb, (event,)) for cb in self._on_prey_confirmed_callbacks]
                        )
            elif positive_count > 0:
                if self._state == DetectionState.IDLE:
                    self._state = DetectionState.VERIFYING
                    logger.info("State transition: IDLE -> VERIFYING")
                    pending_callbacks.extend(
                        [(cb, (DetectionState.VERIFYING,)) for cb in self._on_state_change_callbacks]
                    )
            else:
                if self._state == DetectionState.VERIFYING:
                    self._state = DetectionState.IDLE
                    self._last_match = None
                    logger.info("State transition: VERIFYING -> IDLE")
                    pending_callbacks.extend(
                        [(cb, (DetectionState.IDLE,)) for cb in self._on_state_change_callbacks]
                    )

        # Invoke callbacks OUTSIDE the lock
        for callback, args in pending_callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Callback {getattr(callback, '__name__', callback)} error: {e}", exc_info=True)

    def _create_confirmation_event(
        self, frame: np.ndarray, accumulated_score: float, detection_count: int
    ) -> PreyDetectionEvent | None:
        """Create prey confirmation event for score accumulation mode."""
        self._state = DetectionState.CONFIRMED
        logger.info(f"State: VERIFYING -> CONFIRMED")

        if self._last_match is None:
            logger.error("Prey confirmed but no match stored")
            return None

        event = PreyDetectionEvent(
            timestamp=datetime.now(),
            accumulated_score=accumulated_score,
            score_threshold=self.prey_score_threshold,
            detection_count=detection_count,
            window_seconds=self.prey_window_seconds,
            confidence=self._last_match.confidence,
            cat_detection=self._last_match.cat,
            prey_detection=self._last_match.prey,
            frame=frame.copy(),
        )

        logger.warning(
            f"PREY CONFIRMED! score={accumulated_score:.2f}/{self.prey_score_threshold}, "
            f"count={detection_count}/{self.prey_min_detection_count} in {self.prey_window_seconds}s window, "
            f"prey={self._last_match.prey.class_name} "
            f"({self._last_match.prey.confidence:.2f})"
        )

        return event

    def _create_legacy_confirmation_event(
        self, frame: np.ndarray, positive_count: int
    ) -> PreyDetectionEvent | None:
        """Create prey confirmation event for legacy frame_count mode."""
        self._state = DetectionState.CONFIRMED
        logger.info(f"State: -> CONFIRMED (legacy mode)")

        if self._last_match is None:
            logger.error("Prey confirmed but no match stored")
            return None

        event = PreyDetectionEvent(
            timestamp=datetime.now(),
            accumulated_score=float(positive_count),
            score_threshold=float(self.trigger_count),
            detection_count=positive_count,
            window_seconds=0.0,  # Not applicable in frame mode
            confidence=self._last_match.confidence,
            cat_detection=self._last_match.cat,
            prey_detection=self._last_match.prey,
            frame=frame.copy(),
        )

        logger.warning(
            f"PREY CONFIRMED! (legacy) {positive_count}/{self.window_size} frames, "
            f"prey={self._last_match.prey.class_name} "
            f"({self._last_match.prey.confidence:.2f})"
        )

        return event

    def reset(self) -> None:
        """Reset detector to IDLE state."""
        pending_callbacks: list[tuple[Callable, tuple]] = []

        with self._lock:
            self._detection_window.clear()
            self._prey_scores.clear()
            self._last_match = None
            self._last_cat_time = None
            self._monitoring_start_time = None

            if self._state != DetectionState.IDLE:
                old_state = self._state
                self._state = DetectionState.IDLE
                logger.info(f"State: {old_state.name} -> IDLE (reset)")
                pending_callbacks = [
                    (cb, (DetectionState.IDLE,)) for cb in self._on_state_change_callbacks
                ]

        # Invoke callbacks OUTSIDE the lock
        for callback, args in pending_callbacks:
            try:
                callback(*args)
            except Exception as e:
                logger.error(f"Reset callback {getattr(callback, '__name__', callback)} error: {e}", exc_info=True)

    def on_prey_confirmed(self, callback: Callable[[PreyDetectionEvent], None]) -> None:
        """Register callback for when prey is confirmed."""
        self._on_prey_confirmed_callbacks.append(callback)

    def on_state_change(self, callback: Callable[[DetectionState], None]) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    def on_cat_only(self, callback: Callable[[Detection, list[Detection], np.ndarray], None]) -> None:
        """
        Register callback for cat-only detections (cat without prey).

        Useful for collecting training data of cats without prey.

        Args:
            callback: Function receiving (cat_detection, all_detections, frame)
        """
        self._on_cat_only_callbacks.append(callback)

    def on_near_miss(self, callback: Callable[[float, list[Detection], np.ndarray], None]) -> None:
        """
        Register callback for near-miss detections (VERIFYING that reset to IDLE).

        Useful for collecting hard negative training data.

        Args:
            callback: Function receiving (accumulated_score, all_detections, frame)
        """
        self._on_near_miss_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get detector status."""
        return {
            "state": self._state.name,
            "confirmation_mode": self.confirmation_mode,
            # Score accumulation status
            "accumulated_score": self.accumulated_score,
            "score_threshold": self.prey_score_threshold,
            "detection_count_in_window": self.detection_count_in_window,
            "min_detection_count": self.prey_min_detection_count,
            "window_seconds": self.prey_window_seconds,
            # Legacy status
            "window_fill": self.window_fill,
            "positive_count": self.positive_count,
            "window_size": self.window_size,
            "trigger_count": self.trigger_count,
            # Config
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
            spatial_validation_enabled=inference_config.spatial_validation_enabled,
            box_expansion=inference_config.box_expansion,
            # Score accumulation params
            prey_confirmation_mode=inference_config.prey_confirmation_mode,
            prey_window_seconds=inference_config.prey_window_seconds,
            prey_score_threshold=inference_config.prey_score_threshold,
            prey_min_detection_score=inference_config.prey_min_detection_score,
            prey_min_detection_count=inference_config.prey_min_detection_count,
            reset_on_cat_lost_seconds=inference_config.reset_on_cat_lost_seconds,
            # Legacy params
            window_size=inference_config.window_size,
            trigger_count=inference_config.trigger_count,
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
    print("=== Prey Detector Test (Score Accumulation) ===")

    detector = PreyDetector(
        thresholds={"cat": 0.50, "rodent": 0.20},
        prey_confirmation_mode="score_accumulation",
        prey_window_seconds=3.0,
        prey_score_threshold=0.9,
        prey_min_detection_score=0.20,
        reset_on_cat_lost_seconds=1.5,
        spatial_validation_enabled=True,
        box_expansion=0.25,
    )

    def on_prey(event: PreyDetectionEvent):
        print(f"\n*** PREY ALERT! ***")
        print(f"  Prey: {event.prey_detection.class_name}")
        print(f"  Score: {event.accumulated_score:.2f}/{event.score_threshold}")
        print(f"  Detections: {event.detection_count} in {event.window_seconds}s")

    def on_state(state: DetectionState):
        print(f"State -> {state.name}")

    detector.on_prey_confirmed(on_prey)
    detector.on_state_change(on_state)

    print("\nProcessing test frames...")
    print(f"Config: window={detector.prey_window_seconds}s, "
          f"threshold={detector.prey_score_threshold}")

    for i in range(20):
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = detector.process_frame(frame)
        print(
            f"Frame {i+1}: {len(result.detections)} detections, "
            f"score={detector.accumulated_score:.2f}/{detector.prey_score_threshold}, "
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
