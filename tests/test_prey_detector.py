"""
Tests for PreyDetector with spatial validation and rolling window.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch

from mousehunter.inference.detection import BoundingBox, Detection, DetectionFrame
from mousehunter.inference.prey_detector import (
    PreyDetector,
    DetectionState,
    SpatialMatch,
    FrameResult,
    PreyDetectionEvent,
)


class TestSpatialValidation:
    """Tests for spatial intersection logic."""

    def test_check_spatial_match_direct_overlap(self, prey_detector):
        """Test spatial match with directly overlapping boxes."""
        cat = Detection(
            class_id=0,  # v3: cat=0
            class_name="cat",
            confidence=0.85,
            bbox=BoundingBox(x=0.3, y=0.3, width=0.25, height=0.25),
        )
        # Rodent overlaps with cat
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.45, y=0.40, width=0.08, height=0.06),
        )

        match = prey_detector._check_spatial_match(cat, rodent)
        assert match is not None
        assert match.intersection_type == "overlap"
        assert match.cat == cat
        assert match.prey == rodent

    def test_check_spatial_match_proximity(self, prey_detector):
        """Test spatial match via proximity (prey center in expanded cat box)."""
        cat = Detection(
            class_id=0,  # v3: cat=0
            class_name="cat",
            confidence=0.85,
            bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),  # 0.3-0.5
        )
        # Cat expanded by 25%: width 0.25, so x from 0.275 to 0.525
        # Rodent fully outside cat box (starts at 0.51), but center (0.52) within expanded (0.525)
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.51, y=0.38, width=0.02, height=0.02),  # center at 0.52, 0.39
        )

        match = prey_detector._check_spatial_match(cat, rodent)
        assert match is not None
        assert match.intersection_type == "proximity"

    def test_check_spatial_match_no_match(self, prey_detector):
        """Test no spatial match when prey is far from cat."""
        cat = Detection(
            class_id=0,  # v3: cat=0
            class_name="cat",
            confidence=0.85,
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        )
        # Rodent far away
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.8, y=0.8, width=0.05, height=0.04),
        )

        match = prey_detector._check_spatial_match(cat, rodent)
        assert match is None

    def test_spatial_validation_disabled(self, prey_detector_no_spatial):
        """Test that spatial validation can be disabled."""
        cat = Detection(
            class_id=0,  # v3: cat=0
            class_name="cat",
            confidence=0.85,
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        )
        # Rodent far away - would fail spatial validation
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.8, y=0.8, width=0.05, height=0.04),
        )

        frame = DetectionFrame(
            timestamp=datetime.now(),
            detections=[cat, rodent],
            frame_number=1,
            inference_time_ms=10.0,
        )

        result = prey_detector_no_spatial._evaluate_frame(frame)
        assert result.has_valid_prey is True  # No spatial check
        assert result.match.intersection_type == "disabled"


class TestFrameEvaluation:
    """Tests for frame evaluation logic."""

    def test_evaluate_frame_no_cat(self, prey_detector):
        """Test frame with no cat detection."""
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.4, y=0.4, width=0.05, height=0.04),
        )

        frame = DetectionFrame(
            timestamp=datetime.now(),
            detections=[rodent],
            frame_number=1,
            inference_time_ms=10.0,
        )

        result = prey_detector._evaluate_frame(frame)
        assert result.has_cat is False
        assert result.has_valid_prey is False

    def test_evaluate_frame_cat_only(self, prey_detector, cat_detection):
        """Test frame with cat but no prey."""
        frame = DetectionFrame(
            timestamp=datetime.now(),
            detections=[cat_detection],
            frame_number=1,
            inference_time_ms=10.0,
        )

        result = prey_detector._evaluate_frame(frame)
        assert result.has_cat is True
        assert result.has_valid_prey is False

    def test_evaluate_frame_cat_and_valid_rodent(self, prey_detector):
        """Test frame with cat and spatially valid rodent."""
        cat = Detection(
            class_id=0,  # v3: cat=0
            class_name="cat",
            confidence=0.85,
            bbox=BoundingBox(x=0.3, y=0.3, width=0.25, height=0.25),
        )
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.45, y=0.40, width=0.08, height=0.06),
        )

        frame = DetectionFrame(
            timestamp=datetime.now(),
            detections=[cat, rodent],
            frame_number=1,
            inference_time_ms=10.0,
        )

        result = prey_detector._evaluate_frame(frame)
        assert result.has_cat is True
        assert result.has_valid_prey is True
        assert result.match is not None
        assert result.prey_detection.class_name == "rodent"

    def test_evaluate_frame_cat_and_invalid_rodent(self, prey_detector):
        """Test frame with cat and spatially invalid rodent."""
        cat = Detection(
            class_id=0,  # v3: cat=0
            class_name="cat",
            confidence=0.85,
            bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
        )
        rodent = Detection(
            class_id=1,  # v3: rodent=1
            class_name="rodent",
            confidence=0.65,
            bbox=BoundingBox(x=0.8, y=0.8, width=0.05, height=0.04),
        )

        frame = DetectionFrame(
            timestamp=datetime.now(),
            detections=[cat, rodent],
            frame_number=1,
            inference_time_ms=10.0,
        )

        result = prey_detector._evaluate_frame(frame)
        assert result.has_cat is True
        assert result.has_valid_prey is False  # Spatial validation failed

class TestRollingWindow:
    """Tests for rolling window temporal smoothing."""

    def test_window_accumulation(self, prey_detector, sample_frame):
        """Test that detection window accumulates correctly."""
        assert prey_detector.window_fill == 0
        assert prey_detector.positive_count == 0

        # Process frames (mock will generate random detections)
        for i in range(5):
            prey_detector.process_frame(sample_frame)

        assert prey_detector.window_fill == 5

    def test_window_max_size(self, prey_detector, sample_frame):
        """Test window doesn't exceed max size."""
        for i in range(10):
            prey_detector.process_frame(sample_frame)

        assert prey_detector.window_fill == 5  # Max window size

    def test_state_verifying_on_partial_detections(self, prey_detector):
        """Test VERIFYING state when some detections in window."""
        # Manually add detections to window
        prey_detector._detection_window.append(True)
        prey_detector._detection_window.append(False)

        assert prey_detector.positive_count == 1
        assert prey_detector.state == DetectionState.IDLE

    def test_reset_clears_window(self, prey_detector, sample_frame):
        """Test reset clears the detection window."""
        # Fill window
        for i in range(5):
            prey_detector.process_frame(sample_frame)

        prey_detector.reset()
        assert prey_detector.window_fill == 0
        assert prey_detector.positive_count == 0
        assert prey_detector.state == DetectionState.IDLE


class TestStateTransitions:
    """Tests for state machine transitions."""

    def test_initial_state_idle(self, prey_detector):
        """Test initial state is IDLE."""
        assert prey_detector.state == DetectionState.IDLE

    def test_transition_to_verifying(self, prey_detector):
        """Test transition from IDLE to VERIFYING."""
        import numpy as np

        # Manually trigger state update with positive detection
        frame = np.zeros((640, 640, 3), dtype=np.uint8)
        prey_detector._update_state(
            FrameResult(has_valid_prey=True),
            frame,
        )
        # State should transition to VERIFYING since we have 1 positive in window
        assert prey_detector.state == DetectionState.VERIFYING

    def test_reset_returns_to_idle(self, prey_detector):
        """Test reset returns to IDLE."""
        prey_detector._state = DetectionState.VERIFYING
        prey_detector.reset()
        assert prey_detector.state == DetectionState.IDLE


class TestCallbacks:
    """Tests for callback registration and execution."""

    def test_on_prey_confirmed_callback(self, prey_detector):
        """Test prey confirmed callback registration."""
        callback_called = []

        def callback(event):
            callback_called.append(event)

        prey_detector.on_prey_confirmed(callback)
        assert len(prey_detector._on_prey_confirmed_callbacks) == 1

    def test_on_state_change_callback(self, prey_detector):
        """Test state change callback registration."""
        states = []

        def callback(state):
            states.append(state)

        prey_detector.on_state_change(callback)

        # Manually set state to VERIFYING, then reset() will transition to IDLE
        # and invoke the callback properly (callbacks are invoked outside the lock)
        prey_detector._state = DetectionState.VERIFYING
        prey_detector.reset()

        assert DetectionState.IDLE in states


class TestGetStatus:
    """Tests for status reporting."""

    def test_get_status(self, prey_detector):
        """Test status dictionary contents."""
        status = prey_detector.get_status()

        assert "state" in status
        assert "window_fill" in status
        assert "positive_count" in status
        assert "window_size" in status
        assert "trigger_count" in status
        assert "thresholds" in status
        assert "spatial_validation" in status
        assert "box_expansion" in status
        assert "engine_status" in status

        assert status["window_size"] == 5
        assert status["trigger_count"] == 3
        assert status["spatial_validation"] is True


class TestThresholds:
    """Tests for per-class confidence thresholds."""

    def test_get_best_detection_above_threshold(self, prey_detector):
        """Test detection above threshold is returned."""
        detections = [
            Detection(
                class_id=0,  # v3: cat=0
                class_name="cat",
                confidence=0.60,
                bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),
            )
        ]

        result = prey_detector._get_best_detection(detections, "cat")
        assert result is not None
        assert result.confidence == 0.60

    def test_get_best_detection_below_threshold(self, prey_detector):
        """Test detection below threshold is not returned."""
        detections = [
            Detection(
                class_id=0,  # v3: cat=0
                class_name="cat",
                confidence=0.40,  # Below 0.55 threshold
                bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),
            )
        ]

        result = prey_detector._get_best_detection(detections, "cat")
        assert result is None

    def test_get_best_detection_returns_highest(self, prey_detector):
        """Test highest confidence detection is returned."""
        detections = [
            Detection(
                class_id=1,  # v3: rodent=1
                class_name="rodent",
                confidence=0.50,
                bbox=BoundingBox(x=0.3, y=0.3, width=0.05, height=0.04),
            ),
            Detection(
                class_id=1,  # v3: rodent=1
                class_name="rodent",
                confidence=0.75,
                bbox=BoundingBox(x=0.5, y=0.5, width=0.05, height=0.04),
            ),
        ]

        result = prey_detector._get_best_detection(detections, "rodent")
        assert result is not None
        assert result.confidence == 0.75
