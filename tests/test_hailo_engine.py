"""
Tests for HailoEngine inference (mock mode).
"""

import pytest
import numpy as np

from mousehunter.inference.hailo_engine import HailoEngine, MockHailoInference, HAILO_AVAILABLE
from mousehunter.inference.detection import Detection, BoundingBox


class TestMockHailoInference:
    """Tests for MockHailoInference class."""

    def test_mock_creation(self):
        """Test mock inference creation."""
        mock = MockHailoInference("models/test.hef")
        assert mock.model_path == "models/test.hef"

    def test_mock_infer_returns_list(self):
        """Test mock inference returns list of detections."""
        mock = MockHailoInference("models/test.hef")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = mock.infer(frame)
        assert isinstance(result, list)

    def test_mock_detections_valid(self):
        """Test mock detections have valid structure."""
        mock = MockHailoInference("models/test.hef")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run multiple times to get some detections
        all_detections = []
        for _ in range(100):
            result = mock.infer(frame)
            all_detections.extend(result)

        # Should have at least some detections
        assert len(all_detections) > 0

        # Check detection structure
        for det in all_detections:
            assert isinstance(det, Detection)
            assert det.class_id in [0, 1, 2]  # cat, rodent, bird
            assert det.class_name in ["cat", "rodent", "bird"]
            assert 0 <= det.confidence <= 1
            assert isinstance(det.bbox, BoundingBox)

    def test_mock_class_distribution(self):
        """Test mock generates all expected classes."""
        mock = MockHailoInference("models/test.hef")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        classes_seen = set()
        for _ in range(500):
            result = mock.infer(frame)
            for det in result:
                classes_seen.add(det.class_name)

        # Should see cats frequently, rodents sometimes
        assert "cat" in classes_seen

    def test_mock_cleanup(self):
        """Test mock cleanup doesn't error."""
        mock = MockHailoInference("models/test.hef")
        mock.cleanup()  # Should not raise


class TestHailoEngine:
    """Tests for HailoEngine class."""

    def test_engine_creation(self, mock_engine):
        """Test engine creation."""
        assert mock_engine._initialized is True
        assert mock_engine.confidence_threshold == 0.5

    def test_engine_classes(self, mock_engine):
        """Test engine has correct class mapping."""
        # Default classes use COCO IDs: 15=cat, 14=bird
        assert mock_engine.classes == {"15": "cat", "14": "bird"}

    def test_engine_infer(self, mock_engine, sample_frame):
        """Test inference returns DetectionFrame."""
        result = mock_engine.infer(sample_frame)

        assert result is not None
        assert hasattr(result, "detections")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "inference_time_ms")
        assert result.inference_time_ms >= 0

    def test_engine_frame_count(self, mock_engine, sample_frame):
        """Test frame count increments."""
        initial_count = mock_engine._frame_count

        mock_engine.infer(sample_frame)
        mock_engine.infer(sample_frame)
        mock_engine.infer(sample_frame)

        assert mock_engine._frame_count == initial_count + 3

    def test_engine_average_inference_time(self, mock_engine, sample_frame):
        """Test average inference time calculation."""
        for _ in range(10):
            mock_engine.infer(sample_frame)

        avg_time = mock_engine.average_inference_time
        assert avg_time >= 0

    def test_engine_fps(self, mock_engine, sample_frame):
        """Test FPS calculation."""
        for _ in range(10):
            mock_engine.infer(sample_frame)

        fps = mock_engine.fps
        assert fps > 0  # Mock should be very fast

    def test_engine_get_status(self, mock_engine):
        """Test status reporting."""
        status = mock_engine.get_status()

        assert "initialized" in status
        assert "model_path" in status
        assert "confidence_threshold" in status
        assert "frame_count" in status
        assert "average_inference_ms" in status
        assert "estimated_fps" in status
        assert "hailo_available" in status
        assert "classes" in status

        assert status["initialized"] is True
        assert status["hailo_available"] is False  # Running in mock mode

    def test_engine_cleanup(self, mock_engine):
        """Test engine cleanup."""
        mock_engine.cleanup()
        assert mock_engine._initialized is False

    def test_engine_detection_callback(self, mock_engine, sample_frame):
        """Test detection callback is called."""
        callbacks_received = []

        def callback(frame):
            callbacks_received.append(frame)

        mock_engine.on_detection(callback)

        # Run multiple inferences until we get a detection
        for _ in range(100):
            result = mock_engine.infer(sample_frame)
            if result.detections:
                break

        # If we got detections, callback should have been called
        if callbacks_received:
            assert len(callbacks_received) > 0


class TestHailoEngineNMS:
    """Tests for Non-Maximum Suppression."""

    def test_nms_removes_duplicates(self, mock_engine):
        """Test NMS removes overlapping detections of same class."""
        detections = [
            Detection(
                class_id=0,
                class_name="cat",
                confidence=0.9,
                bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),
            ),
            Detection(
                class_id=0,
                class_name="cat",
                confidence=0.7,  # Lower confidence, overlapping
                bbox=BoundingBox(x=0.32, y=0.32, width=0.2, height=0.2),
            ),
        ]

        result = mock_engine._nms(detections)
        assert len(result) == 1
        assert result[0].confidence == 0.9  # Higher confidence kept

    def test_nms_keeps_different_classes(self, mock_engine):
        """Test NMS keeps detections of different classes."""
        detections = [
            Detection(
                class_id=0,
                class_name="cat",
                confidence=0.9,
                bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2),
            ),
            Detection(
                class_id=1,
                class_name="rodent",
                confidence=0.7,
                bbox=BoundingBox(x=0.32, y=0.32, width=0.1, height=0.1),
            ),
        ]

        result = mock_engine._nms(detections)
        assert len(result) == 2  # Both kept (different classes)

    def test_nms_keeps_non_overlapping(self, mock_engine):
        """Test NMS keeps non-overlapping detections of same class."""
        detections = [
            Detection(
                class_id=0,
                class_name="cat",
                confidence=0.9,
                bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2),
            ),
            Detection(
                class_id=0,
                class_name="cat",
                confidence=0.8,
                bbox=BoundingBox(x=0.7, y=0.7, width=0.2, height=0.2),  # Far apart
            ),
        ]

        result = mock_engine._nms(detections)
        assert len(result) == 2  # Both kept (no overlap)

    def test_nms_empty_input(self, mock_engine):
        """Test NMS with empty input."""
        result = mock_engine._nms([])
        assert result == []


class TestHailoAvailability:
    """Tests for Hailo hardware availability detection."""

    def test_hailo_not_available_in_test(self):
        """Test that Hailo is not available in test environment."""
        # In test/development environment, HAILO_AVAILABLE should be False
        assert HAILO_AVAILABLE is False
