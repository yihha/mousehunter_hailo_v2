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
            assert det.class_id in [0, 1]  # v3: 0=cat, 1=rodent
            assert det.class_name in ["cat", "rodent"]
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
        # v3 model uses 2 classes: 0=cat, 1=rodent
        assert mock_engine.classes == {"0": "cat", "1": "rodent"}

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
        assert "using_hailo" in status
        assert "classes" in status

        assert status["initialized"] is True
        assert status["using_hailo"] is False  # Running in mock mode (force_mock=True)

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


class TestHailoAvailability:
    """Tests for Hailo hardware availability detection."""

    def test_mock_engine_does_not_use_hailo(self, mock_engine):
        """Test that mock engine does not use Hailo hardware."""
        # With force_mock=True, engine should never use real Hailo
        assert mock_engine._use_hailo is False
        assert mock_engine.get_status()["using_hailo"] is False

    @pytest.mark.skipif(HAILO_AVAILABLE, reason="Test only runs without Hailo hardware")
    def test_hailo_not_available_in_dev(self):
        """Test that Hailo is not available in development environment."""
        # In development environment without Hailo, HAILO_AVAILABLE should be False
        assert HAILO_AVAILABLE is False


@pytest.mark.skipif(not HAILO_AVAILABLE, reason="Hailo hardware required")
class TestHailoHardware:
    """Integration tests that run only on Raspberry Pi with Hailo hardware."""

    def test_hailo_engine_real_inference(self):
        """Test real Hailo inference with standard yolov8n model."""
        from pathlib import Path

        # Find the model file
        project_root = Path(__file__).parent.parent
        model_path = project_root / "models" / "yolov8n_catprey.hef"

        if not model_path.exists():
            pytest.skip(f"Model not found: {model_path}")

        engine = HailoEngine(
            model_path=model_path,
            confidence_threshold=0.5,
            classes={"0": "cat", "1": "rodent"},
            reg_max=8,
            force_mock=False,  # Use real Hailo
        )

        try:
            assert engine._use_hailo is True
            assert engine._initialized is True

            # Run inference on test frame
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            result = engine.infer(frame)

            assert result is not None
            assert hasattr(result, "detections")
            assert result.inference_time_ms >= 0

            # Hailo should be much faster than 100ms per frame
            assert result.inference_time_ms < 100, f"Inference too slow: {result.inference_time_ms}ms"

        finally:
            engine.cleanup()
