"""
Pytest configuration and shared fixtures for MouseHunter tests.
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mousehunter.inference.detection import BoundingBox, Detection, DetectionFrame
from mousehunter.inference.hailo_engine import HailoEngine
from mousehunter.inference.prey_detector import PreyDetector


@pytest.fixture
def sample_bbox():
    """A sample bounding box for testing."""
    return BoundingBox(x=0.3, y=0.3, width=0.25, height=0.25)


@pytest.fixture
def cat_detection(sample_bbox):
    """A sample cat detection (COCO class 15)."""
    return Detection(
        class_id=15,
        class_name="cat",
        confidence=0.85,
        bbox=sample_bbox,
    )


@pytest.fixture
def rodent_detection():
    """A sample rodent detection that overlaps with cat."""
    # Note: COCO has no rodent class, using custom class 1 for trained model
    return Detection(
        class_id=1,
        class_name="rodent",
        confidence=0.65,
        bbox=BoundingBox(x=0.45, y=0.35, width=0.06, height=0.05),
    )


@pytest.fixture
def rodent_far_detection():
    """A rodent detection far from cat (no overlap)."""
    return Detection(
        class_id=1,
        class_name="rodent",
        confidence=0.70,
        bbox=BoundingBox(x=0.8, y=0.8, width=0.05, height=0.04),
    )


@pytest.fixture
def bird_detection():
    """A sample bird detection (COCO class 14)."""
    return Detection(
        class_id=14,
        class_name="bird",
        confidence=0.85,
        bbox=BoundingBox(x=0.48, y=0.38, width=0.07, height=0.06),
    )


@pytest.fixture
def mock_engine():
    """Create a HailoEngine in mock mode."""
    return HailoEngine(
        model_path="models/yolov8n_catprey.hef",
        confidence_threshold=0.5,
    )


@pytest.fixture
def prey_detector(mock_engine):
    """Create a PreyDetector with mock engine."""
    return PreyDetector(
        engine=mock_engine,
        thresholds={"cat": 0.55, "rodent": 0.45, "bird": 0.80},
        window_size=5,
        trigger_count=3,
        spatial_validation_enabled=True,
        box_expansion=0.25,
    )


@pytest.fixture
def prey_detector_no_spatial(mock_engine):
    """Create a PreyDetector with spatial validation disabled."""
    return PreyDetector(
        engine=mock_engine,
        thresholds={"cat": 0.55, "rodent": 0.45, "bird": 0.80},
        window_size=5,
        trigger_count=3,
        spatial_validation_enabled=False,
        box_expansion=0.25,
    )


@pytest.fixture
def sample_frame():
    """A sample 640x640 RGB frame."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)


@pytest.fixture
def detection_frame_with_cat_and_rodent(cat_detection, rodent_detection):
    """A DetectionFrame containing cat and overlapping rodent."""
    from datetime import datetime

    return DetectionFrame(
        timestamp=datetime.now(),
        detections=[cat_detection, rodent_detection],
        frame_number=1,
        inference_time_ms=10.0,
    )


@pytest.fixture
def detection_frame_cat_only(cat_detection):
    """A DetectionFrame containing only a cat."""
    from datetime import datetime

    return DetectionFrame(
        timestamp=datetime.now(),
        detections=[cat_detection],
        frame_number=1,
        inference_time_ms=10.0,
    )


@pytest.fixture
def detection_frame_rodent_far(cat_detection, rodent_far_detection):
    """A DetectionFrame with cat and far-away rodent (no spatial match)."""
    from datetime import datetime

    return DetectionFrame(
        timestamp=datetime.now(),
        detections=[cat_detection, rodent_far_detection],
        frame_number=1,
        inference_time_ms=10.0,
    )
