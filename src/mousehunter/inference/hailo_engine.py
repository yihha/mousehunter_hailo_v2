"""
Hailo-8L Inference Engine

Uses the picamera2 Hailo wrapper for running YOLOv8 inference on the
Raspberry Pi AI HAT+ (Hailo-8L NPU).

The picamera2 Hailo wrapper handles:
- DMA buffer management
- Threading (works across threads)
- Input/output formatting
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .detection import Detection, DetectionFrame, BoundingBox

logger = logging.getLogger(__name__)

# Conditional import for development without Hailo hardware
try:
    from picamera2.devices import Hailo
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("picamera2.devices.Hailo not available - running in simulation mode")


class MockHailoInference:
    """Mock inference for development without Hailo hardware."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"[MOCK] Hailo model loaded: {model_path}")

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Generate mock detections with realistic spatial relationships."""
        detections = []

        # 15% chance of detecting a cat
        if np.random.random() < 0.15:
            cat_x = np.random.uniform(0.25, 0.55)
            cat_y = np.random.uniform(0.25, 0.55)
            cat_w = np.random.uniform(0.15, 0.30)
            cat_h = np.random.uniform(0.15, 0.30)

            detections.append(
                Detection(
                    class_id=1,  # Custom model: cat
                    class_name="cat",
                    confidence=np.random.uniform(0.70, 0.98),
                    bbox=BoundingBox(x=cat_x, y=cat_y, width=cat_w, height=cat_h),
                )
            )

            # 8% chance of prey near cat (bird or rodent)
            if np.random.random() < 0.08:
                prey_x = cat_x + cat_w * np.random.uniform(0.3, 0.7)
                prey_y = cat_y + cat_h * np.random.uniform(0.1, 0.4)
                # 50/50 chance bird vs rodent
                if np.random.random() < 0.5:
                    prey_class_id = 0  # bird
                    prey_class_name = "bird"
                else:
                    prey_class_id = 3  # rodent
                    prey_class_name = "rodent"
                detections.append(
                    Detection(
                        class_id=prey_class_id,
                        class_name=prey_class_name,
                        confidence=np.random.uniform(0.50, 0.85),
                        bbox=BoundingBox(
                            x=prey_x, y=prey_y,
                            width=np.random.uniform(0.05, 0.10),
                            height=np.random.uniform(0.04, 0.08),
                        ),
                    )
                )

        return detections

    def cleanup(self):
        logger.info("[MOCK] Hailo resources released")


class HailoEngine:
    """
    Hailo-8L inference engine for YOLOv8 object detection.

    Uses the picamera2 Hailo wrapper which properly handles:
    - DMA buffer management
    - Cross-thread usage
    - Input/output formatting
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.45,
        classes: dict[str, str] | None = None,
        force_mock: bool = False,
    ):
        """
        Initialize the Hailo inference engine.

        Args:
            model_path: Path to compiled HEF model
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS
            classes: Class ID to name mapping (e.g., {"15": "cat", "14": "bird"})
            force_mock: Force mock mode even if Hailo hardware is available
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.classes = classes or {"15": "cat", "14": "bird"}
        self._force_mock = force_mock
        self._use_hailo = HAILO_AVAILABLE and not force_mock

        # State
        self._initialized = False
        self._hailo: Hailo | None = None
        self._input_shape: tuple | None = None
        self._frame_count = 0
        self._total_inference_time = 0.0

        # Callbacks for detection events
        self._detection_callbacks: list[Callable[[DetectionFrame], None]] = []

        # Initialize
        if self._use_hailo:
            self._init_hailo()
        else:
            self._mock_engine = MockHailoInference(str(model_path))
            self._initialized = True
            if force_mock:
                logger.info("Hailo engine running in forced mock mode")

        logger.info(
            f"HailoEngine initialized: model={model_path}, "
            f"threshold={confidence_threshold}, hailo={self._use_hailo}"
        )

    def _init_hailo(self) -> None:
        """Initialize Hailo using picamera2 wrapper."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            # Create Hailo instance using picamera2 wrapper
            # This handles all the complex DMA/buffer management internally
            self._hailo = Hailo(str(self.model_path))

            # Get input shape
            self._input_shape = self._hailo.get_input_shape()
            logger.info(f"Model input shape: {self._input_shape}")

            # Get model info
            inputs, outputs = self._hailo.describe()
            for name, shape, fmt in inputs:
                logger.info(f"Input: {name}, shape={shape}, format={fmt}")
            for name, shape, fmt in outputs:
                logger.info(f"Output: {name}, shape={shape}, format={fmt}")

            self._initialized = True
            logger.info("Hailo device initialized successfully (picamera2 wrapper)")

        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def infer(self, frame: np.ndarray) -> DetectionFrame:
        """
        Run inference on a single frame.

        Args:
            frame: Input image as numpy array (RGB, HWC format, 640x640x3)

        Returns:
            DetectionFrame with all detections
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        # Validate input frame
        if frame is None or frame.size == 0:
            logger.error("Received empty or None frame")
            return DetectionFrame(
                timestamp=datetime.now(),
                detections=[],
                frame_number=self._frame_count,
                inference_time_ms=0.0,
            )

        # Log raw frame info for first few frames
        if self._frame_count < 5:
            logger.info(
                f"Raw frame: shape={frame.shape}, dtype={frame.dtype}"
            )

        start_time = time.perf_counter()
        timestamp = datetime.now()

        if self._use_hailo:
            detections = self._run_hailo_inference(frame)
        else:
            detections = self._mock_engine.infer(frame)

        inference_time = (time.perf_counter() - start_time) * 1000  # ms

        self._frame_count += 1
        self._total_inference_time += inference_time

        result = DetectionFrame(
            timestamp=timestamp,
            detections=detections,
            frame_number=self._frame_count,
            inference_time_ms=inference_time,
        )

        # Notify callbacks
        if detections:
            for callback in self._detection_callbacks:
                try:
                    callback(result)
                except Exception as e:
                    logger.error(f"Detection callback error: {e}")

        return result

    def _run_hailo_inference(self, frame: np.ndarray) -> list[Detection]:
        """Run inference using picamera2 Hailo wrapper."""
        # Preprocess frame
        input_data = self._preprocess(frame)

        # Debug: Log input details for first few frames
        if self._frame_count < 5:
            logger.info(
                f"Hailo input: shape={input_data.shape}, dtype={input_data.dtype}"
            )

        try:
            # Run inference using picamera2 Hailo wrapper
            # This is much simpler - just pass the frame!
            raw_output = self._hailo.run(input_data)

            # Log output format for debugging
            if self._frame_count < 5:
                if isinstance(raw_output, dict):
                    for k, v in raw_output.items():
                        if isinstance(v, list):
                            logger.info(f"Output {k}: list of {len(v)} arrays")
                        else:
                            logger.info(f"Output {k}: shape={np.array(v).shape}")
                elif isinstance(raw_output, list):
                    logger.info(f"Output: list of {len(raw_output)} class arrays")
                else:
                    logger.info(f"Output: type={type(raw_output)}")

        except Exception as e:
            logger.error(f"Hailo inference error: {e}", exc_info=True)
            return []

        # Post-process outputs
        detections = self._postprocess_yolo(raw_output, frame.shape[:2])

        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO inference.

        Args:
            frame: Input RGB image (HWC), uint8 or float

        Returns:
            Preprocessed tensor: uint8 array matching model input shape
        """
        # Get target dimensions from model
        if self._input_shape:
            if len(self._input_shape) == 4:
                target_h, target_w = self._input_shape[1], self._input_shape[2]
            else:
                target_h, target_w = self._input_shape[0], self._input_shape[1]
        else:
            target_h, target_w = 640, 640

        # Handle 4D input
        if len(frame.shape) == 4 and frame.shape[0] == 1:
            frame = frame[0]

        # Ensure uint8
        if frame.dtype != np.uint8:
            if frame.dtype in (np.float32, np.float64) and frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Resize if needed
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            from PIL import Image
            img = Image.fromarray(frame)
            img = img.resize((target_w, target_h), Image.BILINEAR)
            frame = np.array(img, dtype=np.uint8)

        return frame

    def _postprocess_yolo(self, raw_output, original_shape: tuple[int, int]) -> list[Detection]:
        """
        Post-process YOLOv8 output from Hailo NMS.

        The output format from picamera2 Hailo wrapper is a ragged array:
        - List of 80 arrays (one per COCO class)
        - Each array has shape (num_detections_for_class, 5)
        - Each detection: [y_min, x_min, y_max, x_max, score] normalized 0-1
        """
        detections = []

        # Handle dict output (get first value)
        if isinstance(raw_output, dict):
            raw_output = list(raw_output.values())[0]

        # Handle ragged array: list of per-class detection arrays
        if isinstance(raw_output, (list, tuple)):
            for class_id, class_detections in enumerate(raw_output):
                # Convert to numpy if needed
                if not isinstance(class_detections, np.ndarray):
                    class_detections = np.array(class_detections)

                # Skip empty classes
                if class_detections.size == 0:
                    continue

                # Ensure 2D shape (num_detections, 5)
                if class_detections.ndim == 1:
                    class_detections = class_detections.reshape(-1, 5)

                for det in class_detections:
                    if len(det) < 5:
                        continue

                    y_min, x_min, y_max, x_max, score = det[:5]

                    # Skip low confidence
                    if score < self.confidence_threshold:
                        continue

                    # Skip invalid boxes
                    if x_max <= x_min or y_max <= y_min:
                        continue

                    # Get class name from mapping
                    class_name = self.classes.get(str(class_id), f"class_{class_id}")

                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=float(score),
                            bbox=BoundingBox(
                                x=float(x_min),
                                y=float(y_min),
                                width=float(x_max - x_min),
                                height=float(y_max - y_min),
                            ),
                        )
                    )
        else:
            logger.warning(f"Unexpected output type: {type(raw_output)}")

        # Sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        return detections

    def on_detection(self, callback: Callable[[DetectionFrame], None]) -> None:
        """Register callback for detection events."""
        self._detection_callbacks.append(callback)

    @property
    def average_inference_time(self) -> float:
        """Get average inference time in milliseconds."""
        if self._frame_count == 0:
            return 0.0
        return self._total_inference_time / self._frame_count

    @property
    def fps(self) -> float:
        """Get estimated FPS based on average inference time."""
        avg_time = self.average_inference_time
        if avg_time == 0:
            return 0.0
        return 1000.0 / avg_time

    def get_status(self) -> dict:
        """Get engine status."""
        return {
            "initialized": self._initialized,
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "frame_count": self._frame_count,
            "average_inference_ms": self.average_inference_time,
            "estimated_fps": self.fps,
            "hailo_available": HAILO_AVAILABLE,
            "using_hailo": self._use_hailo,
            "classes": self.classes,
        }

    def cleanup(self) -> None:
        """Release Hailo resources."""
        if self._hailo:
            try:
                self._hailo.close()
            except Exception as e:
                logger.error(f"Error closing Hailo: {e}")
            self._hailo = None
        elif hasattr(self, "_mock_engine"):
            self._mock_engine.cleanup()

        self._initialized = False
        logger.info("Hailo engine cleaned up")


# Factory function
def _create_default_engine() -> HailoEngine:
    """Create engine from config."""
    try:
        from mousehunter.config import inference_config, PROJECT_ROOT

        model_path = Path(inference_config.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        return HailoEngine(
            model_path=model_path,
            confidence_threshold=inference_config.confidence_threshold,
            classes=inference_config.classes,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return HailoEngine(model_path="models/yolov8n.hef")


# Global instance (lazy)
_engine_instance: HailoEngine | None = None


def get_hailo_engine() -> HailoEngine:
    """Get or create the global Hailo engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = _create_default_engine()
    return _engine_instance


def test_engine() -> None:
    """Test the inference engine."""
    logging.basicConfig(level=logging.INFO)
    print("=== Hailo Engine Test (picamera2 wrapper) ===")
    print(f"Hailo Available: {HAILO_AVAILABLE}")

    engine = HailoEngine(
        model_path="models/yolov8n.hef",
        confidence_threshold=0.5,
    )

    # Generate test frame
    test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    print("Running inference...")
    for i in range(10):
        result = engine.infer(test_frame)
        print(f"Frame {i+1}: {len(result.detections)} detections, {result.inference_time_ms:.2f}ms")
        for det in result.detections:
            print(f"  - {det}")

    print(f"\nStatus: {engine.get_status()}")
    engine.cleanup()


if __name__ == "__main__":
    test_engine()
