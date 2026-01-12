"""
Hailo-8L Inference Engine

Wraps the hailo_platform API for running YOLOv8 inference on the
Raspberry Pi AI HAT+ (Hailo-8L NPU).

The engine provides:
- HEF model loading and compilation
- Async inference pipeline
- YOLO post-processing (NMS, score filtering)
- Detection result parsing
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
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )

    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    logger.warning("hailo_platform not available - running in simulation mode")


class MockHailoInference:
    """Mock inference for development without Hailo hardware."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"[MOCK] Hailo model loaded: {model_path}")

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Generate mock detections."""
        # Simulate random detections for testing
        detections = []

        # 10% chance of detecting a cat
        if np.random.random() < 0.1:
            detections.append(
                Detection(
                    class_id=0,
                    class_name="cat",
                    confidence=np.random.uniform(0.7, 0.95),
                    bbox=BoundingBox(
                        x=np.random.uniform(0.2, 0.6),
                        y=np.random.uniform(0.2, 0.6),
                        width=np.random.uniform(0.1, 0.3),
                        height=np.random.uniform(0.1, 0.3),
                    ),
                )
            )

            # 5% chance of also detecting prey
            if np.random.random() < 0.05:
                detections.append(
                    Detection(
                        class_id=1,
                        class_name="prey",
                        confidence=np.random.uniform(0.6, 0.85),
                        bbox=BoundingBox(
                            x=np.random.uniform(0.3, 0.5),
                            y=np.random.uniform(0.3, 0.5),
                            width=np.random.uniform(0.05, 0.1),
                            height=np.random.uniform(0.05, 0.1),
                        ),
                    )
                )

        return detections

    def cleanup(self):
        logger.info("[MOCK] Hailo resources released")


class HailoEngine:
    """
    Hailo-8L inference engine for YOLOv8 object detection.

    Loads a compiled HEF model and runs inference on the NPU.
    Provides YOLO-specific post-processing including NMS.
    """

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.6,
        nms_iou_threshold: float = 0.45,
        classes: dict[str, str] | None = None,
    ):
        """
        Initialize the Hailo inference engine.

        Args:
            model_path: Path to compiled HEF model
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS
            classes: Class ID to name mapping (e.g., {"0": "cat", "1": "prey"})
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.classes = classes or {"0": "cat", "1": "prey"}

        # State
        self._initialized = False
        self._frame_count = 0
        self._total_inference_time = 0.0

        # Callbacks for detection events
        self._detection_callbacks: list[Callable[[DetectionFrame], None]] = []

        # Initialize engine
        if HAILO_AVAILABLE:
            self._init_hailo()
        else:
            self._mock_engine = MockHailoInference(str(model_path))
            self._initialized = True

        logger.info(
            f"HailoEngine initialized: model={model_path}, "
            f"threshold={confidence_threshold}"
        )

    def _init_hailo(self) -> None:
        """Initialize Hailo device and load model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        try:
            # Load HEF file
            self._hef = HEF(str(self.model_path))

            # Get model info
            self._input_vstream_infos = self._hef.get_input_vstream_infos()
            self._output_vstream_infos = self._hef.get_output_vstream_infos()

            # Log model details
            for info in self._input_vstream_infos:
                logger.info(f"Input: {info.name}, shape={info.shape}")
            for info in self._output_vstream_infos:
                logger.info(f"Output: {info.name}, shape={info.shape}")

            # Get expected input shape
            self._input_shape = self._input_vstream_infos[0].shape
            logger.info(f"Model input shape: {self._input_shape}")

            # Create virtual device
            self._vdevice = VDevice()

            # Configure network
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(self._hef, configure_params)[0]

            # Create input/output params
            self._input_params = InputVStreamParams.make_from_network_group(
                self._network_group, quantized=False, format_type=FormatType.FLOAT32
            )
            self._output_params = OutputVStreamParams.make_from_network_group(
                self._network_group, quantized=False, format_type=FormatType.FLOAT32
            )

            self._initialized = True
            logger.info("Hailo device initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize Hailo: {e}")
            raise

    def infer(self, frame: np.ndarray) -> DetectionFrame:
        """
        Run inference on a single frame.

        Args:
            frame: Input image as numpy array (RGB, HWC format)

        Returns:
            DetectionFrame with all detections
        """
        if not self._initialized:
            raise RuntimeError("Engine not initialized")

        start_time = time.perf_counter()
        timestamp = datetime.now()

        if HAILO_AVAILABLE:
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
        """Run inference on Hailo hardware."""
        # Preprocess frame
        input_data = self._preprocess(frame)

        # Run inference
        with InferVStreams(
            self._network_group, self._input_params, self._output_params
        ) as pipeline:
            input_dict = {self._input_vstream_infos[0].name: input_data}
            output_dict = pipeline.infer(input_dict)

        # Post-process outputs
        raw_output = list(output_dict.values())[0]
        detections = self._postprocess_yolo(raw_output, frame.shape[:2])

        return detections

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO inference.

        Args:
            frame: Input RGB image (HWC)

        Returns:
            Preprocessed tensor (NHWC or NCHW depending on model)
        """
        target_h, target_w = self._input_shape[1:3]

        # Resize if needed
        if frame.shape[:2] != (target_h, target_w):
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize((target_w, target_h), Image.BILINEAR)
            frame = np.array(img)

        # Normalize to 0-1
        frame = frame.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(frame, axis=0)

    def _postprocess_yolo(
        self, raw_output: np.ndarray, original_shape: tuple[int, int]
    ) -> list[Detection]:
        """
        Post-process YOLOv8 output.

        Args:
            raw_output: Raw network output
            original_shape: Original image (H, W)

        Returns:
            List of Detection objects
        """
        # YOLOv8 output format: [batch, num_detections, 4 + num_classes]
        # Box format: [x_center, y_center, width, height] (normalized)

        detections = []

        # Remove batch dimension if present
        if raw_output.ndim == 3:
            raw_output = raw_output[0]

        num_classes = len(self.classes)

        for detection in raw_output:
            # Extract box and scores
            box = detection[:4]
            scores = detection[4 : 4 + num_classes]

            # Get best class
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            # Filter by confidence
            if confidence < self.confidence_threshold:
                continue

            # Convert box format (center to corner)
            x_center, y_center, width, height = box
            x = x_center - width / 2
            y = y_center - height / 2

            # Get class name
            class_name = self.classes.get(str(class_id), f"class_{class_id}")

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(x=x, y=y, width=width, height=height),
                )
            )

        # Apply NMS
        detections = self._nms(detections)

        return detections

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

        kept = []
        while detections:
            best = detections.pop(0)
            kept.append(best)

            # Remove overlapping detections of same class
            detections = [
                d
                for d in detections
                if d.class_id != best.class_id
                or d.bbox.iou(best.bbox) < self.nms_iou_threshold
            ]

        return kept

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
            "classes": self.classes,
        }

    def cleanup(self) -> None:
        """Release Hailo resources."""
        if HAILO_AVAILABLE and self._initialized:
            try:
                del self._network_group
                del self._vdevice
                del self._hef
            except Exception as e:
                logger.error(f"Error cleaning up Hailo: {e}")
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
        return HailoEngine(model_path="models/yolov8n_catprey.hef")


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
    print("=== Hailo Engine Test ===")
    print(f"Hailo Available: {HAILO_AVAILABLE}")

    engine = HailoEngine(
        model_path="models/yolov8n_catprey.hef",
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
