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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
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

# Inference timeout (seconds) - prevents system hang if Hailo freezes
INFERENCE_TIMEOUT_SECONDS = 5.0


class MockHailoInference:
    """Mock inference for development without Hailo hardware."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"[MOCK] Hailo model loaded: {model_path}")

    def infer(self, frame: np.ndarray) -> list[Detection]:
        """Generate mock detections with realistic spatial relationships.

        Custom model classes: 0=cat, 1=rodent, 2=leaf, 3=bird
        """
        detections = []

        # 15% chance of detecting a cat
        if np.random.random() < 0.15:
            # Cat position (center-ish)
            cat_x = np.random.uniform(0.25, 0.55)
            cat_y = np.random.uniform(0.25, 0.55)
            cat_w = np.random.uniform(0.15, 0.30)
            cat_h = np.random.uniform(0.15, 0.30)

            detections.append(
                Detection(
                    class_id=0,  # Custom model: cat = 0
                    class_name="cat",
                    confidence=np.random.uniform(0.70, 0.98),
                    bbox=BoundingBox(x=cat_x, y=cat_y, width=cat_w, height=cat_h),
                )
            )

            # 8% chance of also detecting rodent (spatially near cat)
            if np.random.random() < 0.08:
                # Place rodent near cat's "mouth" area (front of cat box)
                rodent_x = cat_x + cat_w * np.random.uniform(0.6, 0.9)
                rodent_y = cat_y + cat_h * np.random.uniform(0.2, 0.5)
                rodent_w = np.random.uniform(0.04, 0.08)
                rodent_h = np.random.uniform(0.03, 0.06)

                detections.append(
                    Detection(
                        class_id=1,  # Custom model: rodent = 1
                        class_name="rodent",
                        confidence=np.random.uniform(0.50, 0.85),
                        bbox=BoundingBox(
                            x=rodent_x, y=rodent_y, width=rodent_w, height=rodent_h
                        ),
                    )
                )

            # 2% chance of bird (rare, usually not overlapping)
            elif np.random.random() < 0.02:
                # Bird might be anywhere, sometimes overlapping
                if np.random.random() < 0.5:
                    # Overlapping with cat
                    bird_x = cat_x + cat_w * np.random.uniform(0.3, 0.7)
                    bird_y = cat_y + cat_h * np.random.uniform(0.1, 0.4)
                else:
                    # Random position (might not overlap)
                    bird_x = np.random.uniform(0.1, 0.7)
                    bird_y = np.random.uniform(0.1, 0.7)

                detections.append(
                    Detection(
                        class_id=3,  # Custom model: bird = 3
                        class_name="bird",
                        confidence=np.random.uniform(0.40, 0.75),
                        bbox=BoundingBox(
                            x=bird_x,
                            y=bird_y,
                            width=np.random.uniform(0.05, 0.10),
                            height=np.random.uniform(0.04, 0.08),
                        ),
                    )
                )

            # 3% chance of leaf (false positive that should be ignored)
            elif np.random.random() < 0.03:
                detections.append(
                    Detection(
                        class_id=2,  # Custom model: leaf = 2
                        class_name="leaf",
                        confidence=np.random.uniform(0.40, 0.70),
                        bbox=BoundingBox(
                            x=np.random.uniform(0.1, 0.8),
                            y=np.random.uniform(0.1, 0.8),
                            width=np.random.uniform(0.03, 0.08),
                            height=np.random.uniform(0.02, 0.06),
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
        force_mock: bool = False,
    ):
        """
        Initialize the Hailo inference engine.

        Args:
            model_path: Path to compiled HEF model
            confidence_threshold: Minimum confidence for detections
            nms_iou_threshold: IoU threshold for NMS
            classes: Class ID to name mapping (e.g., {"0": "cat", "1": "prey"})
            force_mock: Force mock mode even if Hailo hardware is available (for testing)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        # Default to COCO classes for standard yolov8n.hef
        # For custom model, pass classes={"0": "cat", "1": "rodent", "2": "leaf", "3": "bird"}
        self.classes = classes or {"15": "cat", "14": "bird"}
        self._force_mock = force_mock
        self._use_hailo = HAILO_AVAILABLE and not force_mock

        # State
        self._initialized = False
        self._frame_count = 0
        self._total_inference_time = 0.0
        self._timeout_count = 0  # Track inference timeouts

        # Thread pool for timeout-protected inference
        self._inference_executor: ThreadPoolExecutor | None = None

        # Callbacks for detection events
        self._detection_callbacks: list[Callable[[DetectionFrame], None]] = []

        # Initialize engine
        if self._use_hailo:
            self._init_hailo()
            # Create executor for timeout protection (single thread to serialize Hailo access)
            self._inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hailo")
        else:
            self._mock_engine = MockHailoInference(str(model_path))
            self._initialized = True
            if force_mock:
                logger.info("Hailo engine running in forced mock mode (for testing)")

        logger.info(
            f"HailoEngine initialized: model={model_path}, "
            f"threshold={confidence_threshold}, hailo={self._use_hailo}"
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
            self._input_name = self._input_vstream_infos[0].name
            logger.info(f"Model input shape: {self._input_shape}")

            # Create virtual device
            self._vdevice = VDevice()

            # Configure network
            configure_params = ConfigureParams.create_from_hef(
                self._hef, interface=HailoStreamInterface.PCIe
            )
            self._network_group = self._vdevice.configure(self._hef, configure_params)[0]

            # Create network group params for activation
            self._network_group_params = self._network_group.create_params()

            # Create input/output vstream params
            self._input_params = InputVStreamParams.make(
                self._network_group, quantized=False, format_type=FormatType.UINT8
            )
            self._output_params = OutputVStreamParams.make(
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
        """Run inference on Hailo hardware with timeout protection."""
        # Preprocess frame
        input_data = self._preprocess(frame)

        # Run inference with timeout to prevent system hang if Hailo freezes
        try:
            if self._inference_executor:
                future = self._inference_executor.submit(
                    self._execute_hailo_pipeline, input_data
                )
                output_dict = future.result(timeout=INFERENCE_TIMEOUT_SECONDS)
            else:
                # Fallback without timeout (shouldn't happen in normal operation)
                output_dict = self._execute_hailo_pipeline(input_data)
        except FuturesTimeoutError:
            self._timeout_count += 1
            logger.error(
                f"Hailo inference timeout after {INFERENCE_TIMEOUT_SECONDS}s! "
                f"(total timeouts: {self._timeout_count})"
            )
            return []  # Return empty detections on timeout
        except Exception as e:
            logger.error(f"Hailo inference error: {e}", exc_info=True)
            return []

        # Post-process outputs
        raw_output = list(output_dict.values())[0]
        detections = self._postprocess_yolo(raw_output, frame.shape[:2])

        return detections

    def _execute_hailo_pipeline(self, input_data: np.ndarray) -> dict:
        """Execute the actual Hailo inference pipeline (called from thread pool)."""
        with self._network_group.activate(self._network_group_params):
            with InferVStreams(
                self._network_group, self._input_params, self._output_params
            ) as pipeline:
                input_dict = {self._input_name: input_data}
                return pipeline.infer(input_dict)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for YOLO inference.

        Args:
            frame: Input RGB image (HWC), uint8

        Returns:
            Preprocessed tensor matching model input shape, uint8
            (Hailo quantized models handle normalization internally)
        """
        # Input shape is (H, W, C) = (640, 640, 3)
        target_h, target_w = self._input_shape[0], self._input_shape[1]

        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Resize if needed
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            from PIL import Image

            img = Image.fromarray(frame)
            img = img.resize((target_w, target_h), Image.BILINEAR)
            frame = np.array(img, dtype=np.uint8)

        # Hailo expects uint8 input - quantization is handled by the HEF model
        # Shape: (H, W, C) without batch dimension for InferVStreams
        return frame

    def _postprocess_yolo(
        self, raw_output: np.ndarray, original_shape: tuple[int, int]
    ) -> list[Detection]:
        """
        Post-process YOLOv8 output from Hailo NMS.

        Hailo's yolov8_nms_postprocess output format: (num_classes, 5, max_detections)
        - num_classes: 80 for COCO
        - 5: (y_min, x_min, y_max, x_max, score) normalized 0-1
        - max_detections: 100 per class

        Args:
            raw_output: Raw network output from Hailo NMS
            original_shape: Original image (H, W) - not used, coords are normalized

        Returns:
            List of Detection objects
        """
        detections = []

        logger.debug(f"Raw output shape: {raw_output.shape}")

        # Handle Hailo NMS output format: (num_classes, 5, max_detections)
        if raw_output.ndim == 3 and raw_output.shape[1] == 5:
            num_classes = raw_output.shape[0]
            max_detections = raw_output.shape[2]

            for class_id in range(num_classes):
                class_output = raw_output[class_id]  # Shape: (5, max_detections)

                for det_idx in range(max_detections):
                    y_min = float(class_output[0, det_idx])
                    x_min = float(class_output[1, det_idx])
                    y_max = float(class_output[2, det_idx])
                    x_max = float(class_output[3, det_idx])
                    score = float(class_output[4, det_idx])

                    # Skip empty detections (score = 0 or invalid bbox)
                    if score < self.confidence_threshold:
                        continue

                    # Skip invalid boxes
                    if x_max <= x_min or y_max <= y_min:
                        continue

                    # Get class name from mapping
                    class_name = self.classes.get(str(class_id), f"class_{class_id}")

                    # Convert to our format (x, y, width, height)
                    width = x_max - x_min
                    height = y_max - y_min

                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=score,
                            bbox=BoundingBox(x=x_min, y=y_min, width=width, height=height),
                        )
                    )
        else:
            # Fallback for other output formats
            logger.warning(f"Unexpected output shape: {raw_output.shape}, trying generic parse")

        # NMS already applied by Hailo, but sort by confidence
        detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

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
            "using_hailo": self._use_hailo,
            "timeout_count": self._timeout_count,
            "classes": self.classes,
        }

    def cleanup(self) -> None:
        """Release Hailo resources."""
        # Shutdown thread pool executor first
        if self._inference_executor:
            self._inference_executor.shutdown(wait=True, cancel_futures=True)
            self._inference_executor = None

        if self._use_hailo and self._initialized:
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
