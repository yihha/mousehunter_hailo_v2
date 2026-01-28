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
            classes: Class ID to name mapping (e.g., {"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"})
            force_mock: Force mock mode even if Hailo hardware is available
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.classes = classes or {"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"}
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
                    logger.info(f"Output: dict with {len(raw_output)} keys")
                    for k, v in raw_output.items():
                        if isinstance(v, np.ndarray):
                            logger.info(f"  {k}: shape={v.shape}, dtype={v.dtype}")
                        elif isinstance(v, list):
                            logger.info(f"  {k}: list of {len(v)} items")
                        else:
                            logger.info(f"  {k}: type={type(v)}")
                elif isinstance(raw_output, np.ndarray):
                    logger.info(f"Output: ndarray shape={raw_output.shape}, dtype={raw_output.dtype}")
                elif isinstance(raw_output, list):
                    logger.info(f"Output: list of {len(raw_output)} items")
                    for i, item in enumerate(raw_output[:3]):
                        if isinstance(item, np.ndarray):
                            logger.info(f"  [{i}]: shape={item.shape}")
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
        Post-process YOLOv8 output - handles both NMS'd and raw tensor formats.

        Format 1 (NMS'd - standard COCO model):
        - List of N arrays (one per class)
        - Each array: (num_detections, 5) with [y_min, x_min, y_max, x_max, score]

        Format 2 (Raw tensors - custom model without NMS):
        - Dict or numpy array with 6 output tensors
        - Needs decoding: DFL box regression + class scores + NMS
        """
        detections = []

        # Handle dict output
        if isinstance(raw_output, dict):
            # Check if it's raw tensors (6 outputs) or single NMS'd output
            if len(raw_output) > 1:
                # Multiple outputs = raw tensors, need full decoding
                return self._postprocess_yolo_raw(raw_output, original_shape)
            else:
                raw_output = list(raw_output.values())[0]

        # Handle numpy array directly (raw output from custom model)
        if isinstance(raw_output, np.ndarray):
            # Raw tensor output - need to decode
            return self._postprocess_yolo_raw({"output": raw_output}, original_shape)

        # Handle ragged array: list of per-class detection arrays (NMS'd format)
        if isinstance(raw_output, (list, tuple)):
            # Check if first element looks like per-class detections or raw tensors
            if len(raw_output) > 0:
                first = raw_output[0]
                if isinstance(first, np.ndarray):
                    # Check shape to determine format
                    if first.ndim >= 2 and first.shape[-1] == 5:
                        # NMS'd format: (num_dets, 5)
                        return self._postprocess_nms_output(raw_output)
                    elif first.ndim >= 2 and first.shape[0] > 10:
                        # Likely raw tensor format
                        return self._postprocess_yolo_raw(
                            {f"out_{i}": t for i, t in enumerate(raw_output)},
                            original_shape
                        )

            # Default: try NMS'd format
            return self._postprocess_nms_output(raw_output)

        logger.warning(f"Unexpected output type: {type(raw_output)}")
        return detections

    def _postprocess_nms_output(self, raw_output) -> list[Detection]:
        """Process NMS'd output format (list of per-class detections)."""
        detections = []

        for class_id, class_detections in enumerate(raw_output):
            if not isinstance(class_detections, np.ndarray):
                class_detections = np.array(class_detections)

            if class_detections.size == 0:
                continue

            if class_detections.ndim == 1:
                class_detections = class_detections.reshape(-1, 5)

            for det in class_detections:
                if len(det) < 5:
                    continue

                y_min, x_min, y_max, x_max, score = det[:5]

                if score < self.confidence_threshold:
                    continue

                if x_max <= x_min or y_max <= y_min:
                    continue

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

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _postprocess_yolo_raw(self, outputs: dict, original_shape: tuple[int, int]) -> list[Detection]:
        """
        Post-process raw YOLOv8 output tensors (no NMS).

        YOLOv8 outputs 6 tensors (3 scales x 2 outputs each):
        - Box regression with DFL: 64 channels (4 coords * 16 DFL bins)
        - Class scores: num_classes channels

        Tensor formats:
        - ONNX/PyTorch: NCHW [batch, channels, H, W] e.g., [1, 64, 80, 80]
        - Hailo output: Usually HWC [H, W, channels] e.g., [80, 80, 64]

        Scales: 80x80, 40x40, 20x20 for 640x640 input

        DFL (Distribution Focal Loss) outputs distance from cell center to each edge:
        - decoded[0] = left distance (cells)
        - decoded[1] = top distance (cells)
        - decoded[2] = right distance (cells)
        - decoded[3] = bottom distance (cells)
        """
        detections = []
        num_classes = len(self.classes)
        input_size = 640.0  # Model input size

        # Log tensor info for debugging (first few frames only)
        if self._frame_count < 3:
            logger.info(f"Raw outputs: {len(outputs)} items")
            for k, v in outputs.items():
                if isinstance(v, np.ndarray):
                    logger.info(f"  '{k}': shape={v.shape}, dtype={v.dtype}")

        # Try to match tensors by name first (more reliable)
        # Expected names: bbox_scale0/cls_scale0, bbox_scale1/cls_scale1, bbox_scale2/cls_scale2
        # Or: output0, output1, etc.
        box_tensors_by_scale = {}
        class_tensors_by_scale = {}

        for name, tensor in outputs.items():
            if not isinstance(tensor, np.ndarray):
                continue

            # Normalize tensor to HWC format [H, W, C]
            tensor = self._normalize_tensor_format(tensor, num_classes)
            if tensor is None:
                continue

            h, w, c = tensor.shape

            # Identify by name pattern
            name_lower = name.lower()
            if 'bbox' in name_lower or 'box' in name_lower:
                # Extract scale from name (scale0, scale1, scale2) or use spatial size
                scale = self._get_scale_from_name_or_size(name_lower, h)
                box_tensors_by_scale[scale] = tensor
            elif 'cls' in name_lower or 'class' in name_lower:
                scale = self._get_scale_from_name_or_size(name_lower, h)
                class_tensors_by_scale[scale] = tensor
            else:
                # Identify by channel count
                if c == 64:
                    scale = self._get_scale_from_name_or_size(name_lower, h)
                    box_tensors_by_scale[scale] = tensor
                elif c == num_classes:
                    scale = self._get_scale_from_name_or_size(name_lower, h)
                    class_tensors_by_scale[scale] = tensor

        # If name-based matching didn't work, fall back to shape-based matching
        if not class_tensors_by_scale:
            logger.info("Name-based matching failed, trying shape-based matching")
            all_tensors = []
            for name, tensor in outputs.items():
                if isinstance(tensor, np.ndarray):
                    norm = self._normalize_tensor_format(tensor, num_classes)
                    if norm is not None:
                        all_tensors.append((name, norm))

            for name, tensor in all_tensors:
                h, w, c = tensor.shape
                scale = self._get_scale_from_name_or_size("", h)
                if c == 64:
                    box_tensors_by_scale[scale] = tensor
                elif c == num_classes:
                    class_tensors_by_scale[scale] = tensor

        if not class_tensors_by_scale:
            logger.warning("Could not identify class score tensors")
            return detections

        if self._frame_count < 3:
            logger.info(f"Matched tensors - boxes: {list(box_tensors_by_scale.keys())}, "
                       f"classes: {list(class_tensors_by_scale.keys())}")

        # Process each scale
        all_boxes = []
        all_scores = []
        all_class_ids = []

        for scale, class_tensor in class_tensors_by_scale.items():
            h, w, c = class_tensor.shape
            stride = int(input_size / h)  # 8, 16, or 32 for 80x80, 40x40, 20x20

            # Apply sigmoid to class scores
            scores = self._sigmoid(class_tensor)

            # Get corresponding box tensor
            box_tensor = box_tensors_by_scale.get(scale)

            if self._frame_count < 3:
                logger.info(f"Processing scale {scale}: grid={h}x{w}, stride={stride}, "
                           f"has_box_tensor={box_tensor is not None}")

            # For each cell, get best class
            for y in range(h):
                for x in range(w):
                    class_scores = scores[y, x]
                    max_score = np.max(class_scores)

                    if max_score < self.confidence_threshold:
                        continue

                    class_id = int(np.argmax(class_scores))

                    # Grid cell center in pixels
                    cx_pixels = (x + 0.5) * stride
                    cy_pixels = (y + 0.5) * stride

                    # Default box (if no box tensor)
                    x1_pixels = cx_pixels - stride
                    y1_pixels = cy_pixels - stride
                    x2_pixels = cx_pixels + stride
                    y2_pixels = cy_pixels + stride

                    # Get box from DFL tensor if available
                    if box_tensor is not None:
                        if box_tensor.shape[-1] == 64:
                            # DFL format - decode to [left, top, right, bottom] distances
                            box_data = box_tensor[y, x]
                            decoded = self._decode_dfl(box_data)

                            # DFL values are distances in grid cell units
                            # Multiply by stride to get pixel distances from cell center
                            left_dist = decoded[0] * stride
                            top_dist = decoded[1] * stride
                            right_dist = decoded[2] * stride
                            bottom_dist = decoded[3] * stride

                            # Calculate box edges
                            x1_pixels = cx_pixels - left_dist
                            y1_pixels = cy_pixels - top_dist
                            x2_pixels = cx_pixels + right_dist
                            y2_pixels = cy_pixels + bottom_dist

                        elif box_tensor.shape[-1] == 4:
                            # Direct format - could be xywh or ltrb
                            box_data = box_tensor[y, x]
                            # Assume xywh format, scaled by stride
                            bw_pixels = box_data[2] * stride
                            bh_pixels = box_data[3] * stride
                            x1_pixels = cx_pixels - bw_pixels / 2
                            y1_pixels = cy_pixels - bh_pixels / 2
                            x2_pixels = cx_pixels + bw_pixels / 2
                            y2_pixels = cy_pixels + bh_pixels / 2

                    # Clamp to image bounds
                    x1_pixels = max(0, x1_pixels)
                    y1_pixels = max(0, y1_pixels)
                    x2_pixels = min(input_size, x2_pixels)
                    y2_pixels = min(input_size, y2_pixels)

                    # Convert to normalized coordinates [0, 1]
                    x1_norm = x1_pixels / input_size
                    y1_norm = y1_pixels / input_size
                    w_norm = (x2_pixels - x1_pixels) / input_size
                    h_norm = (y2_pixels - y1_pixels) / input_size

                    # Filter invalid boxes
                    if w_norm > 0.01 and h_norm > 0.01 and w_norm < 0.95 and h_norm < 0.95:
                        all_boxes.append([x1_norm, y1_norm, w_norm, h_norm])
                        all_scores.append(float(max_score))
                        all_class_ids.append(class_id)

        if self._frame_count < 3:
            logger.info(f"Pre-NMS detections: {len(all_boxes)}")

        # Apply NMS
        if all_boxes:
            keep_indices = self._nms_numpy(
                np.array(all_boxes),
                np.array(all_scores),
                self.nms_iou_threshold
            )

            for idx in keep_indices:
                class_id = all_class_ids[idx]
                class_name = self.classes.get(str(class_id), f"class_{class_id}")
                box = all_boxes[idx]

                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=all_scores[idx],
                        bbox=BoundingBox(
                            x=box[0], y=box[1], width=box[2], height=box[3]
                        ),
                    )
                )

        return sorted(detections, key=lambda d: d.confidence, reverse=True)

    def _normalize_tensor_format(self, tensor: np.ndarray, num_classes: int) -> np.ndarray | None:
        """
        Normalize tensor to HWC format [H, W, C].

        Handles:
        - NCHW [batch, channels, H, W] -> [H, W, C]
        - CHW [channels, H, W] -> [H, W, C]
        - HWC [H, W, channels] -> [H, W, C] (no change)

        Args:
            tensor: Input tensor in any format
            num_classes: Number of classes (to help identify format)

        Returns:
            Tensor in HWC format, or None if format can't be determined
        """
        if tensor.ndim == 4:
            # NCHW format [batch, channels, H, W]
            # Remove batch dimension and transpose to HWC
            tensor = tensor[0]  # [C, H, W]
            tensor = np.transpose(tensor, (1, 2, 0))  # [H, W, C]
            return tensor

        elif tensor.ndim == 3:
            # Could be CHW or HWC
            d0, d1, d2 = tensor.shape

            # CHW format: first dim is channels (64 or num_classes)
            if d0 in (64, num_classes) and d1 == d2:
                # [C, H, W] -> [H, W, C]
                return np.transpose(tensor, (1, 2, 0))

            # HWC format: last dim is channels
            elif d2 in (64, num_classes) and d0 == d1:
                # Already [H, W, C]
                return tensor

            # Ambiguous - check if spatial dims are typical YOLO sizes
            yolo_sizes = {80, 40, 20}
            if d1 in yolo_sizes and d2 in yolo_sizes and d0 in (64, num_classes):
                # [C, H, W] -> [H, W, C]
                return np.transpose(tensor, (1, 2, 0))
            elif d0 in yolo_sizes and d1 in yolo_sizes and d2 in (64, num_classes):
                # [H, W, C] - already correct
                return tensor

            # Last resort: assume HWC if last dim is small
            if d2 <= 64:
                return tensor
            elif d0 <= 64:
                return np.transpose(tensor, (1, 2, 0))

        elif tensor.ndim == 2:
            # Flattened - can't use
            return None

        logger.warning(f"Cannot normalize tensor with shape {tensor.shape}")
        return None

    def _get_scale_from_name_or_size(self, name: str, spatial_size: int) -> int:
        """
        Get scale index from tensor name or spatial size.

        Args:
            name: Tensor name (e.g., 'bbox_scale0', 'cls_scale2')
            spatial_size: Spatial dimension (80, 40, or 20)

        Returns:
            Scale index (0, 1, or 2)
        """
        # Try to extract from name
        import re
        match = re.search(r'scale[_]?(\d+)', name)
        if match:
            return int(match.group(1))

        # Fall back to spatial size
        size_to_scale = {80: 0, 40: 1, 20: 2}
        return size_to_scale.get(spatial_size, 0)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _decode_dfl(self, dfl_output: np.ndarray, reg_max: int = 16) -> np.ndarray:
        """
        Decode DFL (Distribution Focal Loss) box regression.

        Args:
            dfl_output: (64,) array = 4 coords * 16 bins
            reg_max: Number of bins per coordinate (default 16)

        Returns:
            (4,) array of [left, top, right, bottom] offsets
        """
        # Reshape to (4, 16)
        dfl = dfl_output.reshape(4, reg_max)
        # Softmax over bins
        dfl_softmax = np.exp(dfl) / np.sum(np.exp(dfl), axis=1, keepdims=True)
        # Expected value
        bins = np.arange(reg_max)
        return np.sum(dfl_softmax * bins, axis=1)

    def _nms_numpy(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        """
        Simple NMS implementation in numpy.

        Args:
            boxes: (N, 4) array of [x, y, w, h]
            scores: (N,) array of confidence scores
            iou_threshold: IoU threshold for suppression

        Returns:
            List of indices to keep
        """
        if len(boxes) == 0:
            return []

        # Convert to x1, y1, x2, y2
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        areas = boxes[:, 2] * boxes[:, 3]
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU with rest
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h

            iou = intersection / (areas[i] + areas[order[1:]] - intersection + 1e-6)

            # Keep boxes with IoU below threshold
            mask = iou <= iou_threshold
            order = order[1:][mask]

        return keep

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
    print("=== Hailo Engine Test (picamera2 wrapper) ===")
    print(f"Hailo Available: {HAILO_AVAILABLE}")

    # Use config-based model path
    engine = _create_default_engine()

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
