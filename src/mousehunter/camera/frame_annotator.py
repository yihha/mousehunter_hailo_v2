"""
Frame Annotation Utility

Draws bounding boxes and labels on detection frames for evidence capture.
Uses PIL ImageDraw for annotation with graceful fallback.
"""

import logging
from io import BytesIO

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image, ImageDraw, ImageFont

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available - frame annotation disabled")

# Cached font instance
_cached_font: "ImageFont.FreeTypeFont | ImageFont.ImageFont | None" = None

# Color scheme: class_name -> RGB tuple
CLASS_COLORS = {
    "cat": (0, 200, 0),  # Green
    "rodent": (255, 0, 0),  # Red
}
DEFAULT_COLOR = (255, 255, 0)  # Yellow for unknown classes


def _get_font(size: int = 16) -> "ImageFont.FreeTypeFont | ImageFont.ImageFont":
    """Get font for label rendering, with caching and fallback."""
    global _cached_font
    if _cached_font is not None:
        return _cached_font

    try:
        _cached_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except (OSError, IOError):
        logger.debug("DejaVuSans-Bold not found, using PIL default font")
        _cached_font = ImageFont.load_default()

    return _cached_font


def annotate_frame(
    frame: np.ndarray,
    detections: list,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.

    Args:
        frame: RGB numpy array (H, W, 3)
        detections: List of Detection objects with .class_name, .confidence, .bbox

    Returns:
        Annotated frame as numpy array (same shape as input)
    """
    if not PIL_AVAILABLE:
        logger.warning("PIL not available, returning unannotated frame")
        return frame.copy()

    if not detections:
        return frame.copy()

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = _get_font()

    h, w = frame.shape[:2]

    for det in detections:
        if det is None:
            continue

        # Convert normalized bbox to pixel coordinates
        pixel_box = det.bbox.to_pixels(w, h)
        x1, y1 = int(pixel_box.x), int(pixel_box.y)
        x2, y2 = int(pixel_box.x + pixel_box.width), int(pixel_box.y + pixel_box.height)

        color = CLASS_COLORS.get(det.class_name, DEFAULT_COLOR)

        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label with background fill
        label = f"{det.class_name}: {det.confidence:.0%}"
        # Place label above box, or below top edge if near frame top
        label_y = y1 - 20 if y1 >= 20 else y2 + 2
        label_bbox = draw.textbbox((x1, label_y), label, font=font)
        draw.rectangle(label_bbox, fill=color)
        draw.text((x1, label_y), label, fill=(0, 0, 0), font=font)

    return np.array(img)


def annotate_frame_to_jpeg(
    frame: np.ndarray,
    detections: list,
    quality: int = 85,
) -> bytes:
    """
    Annotate a frame and encode as JPEG bytes.

    Args:
        frame: RGB numpy array (H, W, 3)
        detections: List of Detection objects
        quality: JPEG quality (1-100)

    Returns:
        JPEG-encoded bytes of the annotated frame

    Raises:
        RuntimeError: If PIL is not available
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("PIL not available, cannot encode JPEG")

    annotated = annotate_frame(frame, detections)
    img = Image.fromarray(annotated)
    buf = BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()
