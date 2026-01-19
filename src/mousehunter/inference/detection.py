"""
Detection data structures for object detection results.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class BoundingBox:
    """Bounding box coordinates (normalized 0-1 or pixel coordinates)."""

    x: float  # Left edge
    y: float  # Top edge
    width: float
    height: float

    @property
    def center_x(self) -> float:
        """Get center X coordinate."""
        return self.x + self.width / 2

    @property
    def center_y(self) -> float:
        """Get center Y coordinate."""
        return self.y + self.height / 2

    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height

    @property
    def right(self) -> float:
        """Get right edge X coordinate."""
        return self.x + self.width

    @property
    def bottom(self) -> float:
        """Get bottom edge Y coordinate."""
        return self.y + self.height

    def iou(self, other: "BoundingBox") -> float:
        """
        Calculate Intersection over Union with another box.

        Args:
            other: Another bounding box

        Returns:
            IoU value (0-1)
        """
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.right, other.right)
        y2 = min(self.bottom, other.bottom)

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        union = self.area + other.area - intersection

        return intersection / union if union > 0 else 0.0

    def intersects(self, other: "BoundingBox") -> bool:
        """
        Check if this box intersects (overlaps) with another box.

        Args:
            other: Another bounding box

        Returns:
            True if boxes overlap, False otherwise
        """
        return not (
            self.right < other.x
            or other.right < self.x
            or self.bottom < other.y
            or other.bottom < self.y
        )

    def expanded(self, factor: float) -> "BoundingBox":
        """
        Return a new box expanded by a factor around its center.

        Args:
            factor: Expansion factor (0.25 = 25% larger in each dimension)

        Returns:
            New expanded BoundingBox
        """
        expand_w = self.width * factor
        expand_h = self.height * factor
        return BoundingBox(
            x=self.x - expand_w / 2,
            y=self.y - expand_h / 2,
            width=self.width + expand_w,
            height=self.height + expand_h,
        )

    def contains_point(self, px: float, py: float) -> bool:
        """
        Check if a point is inside this bounding box.

        Args:
            px: Point X coordinate
            py: Point Y coordinate

        Returns:
            True if point is inside box
        """
        return self.x <= px <= self.right and self.y <= py <= self.bottom

    def to_pixels(self, width: int, height: int) -> "BoundingBox":
        """
        Convert normalized coordinates to pixel coordinates.

        Args:
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            New BoundingBox with pixel coordinates
        """
        return BoundingBox(
            x=self.x * width,
            y=self.y * height,
            width=self.width * width,
            height=self.height * height,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "center_x": self.center_x,
            "center_y": self.center_y,
        }


@dataclass
class Detection:
    """Single object detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox": self.bbox.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"{self.class_name} ({self.confidence:.2f}) "
            f"at ({self.bbox.center_x:.2f}, {self.bbox.center_y:.2f})"
        )


@dataclass
class DetectionFrame:
    """Collection of detections from a single frame."""

    timestamp: datetime
    detections: list[Detection]
    frame_number: int = 0
    inference_time_ms: float = 0.0

    @property
    def has_cat(self) -> bool:
        """Check if frame contains a cat detection."""
        return any(d.class_name.lower() == "cat" for d in self.detections)

    @property
    def has_prey(self) -> bool:
        """Check if frame contains a prey detection."""
        return any(d.class_name.lower() == "prey" for d in self.detections)

    @property
    def has_cat_with_prey(self) -> bool:
        """Check if frame shows cat with prey (both detected)."""
        return self.has_cat and self.has_prey

    def get_by_class(self, class_name: str) -> list[Detection]:
        """Get all detections of a specific class."""
        return [d for d in self.detections if d.class_name.lower() == class_name.lower()]

    def get_highest_confidence(self, class_name: str | None = None) -> Detection | None:
        """Get detection with highest confidence."""
        candidates = self.detections
        if class_name:
            candidates = self.get_by_class(class_name)
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.confidence)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "frame_number": self.frame_number,
            "inference_time_ms": self.inference_time_ms,
            "detections": [d.to_dict() for d in self.detections],
            "has_cat": self.has_cat,
            "has_prey": self.has_prey,
        }
