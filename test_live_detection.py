#!/usr/bin/env python3
"""
Live Detection Test

Tests the custom YOLOv8 model with real camera images.
Saves annotated frames showing detections.

Usage:
    python test_live_detection.py              # Run for 30 seconds
    python test_live_detection.py --duration 60  # Run for 60 seconds
    python test_live_detection.py --save-all   # Save every frame (not just detections)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Output directory for saved frames
OUTPUT_DIR = Path(__file__).parent / "test_output"


def draw_detections(frame: np.ndarray, detections: list, class_colors: dict) -> np.ndarray:
    """Draw bounding boxes and labels on frame."""
    from PIL import Image, ImageDraw, ImageFont

    # Convert to PIL for drawing
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    h, w = frame.shape[:2]

    for det in detections:
        # Get box coordinates (normalized 0-1)
        x = det.bbox.x * w
        y = det.bbox.y * h
        box_w = det.bbox.width * w
        box_h = det.bbox.height * h

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + box_w), int(y + box_h)

        # Get color for class
        color = class_colors.get(det.class_name, (255, 255, 255))

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Draw label
        label = f"{det.class_name}: {det.confidence:.2f}"

        # Label background
        bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill=(0, 0, 0), font=font)

    return np.array(img)


def run_live_detection(duration: int = 30, save_all: bool = False, confidence: float = 0.3,
                       vflip: bool = False, hflip: bool = False):
    """Run live detection test.

    Args:
        duration: Test duration in seconds
        save_all: Save periodic frames even without detections
        confidence: Confidence threshold for detections
        vflip: Vertical flip (for upside-down mounted camera)
        hflip: Horizontal flip (for mirror-mounted camera)
    """

    print("=" * 60)
    print("  Live Detection Test")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_DIR / f"session_{timestamp}"
    session_dir.mkdir(exist_ok=True)
    print(f"Output directory: {session_dir}")

    # Class colors (BGR for consistency, but we use RGB)
    class_colors = {
        "bird": (255, 100, 100),    # Red
        "cat": (100, 255, 100),     # Green
        "leaf": (100, 100, 255),    # Blue
        "rodent": (255, 255, 100),  # Yellow
    }

    # Initialize camera
    print("\nInitializing camera...")
    try:
        from picamera2 import Picamera2
        from libcamera import Transform

        camera = Picamera2()

        # Create transform for camera orientation
        # vflip=True, hflip=True = 180 degree rotation (for upside-down mounted camera)
        transform = Transform(vflip=vflip, hflip=hflip)

        config = camera.create_video_configuration(
            main={"size": (1280, 720), "format": "RGB888"},
            lores={"size": (640, 640), "format": "RGB888"},
            buffer_count=4,
            transform=transform,
        )
        camera.configure(config)
        camera.start()
        flip_status = f"vflip={vflip}, hflip={hflip}" if (vflip or hflip) else "no flip"
        print(f"Camera started ({flip_status})")

        # Let camera warm up
        time.sleep(1)
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return False

    # Initialize Hailo engine
    print("\nInitializing Hailo engine...")
    try:
        from mousehunter.inference.hailo_engine import HailoEngine
        from mousehunter.config import inference_config, PROJECT_ROOT

        model_path = Path(inference_config.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        engine = HailoEngine(
            model_path=model_path,
            confidence_threshold=confidence,
            classes=inference_config.classes,
        )
        print(f"Engine initialized (threshold={confidence})")
    except Exception as e:
        print(f"Engine initialization failed: {e}")
        camera.stop()
        camera.close()
        return False

    # Stats
    frame_count = 0
    detection_count = 0
    total_detections = 0
    frames_with_detections = 0
    inference_times = []
    class_counts = {name: 0 for name in class_colors.keys()}

    print(f"\nRunning detection for {duration} seconds...")
    print("Press Ctrl+C to stop early\n")
    print("-" * 60)

    start_time = time.time()
    last_status_time = start_time

    try:
        while time.time() - start_time < duration:
            # Capture frame
            lores_frame = camera.capture_array("lores")
            main_frame = camera.capture_array("main")

            frame_count += 1

            # Run inference on lores (640x640)
            result = engine.infer(lores_frame)
            inference_times.append(result.inference_time_ms)

            detections = result.detections

            # Count detections by class
            for det in detections:
                if det.class_name in class_counts:
                    class_counts[det.class_name] += 1
                total_detections += 1

            # If we have detections, save the frame
            if detections:
                frames_with_detections += 1
                detection_count += 1

                # Draw on main frame (higher res)
                # Scale detection boxes from 640x640 to main frame size
                annotated = draw_detections(main_frame, detections, class_colors)

                # Save frame
                frame_path = session_dir / f"detection_{detection_count:04d}.jpg"
                from PIL import Image
                Image.fromarray(annotated).save(frame_path, quality=90)

                # Print detection
                det_str = ", ".join([f"{d.class_name}:{d.confidence:.2f}" for d in detections])
                print(f"[Frame {frame_count}] DETECTED: {det_str}")
                print(f"  Saved: {frame_path.name}")

            elif save_all and frame_count % 30 == 0:
                # Save periodic frames even without detections
                frame_path = session_dir / f"frame_{frame_count:04d}.jpg"
                from PIL import Image
                Image.fromarray(main_frame).save(frame_path, quality=85)

            # Print status every 5 seconds
            if time.time() - last_status_time >= 5:
                elapsed = time.time() - start_time
                avg_fps = frame_count / elapsed
                avg_inference = sum(inference_times[-100:]) / len(inference_times[-100:])

                print(f"\n[Status] {elapsed:.0f}s elapsed | "
                      f"Frames: {frame_count} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Inference: {avg_inference:.1f}ms | "
                      f"Detections: {total_detections}")

                last_status_time = time.time()

    except KeyboardInterrupt:
        print("\n\nStopped by user")

    finally:
        # Cleanup
        camera.stop()
        camera.close()
        engine.cleanup()

    # Print summary
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"Duration:              {elapsed:.1f} seconds")
    print(f"Total frames:          {frame_count}")
    print(f"Average FPS:           {avg_fps:.1f}")
    print(f"Average inference:     {avg_inference:.1f}ms")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Total detections:      {total_detections}")
    print(f"\nDetections by class:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")
    print(f"\nSaved frames: {session_dir}")
    print("=" * 60)

    # Save summary to file
    summary_path = session_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Live Detection Test Summary\n")
        f.write(f"{'='*40}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Duration: {elapsed:.1f}s\n")
        f.write(f"Frames: {frame_count}\n")
        f.write(f"FPS: {avg_fps:.1f}\n")
        f.write(f"Avg inference: {avg_inference:.1f}ms\n")
        f.write(f"Confidence threshold: {confidence}\n")
        f.write(f"\nDetections by class:\n")
        for class_name, count in class_counts.items():
            f.write(f"  {class_name}: {count}\n")
        f.write(f"\nTotal detections: {total_detections}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Live Detection Test")
    parser.add_argument("--duration", type=int, default=30,
                        help="Test duration in seconds (default: 30)")
    parser.add_argument("--save-all", action="store_true",
                        help="Save periodic frames even without detections")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--vflip", action="store_true",
                        help="Vertical flip (for upside-down mounted camera)")
    parser.add_argument("--hflip", action="store_true",
                        help="Horizontal flip (for mirrored camera)")
    parser.add_argument("--flip", action="store_true",
                        help="Apply both vflip and hflip (180° rotation)")
    args = parser.parse_args()

    # --flip is shorthand for both vflip and hflip
    vflip = args.vflip or args.flip
    hflip = args.hflip or args.flip

    success = run_live_detection(
        duration=args.duration,
        save_all=args.save_all,
        confidence=args.confidence,
        vflip=vflip,
        hflip=hflip,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
