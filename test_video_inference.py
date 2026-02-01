#!/usr/bin/env python3
"""
Video Inference Test

Runs detection on a video file to evaluate model performance before deployment.
Draws bounding boxes and saves annotated output video.

Usage:
    python test_video_inference.py input.mp4                    # Process video, save output
    python test_video_inference.py input.mp4 --show             # Also show preview window
    python test_video_inference.py input.mp4 --confidence 0.4   # Custom threshold
    python test_video_inference.py input.mp4 --skip 2           # Process every 2nd frame
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def draw_detections_cv2(frame: np.ndarray, detections: list, class_colors: dict) -> np.ndarray:
    """Draw bounding boxes and labels on frame using OpenCV."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    for det in detections:
        # Get box coordinates (normalized 0-1) and scale to frame size
        x = det.bbox.x * w
        y = det.bbox.y * h
        box_w = det.bbox.width * w
        box_h = det.bbox.height * h

        x1, y1 = int(x), int(y)
        x2, y2 = int(x + box_w), int(y + box_h)

        # Get color for class (BGR for OpenCV)
        color = class_colors.get(det.class_name, (255, 255, 255))

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw label with background
        label = f"{det.class_name}: {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Label background
        cv2.rectangle(annotated, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)

        # Label text
        cv2.putText(annotated, label, (x1 + 2, y1 - 5), font, font_scale, (0, 0, 0), thickness)

    return annotated


def print_confidence_histogram(confidences: list, bins: int = 10):
    """Print a text-based histogram of confidence scores."""
    if not confidences:
        print("  No detections to analyze")
        return

    hist, edges = np.histogram(confidences, bins=bins, range=(0, 1))
    max_count = max(hist) if max(hist) > 0 else 1

    print("\n  Confidence Distribution:")
    for i in range(bins):
        bar_len = int(40 * hist[i] / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {edges[i]:.1f}-{edges[i+1]:.1f}: {bar} ({hist[i]})")


def run_video_inference(
    video_path: str,
    output_path: str = None,
    confidence: float = 0.3,
    show_preview: bool = False,
    skip_frames: int = 1,
    max_frames: int = None,
):
    """Run inference on video file.

    Args:
        video_path: Path to input video file
        output_path: Path for output video (auto-generated if None)
        confidence: Confidence threshold for detections
        show_preview: Show live preview window
        skip_frames: Process every Nth frame (1 = all frames)
        max_frames: Maximum frames to process (None = all)
    """
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False

    # Auto-generate output path
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_detected{video_path.suffix}"
    output_path = Path(output_path)

    print("=" * 60)
    print("  Video Inference Test")
    print("=" * 60)
    print(f"Input:      {video_path}")
    print(f"Output:     {output_path}")
    print(f"Confidence: {confidence}")
    print(f"Skip:       every {skip_frames} frame(s)")

    # Class colors (BGR for OpenCV)
    class_colors = {
        "cat": (100, 255, 100),     # Green
        "rodent": (100, 255, 255),  # Yellow
    }

    # Open input video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo info:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS:        {fps:.1f}")
    print(f"  Frames:     {total_frames}")
    print(f"  Duration:   {total_frames/fps:.1f}s")

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
        print(f"Engine initialized (model: {model_path.name})")
    except Exception as e:
        print(f"Error: Failed to initialize engine: {e}")
        cap.release()
        return False

    # Initialize output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / skip_frames  # Adjust FPS if skipping frames
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

    if not writer.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        engine.cleanup()
        return False

    # Stats tracking
    frame_count = 0
    processed_count = 0
    total_detections = 0
    class_counts = defaultdict(int)
    all_confidences = []
    inference_times = []
    frames_with_detections = 0

    print(f"\nProcessing video...")
    print("-" * 60)

    start_time = time.time()
    last_progress_time = start_time

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames if requested
            if frame_count % skip_frames != 0:
                continue

            # Check max frames limit
            if max_frames and processed_count >= max_frames:
                break

            processed_count += 1

            # Resize for inference (640x640)
            inference_frame = cv2.resize(frame, (640, 640))
            # Convert BGR to RGB for model
            inference_frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)

            # Run inference
            result = engine.infer(inference_frame_rgb)
            inference_times.append(result.inference_time_ms)

            detections = result.detections

            # Track stats
            if detections:
                frames_with_detections += 1
                for det in detections:
                    class_counts[det.class_name] += 1
                    all_confidences.append(det.confidence)
                    total_detections += 1

            # Draw on original frame
            annotated = draw_detections_cv2(frame, detections, class_colors)

            # Add frame info overlay
            info_text = f"Frame {frame_count}/{total_frames} | Detections: {len(detections)}"
            cv2.putText(annotated, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

            # Write to output
            writer.write(annotated)

            # Show preview if requested
            if show_preview:
                # Resize for display if too large
                display_frame = annotated
                if width > 1280:
                    scale = 1280 / width
                    display_frame = cv2.resize(annotated, None, fx=scale, fy=scale)

                cv2.imshow("Detection Preview (press 'q' to quit)", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            # Progress update every 2 seconds
            if time.time() - last_progress_time >= 2:
                elapsed = time.time() - start_time
                progress = frame_count / total_frames * 100
                proc_fps = processed_count / elapsed
                avg_inference = sum(inference_times[-50:]) / len(inference_times[-50:])

                print(f"  [{progress:5.1f}%] Frame {frame_count}/{total_frames} | "
                      f"Detections: {total_detections} | "
                      f"Inference: {avg_inference:.1f}ms | "
                      f"FPS: {proc_fps:.1f}")

                last_progress_time = time.time()

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        engine.cleanup()

    # Calculate final stats
    elapsed = time.time() - start_time
    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0
    processing_fps = processed_count / elapsed if elapsed > 0 else 0

    # Print summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    print(f"\nProcessing Stats:")
    print(f"  Frames processed: {processed_count}/{total_frames}")
    print(f"  Processing time:  {elapsed:.1f}s")
    print(f"  Processing FPS:   {processing_fps:.1f}")
    print(f"  Avg inference:    {avg_inference:.1f}ms")

    print(f"\nDetection Stats:")
    print(f"  Total detections:       {total_detections}")
    print(f"  Frames with detections: {frames_with_detections} ({100*frames_with_detections/processed_count:.1f}%)")

    print(f"\nDetections by class:")
    for class_name in ["cat", "rodent"]:
        count = class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")

    # Confidence analysis
    if all_confidences:
        conf_array = np.array(all_confidences)
        print(f"\nConfidence Stats:")
        print(f"  Min:    {conf_array.min():.3f}")
        print(f"  Max:    {conf_array.max():.3f}")
        print(f"  Mean:   {conf_array.mean():.3f}")
        print(f"  Median: {np.median(conf_array):.3f}")

        print_confidence_histogram(all_confidences)

        # Warning signs
        print("\n" + "-" * 60)
        print("  HEALTH CHECK")
        print("-" * 60)

        issues = []
        if conf_array.max() < 0.5:
            issues.append("WARNING: All confidences below 0.5 - model may be underconfident")
        if conf_array.min() > 0.95:
            issues.append("WARNING: All confidences above 0.95 - possible quantization issue")
        if total_detections == 0:
            issues.append("CRITICAL: No detections at all - check if model outputs are nullified")
        if frames_with_detections > processed_count * 0.9:
            issues.append("WARNING: Detections in >90% of frames - possible false positive issue")

        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print("  OK: Confidence distribution looks healthy")
            print("  OK: Detection rate appears reasonable")

    else:
        print("\n  CRITICAL: No detections found in entire video!")
        print("  Possible causes:")
        print("    - Model outputs may be nullified (quantization failure)")
        print("    - Confidence threshold too high")
        print("    - No cats/rodents in the video")

    print(f"\nOutput saved: {output_path}")
    print("=" * 60)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run detection on video file to evaluate model performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_video_inference.py test.mp4
  python test_video_inference.py test.mp4 --show --confidence 0.4
  python test_video_inference.py test.mp4 --skip 3 --max-frames 500
        """
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path for output video (auto-generated if not specified)")
    parser.add_argument("--confidence", type=float, default=0.3,
                        help="Confidence threshold (default: 0.3)")
    parser.add_argument("--show", action="store_true",
                        help="Show preview window (requires display)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (default: 1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process (default: all)")

    args = parser.parse_args()

    success = run_video_inference(
        video_path=args.video,
        output_path=args.output,
        confidence=args.confidence,
        show_preview=args.show,
        skip_frames=args.skip,
        max_frames=args.max_frames,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
