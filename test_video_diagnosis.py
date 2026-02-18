#!/usr/bin/env python3
"""
Video Diagnosis Tool — Full pipeline analysis for prey detection debugging.

Runs the COMPLETE detection pipeline (HailoEngine + PreyDetector spatial logic)
on a video file and produces:
  1. Annotated output video with bounding boxes, state, and scores
  2. Per-frame CSV log with all detection details
  3. Terminal summary with detection statistics

Unlike test_video_inference.py (raw engine only), this tool runs the full
prey detection state machine so you can see exactly why detection succeeds or
fails in real scenarios.

IMPORTANT: Uses video timestamps (not wall clock) for the score accumulation
window and cat-lost reset, so results accurately reflect real-time behavior
regardless of processing speed.

Usage on Pi:
    python test_video_diagnosis.py input.mp4
    python test_video_diagnosis.py input.mp4 --engine-threshold 0.05
    python test_video_diagnosis.py input.mp4 --show-all-scores
    python test_video_diagnosis.py input.mp4 --skip 2 --max-frames 500
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


# ── Video-time state tracker ────────────────────────────────────────────────
# Mirrors PreyDetector logic but uses video timestamps instead of time.time()
# so diagnosis results match real-time behavior regardless of processing speed.

@dataclass
class PreyScoreEntry:
    """A prey score entry with video timestamp."""
    video_time: float
    confidence: float


class VideoTimeStateTracker:
    """
    Tracks prey detection state using video timestamps.

    This mirrors PreyDetector._update_state_score_accumulation() exactly,
    but replaces time.time() with the video timestamp so that the 5-second
    window and cat-lost reset work correctly when processing offline.
    """

    def __init__(
        self,
        prey_window_seconds: float = 5.0,
        prey_score_threshold: float = 0.9,
        prey_min_detection_score: float = 0.20,
        prey_min_detection_count: int = 3,
        reset_on_cat_lost_seconds: float = 5.0,
    ):
        self.prey_window_seconds = prey_window_seconds
        self.prey_score_threshold = prey_score_threshold
        self.prey_min_detection_score = prey_min_detection_score
        self.prey_min_detection_count = prey_min_detection_count
        self.reset_on_cat_lost_seconds = reset_on_cat_lost_seconds

        self.state = "IDLE"
        self._prey_scores: list[PreyScoreEntry] = []
        self._last_cat_time: float | None = None
        self._last_match = None
        self.confirmed_events: list[dict] = []

    @property
    def accumulated_score(self) -> float:
        return sum(e.confidence for e in self._prey_scores)

    @property
    def detection_count(self) -> int:
        return len(self._prey_scores)

    def update(self, video_time: float, frame_result, frame: np.ndarray | None = None) -> str | None:
        """
        Update state machine with a new frame result.

        Args:
            video_time: Current video timestamp in seconds
            frame_result: FrameResult from PreyDetector._evaluate_frame()
            frame: The inference frame (for event reporting)

        Returns:
            "CONFIRMED" if prey just confirmed this frame, else None
        """
        # Clean old scores outside window
        self._prey_scores = [
            entry for entry in self._prey_scores
            if video_time - entry.video_time < self.prey_window_seconds
        ]

        if frame_result.has_cat:
            self._last_cat_time = video_time

            # Start monitoring
            if self.state == "IDLE":
                self.state = "MONITORING"

            # Handle prey detection
            if frame_result.has_valid_prey and frame_result.prey_detection:
                prey = frame_result.prey_detection

                # Only accumulate if above minimum score
                if prey.confidence >= self.prey_min_detection_score:
                    self._prey_scores.append(PreyScoreEntry(
                        video_time=video_time,
                        confidence=prey.confidence,
                    ))

                    if frame_result.match:
                        self._last_match = frame_result.match

                    # Transition to VERIFYING
                    if self.state == "MONITORING":
                        self.state = "VERIFYING"

                    # Check dual confirmation
                    accumulated = self.accumulated_score
                    count = self.detection_count
                    if (accumulated >= self.prey_score_threshold
                            and count >= self.prey_min_detection_count
                            and self.state != "CONFIRMED"):
                        self.state = "CONFIRMED"
                        self.confirmed_events.append({
                            "video_time": video_time,
                            "accumulated_score": accumulated,
                            "detection_count": count,
                            "cat_detection": frame_result.cat_detection,
                            "prey_detection": prey,
                            "match": self._last_match,
                        })
                        return "CONFIRMED"
        else:
            # No cat detected
            if self._last_cat_time is not None:
                time_since_cat = video_time - self._last_cat_time
                if time_since_cat >= self.reset_on_cat_lost_seconds:
                    if self.state != "IDLE":
                        old_state = self.state
                        self.state = "IDLE"
                        self._prey_scores.clear()
                        self._last_match = None

        return None


# ── Colors (BGR for OpenCV) ─────────────────────────────────────────────────
COLOR_CAT = (100, 255, 100)          # Green — cat above threshold
COLOR_CAT_WEAK = (100, 180, 100)     # Dim green — cat below threshold
COLOR_RODENT = (0, 255, 255)         # Yellow — rodent above threshold
COLOR_RODENT_WEAK = (0, 180, 180)    # Dim yellow — rodent below threshold
COLOR_SPATIAL = (255, 100, 255)      # Magenta — expanded cat box (spatial)
COLOR_WHITE = (255, 255, 255)
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 165, 255)

STATE_COLORS = {
    "IDLE": (180, 180, 180),        # Gray
    "MONITORING": (255, 200, 0),    # Cyan-ish
    "VERIFYING": (0, 165, 255),     # Orange
    "CONFIRMED": (0, 0, 255),       # Red
}


def draw_diagnosis_frame(
    frame: np.ndarray,
    detections: list,
    frame_info: dict,
    thresholds: dict,
    show_all_scores: bool = False,
    box_expansion: float = 0.25,
) -> np.ndarray:
    """Draw comprehensive diagnostic annotations on frame."""
    h, w = frame.shape[:2]
    annotated = frame.copy()

    cat_det = None
    rodent_det = None
    cat_threshold = thresholds.get("cat", 0.60)
    rodent_threshold = thresholds.get("rodent", 0.15)

    # ── Draw all detections ──────────────────────────────────────────────
    for det in detections:
        x1 = int(det.bbox.x * w)
        y1 = int(det.bbox.y * h)
        x2 = int((det.bbox.x + det.bbox.width) * w)
        y2 = int((det.bbox.y + det.bbox.height) * h)

        if det.class_name == "cat":
            above = det.confidence >= cat_threshold
            color = COLOR_CAT if above else COLOR_CAT_WEAK
            thickness = 3 if above else 1
            if cat_det is None or det.confidence > cat_det.confidence:
                cat_det = det
        elif det.class_name == "rodent":
            above = det.confidence >= rodent_threshold
            color = COLOR_RODENT if above else COLOR_RODENT_WEAK
            thickness = 3 if above else 1
            if rodent_det is None or det.confidence > rodent_det.confidence:
                rodent_det = det
        else:
            color = COLOR_WHITE
            thickness = 1

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Label
        label = f"{det.class_name}: {det.confidence:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        (tw, th_text), baseline = cv2.getTextSize(label, font, font_scale, 1)

        label_y = y1 - 4 if y1 > 20 else y2 + th_text + 4

        # Background
        cv2.rectangle(
            annotated,
            (x1, label_y - th_text - 2),
            (x1 + tw + 4, label_y + 2),
            color, -1,
        )
        cv2.putText(annotated, label, (x1 + 2, label_y), font, font_scale, (0, 0, 0), 1)

        # Show pixel dimensions of detection box
        if show_all_scores:
            box_px_w = int(det.bbox.width * w)
            box_px_h = int(det.bbox.height * h)
            size_label = f"{box_px_w}x{box_px_h}px"
            cv2.putText(annotated, size_label, (x1, y2 + 14), font, 0.4, color, 1)

    # ── Draw expanded cat box (spatial validation area) ──────────────────
    if cat_det is not None:
        expanded = cat_det.bbox.expanded(box_expansion)
        ex1 = int(expanded.x * w)
        ey1 = int(expanded.y * h)
        ex2 = int((expanded.x + expanded.width) * w)
        ey2 = int((expanded.y + expanded.height) * h)
        cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), COLOR_SPATIAL, 1, cv2.LINE_AA)

    # ── HUD overlay ──────────────────────────────────────────────────────
    state = frame_info.get("state", "IDLE")
    state_color = STATE_COLORS.get(state, COLOR_WHITE)
    accum_score = frame_info.get("accumulated_score", 0.0)
    score_threshold = frame_info.get("score_threshold", 0.9)
    det_count = frame_info.get("detection_count", 0)
    min_count = frame_info.get("min_detection_count", 3)
    frame_num = frame_info.get("frame_number", 0)
    total_frames = frame_info.get("total_frames", 0)
    video_time = frame_info.get("video_time", 0.0)
    spatial_match = frame_info.get("spatial_match", "")
    has_valid_prey = frame_info.get("has_valid_prey", False)
    prey_accumulated = frame_info.get("prey_accumulated", False)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # State badge (top-left)
    badge_text = f" {state} "
    (btw, bth), _ = cv2.getTextSize(badge_text, font, 0.7, 2)
    cv2.rectangle(annotated, (5, 5), (15 + btw, 15 + bth), state_color, -1)
    cv2.putText(annotated, badge_text, (10, 10 + bth), font, 0.7, (0, 0, 0), 2)

    # Score bar (below state)
    bar_x, bar_y = 10, 25 + bth
    bar_w, bar_h = 200, 18
    fill_ratio = min(accum_score / score_threshold, 1.0) if score_threshold > 0 else 0
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)
    fill_color = COLOR_RED if fill_ratio >= 1.0 else COLOR_ORANGE
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + int(bar_w * fill_ratio), bar_y + bar_h), fill_color, -1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_WHITE, 1)
    score_text = f"Score: {accum_score:.2f}/{score_threshold} | Count: {det_count}/{min_count}"
    cv2.putText(annotated, score_text, (bar_x + 4, bar_y + 14), font, 0.4, COLOR_WHITE, 1)

    # Frame info (top-right)
    info_lines = [
        f"Frame {frame_num}/{total_frames}  t={video_time:.2f}s",
        f"Cat: {'YES' if cat_det else 'no'}" + (f" ({cat_det.confidence:.3f})" if cat_det else ""),
        f"Rodent: {'YES' if rodent_det else 'no'}" + (f" ({rodent_det.confidence:.3f})" if rodent_det else ""),
    ]
    if spatial_match:
        info_lines.append(f"Spatial: {spatial_match}")
    if has_valid_prey and prey_accumulated:
        info_lines.append(">> PREY ACCUMULATED <<")
    elif has_valid_prey:
        info_lines.append(">> VALID PREY (below min_det_score) <<")

    for i, line in enumerate(info_lines):
        y_pos = 20 + i * 18
        # Shadow for readability
        cv2.putText(annotated, line, (w - 350 + 1, y_pos + 1), font, 0.45, (0, 0, 0), 2)
        cv2.putText(annotated, line, (w - 350, y_pos), font, 0.45, COLOR_WHITE, 1)

    return annotated


def run_diagnosis(
    video_path: str,
    output_path: str = None,
    engine_threshold: float = None,
    show_all_scores: bool = False,
    show_preview: bool = False,
    skip_frames: int = 1,
    max_frames: int = None,
):
    """Run full pipeline diagnosis on a video file."""
    video_path = Path(video_path)
    if not video_path.exists():
        print(f"Error: Video file not found: {video_path}")
        return False

    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_diagnosis.mp4"
    output_path = Path(output_path)

    csv_path = output_path.with_suffix(".csv")

    print("=" * 70)
    print("  VIDEO DIAGNOSIS — Full Pipeline Analysis")
    print("=" * 70)
    print(f"Input:   {video_path}")
    print(f"Output:  {output_path}")
    print(f"CSV:     {csv_path}")

    # ── Open video ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo:   {width}x{height} @ {fps:.1f} FPS, {total_frames} frames ({total_frames/fps:.1f}s)")

    # ── Load config ──────────────────────────────────────────────────────
    try:
        from mousehunter.config import inference_config, PROJECT_ROOT

        if engine_threshold is None:
            engine_threshold = inference_config.engine_confidence_threshold
        thresholds = dict(inference_config.thresholds)
        box_expansion = inference_config.box_expansion
        prey_window = inference_config.prey_window_seconds
        prey_score_threshold = inference_config.prey_score_threshold
        prey_min_score = inference_config.prey_min_detection_score
        prey_min_count = inference_config.prey_min_detection_count
        reset_cat_lost = inference_config.reset_on_cat_lost_seconds
        model_path = Path(inference_config.model_path)
        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path
        classes = inference_config.classes
        reg_max = inference_config.reg_max
        print(f"\nConfig loaded from config.json")
    except ImportError:
        print("Warning: Could not load config, using defaults")
        if engine_threshold is None:
            engine_threshold = 0.10
        thresholds = {"cat": 0.60, "rodent": 0.15}
        box_expansion = 0.25
        prey_window = 5.0
        prey_score_threshold = 0.9
        prey_min_score = 0.20
        prey_min_count = 3
        reset_cat_lost = 5.0
        model_path = Path("models/yolov8n_catprey.hef")
        classes = {"0": "cat", "1": "rodent"}
        reg_max = 8

    print(f"\n--- Detection Config ---")
    print(f"  Engine threshold:    {engine_threshold}")
    print(f"  Cat threshold:       {thresholds.get('cat', '?')}")
    print(f"  Rodent threshold:    {thresholds.get('rodent', '?')}")
    print(f"  Box expansion:       {box_expansion}")
    print(f"  Prey window:         {prey_window}s")
    print(f"  Score threshold:     {prey_score_threshold}")
    print(f"  Min detection score: {prey_min_score}")
    print(f"  Min detection count: {prey_min_count}")
    print(f"  Cat lost reset:      {reset_cat_lost}s")
    print(f"  Model:               {model_path}")

    # ── Initialize engine ────────────────────────────────────────────────
    print(f"\nInitializing Hailo engine...")
    try:
        from mousehunter.inference.hailo_engine import HailoEngine

        engine = HailoEngine(
            model_path=model_path,
            confidence_threshold=engine_threshold,
            classes=classes,
            reg_max=reg_max,
        )
        print(f"  Engine ready (threshold={engine_threshold})")
    except Exception as e:
        print(f"Error: Engine init failed: {e}")
        cap.release()
        return False

    # ── Initialize spatial evaluator ─────────────────────────────────────
    # We use PreyDetector ONLY for _evaluate_frame() (spatial + threshold
    # logic, no time dependency). State tracking uses VideoTimeStateTracker.
    from mousehunter.inference.prey_detector import PreyDetector

    evaluator = PreyDetector(
        engine=engine,
        thresholds=thresholds,
        spatial_validation_enabled=True,
        box_expansion=box_expansion,
    )

    # ── Initialize video-time state tracker ──────────────────────────────
    tracker = VideoTimeStateTracker(
        prey_window_seconds=prey_window,
        prey_score_threshold=prey_score_threshold,
        prey_min_detection_score=prey_min_score,
        prey_min_detection_count=prey_min_count,
        reset_on_cat_lost_seconds=reset_cat_lost,
    )

    # ── Output video writer ──────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_fps = fps / skip_frames
    writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        engine.cleanup()
        return False

    # ── CSV log ──────────────────────────────────────────────────────────
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "video_time_s", "state",
        "cat_conf", "cat_above_thresh", "cat_bbox",
        "rodent_conf", "rodent_above_thresh", "rodent_bbox", "rodent_px_size",
        "spatial_match", "has_valid_prey", "prey_accumulated",
        "accum_score", "det_count",
        "all_detections",
        "inference_ms",
    ])

    # ── Stats ────────────────────────────────────────────────────────────
    frame_count = 0
    processed_count = 0
    stats = {
        "frames_with_cat_any": 0,          # Cat from engine (>= engine_threshold)
        "frames_with_cat_above_thresh": 0,  # Cat >= cat threshold (0.60)
        "frames_with_rodent_any": 0,        # Rodent from engine (>= engine_threshold)
        "frames_with_rodent_above_thresh": 0,  # Rodent >= rodent threshold (0.15)
        "frames_with_valid_prey": 0,        # Spatial match passed
        "frames_prey_accumulated": 0,       # Above min_detection_score (0.20), actually accumulated
        "max_rodent_conf": 0.0,
        "max_accum_score": 0.0,
        "rodent_confidences": [],
        "cat_confidences": [],
        "state_frames": defaultdict(int),
    }
    inference_times = []

    print(f"\nProcessing video...")
    print("-" * 70)

    wall_start = time.time()
    last_progress = wall_start

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            if max_frames and processed_count >= max_frames:
                break

            processed_count += 1
            video_time = frame_count / fps

            # ── Resize for inference (640x640 RGB) ───────────────────────
            inference_frame = cv2.resize(frame, (640, 640))
            inference_frame_rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)

            # ── Run inference ────────────────────────────────────────────
            result = engine.infer(inference_frame_rgb)
            inference_times.append(result.inference_time_ms)
            detections = result.detections

            # ── Evaluate frame (spatial + threshold, no time dependency) ──
            frame_result = evaluator._evaluate_frame(result)

            # ── Update state machine with video time ─────────────────────
            confirmed = tracker.update(video_time, frame_result, inference_frame_rgb)

            if confirmed:
                ev = tracker.confirmed_events[-1]
                print(f"\n  *** PREY CONFIRMED at t={video_time:.2f}s (frame {frame_count})! ***")
                print(f"      Score: {ev['accumulated_score']:.2f}/{prey_score_threshold}")
                print(f"      Count: {ev['detection_count']}/{prey_min_count}")
                print(f"      Prey:  {ev['prey_detection'].class_name} ({ev['prey_detection'].confidence:.3f})")

            # ── Gather state info ────────────────────────────────────────
            state_name = tracker.state
            accum_score = tracker.accumulated_score
            det_count_window = tracker.detection_count

            if accum_score > stats["max_accum_score"]:
                stats["max_accum_score"] = accum_score

            stats["state_frames"][state_name] += 1

            # ── Extract per-frame detection details ──────────────────────
            cat_conf = 0.0
            cat_above = False
            cat_bbox_str = ""
            rodent_conf = 0.0
            rodent_above = False
            rodent_bbox_str = ""
            rodent_px_size = ""
            spatial_match_str = ""
            has_valid_prey = frame_result.has_valid_prey
            prey_accumulated = False  # Did this frame's prey actually get accumulated?

            cat_threshold = thresholds.get("cat", 0.60)
            rodent_threshold = thresholds.get("rodent", 0.15)

            # Find best cat and best rodent from ALL engine detections
            best_cat = None
            best_rodent = None
            for det in detections:
                if det.class_name == "cat":
                    if best_cat is None or det.confidence > best_cat.confidence:
                        best_cat = det
                elif det.class_name == "rodent":
                    if best_rodent is None or det.confidence > best_rodent.confidence:
                        best_rodent = det

            if best_cat:
                cat_conf = best_cat.confidence
                cat_above = cat_conf >= cat_threshold
                cat_bbox_str = f"({best_cat.bbox.x:.3f},{best_cat.bbox.y:.3f},{best_cat.bbox.width:.3f},{best_cat.bbox.height:.3f})"
                stats["frames_with_cat_any"] += 1
                stats["cat_confidences"].append(cat_conf)
                if cat_above:
                    stats["frames_with_cat_above_thresh"] += 1

            if best_rodent:
                rodent_conf = best_rodent.confidence
                rodent_above = rodent_conf >= rodent_threshold
                rodent_bbox_str = f"({best_rodent.bbox.x:.3f},{best_rodent.bbox.y:.3f},{best_rodent.bbox.width:.3f},{best_rodent.bbox.height:.3f})"
                rpx_w = int(best_rodent.bbox.width * 640)
                rpx_h = int(best_rodent.bbox.height * 640)
                rodent_px_size = f"{rpx_w}x{rpx_h}"
                stats["frames_with_rodent_any"] += 1
                stats["rodent_confidences"].append(rodent_conf)
                if rodent_conf > stats["max_rodent_conf"]:
                    stats["max_rodent_conf"] = rodent_conf
                if rodent_above:
                    stats["frames_with_rodent_above_thresh"] += 1

            if frame_result.match:
                spatial_match_str = frame_result.match.intersection_type

            if has_valid_prey:
                stats["frames_with_valid_prey"] += 1
                # Check if this prey actually gets accumulated (>= min_detection_score)
                if (frame_result.prey_detection
                        and frame_result.prey_detection.confidence >= prey_min_score):
                    prey_accumulated = True
                    stats["frames_prey_accumulated"] += 1

            # All detections summary
            all_det_str = "; ".join(
                f"{d.class_name}:{d.confidence:.3f}" for d in detections
            )

            # ── Write CSV row ────────────────────────────────────────────
            csv_writer.writerow([
                frame_count, f"{video_time:.3f}", state_name,
                f"{cat_conf:.4f}", cat_above, cat_bbox_str,
                f"{rodent_conf:.4f}", rodent_above, rodent_bbox_str, rodent_px_size,
                spatial_match_str, has_valid_prey, prey_accumulated,
                f"{accum_score:.4f}", det_count_window,
                all_det_str,
                f"{result.inference_time_ms:.1f}",
            ])

            # ── Draw annotated frame ─────────────────────────────────────
            frame_info = {
                "state": state_name,
                "accumulated_score": accum_score,
                "score_threshold": prey_score_threshold,
                "detection_count": det_count_window,
                "min_detection_count": prey_min_count,
                "frame_number": frame_count,
                "total_frames": total_frames,
                "video_time": video_time,
                "spatial_match": spatial_match_str,
                "has_valid_prey": has_valid_prey,
                "prey_accumulated": prey_accumulated,
            }

            annotated = draw_diagnosis_frame(
                frame, detections, frame_info,
                thresholds=thresholds,
                show_all_scores=show_all_scores,
                box_expansion=box_expansion,
            )

            writer.write(annotated)

            # ── Show preview ─────────────────────────────────────────────
            if show_preview:
                display = annotated
                if width > 1280:
                    scale = 1280 / width
                    display = cv2.resize(annotated, None, fx=scale, fy=scale)
                cv2.imshow("Diagnosis (q=quit)", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break

            # ── Progress ─────────────────────────────────────────────────
            now = time.time()
            if now - last_progress >= 2:
                elapsed = now - wall_start
                progress = frame_count / total_frames * 100
                proc_fps = processed_count / elapsed
                avg_inf = sum(inference_times[-50:]) / len(inference_times[-50:])

                cat_str = f"cat:{cat_conf:.2f}" if cat_conf > 0 else "cat:---"
                rod_str = f"rod:{rodent_conf:.2f}" if rodent_conf > 0 else "rod:---"
                print(
                    f"  [{progress:5.1f}%] f={frame_count}/{total_frames} | "
                    f"{cat_str} {rod_str} | "
                    f"state={state_name} score={accum_score:.2f}/{prey_score_threshold} "
                    f"cnt={det_count_window}/{prey_min_count} | "
                    f"{avg_inf:.1f}ms {proc_fps:.1f}fps"
                )
                last_progress = now

            # ── Print notable events ─────────────────────────────────────
            if prey_accumulated:
                print(
                    f"  ** Frame {frame_count} (t={video_time:.2f}s): PREY ACCUMULATED — "
                    f"rodent={rodent_conf:.3f} spatial={spatial_match_str} "
                    f"score={accum_score:.2f}/{prey_score_threshold} "
                    f"count={det_count_window}/{prey_min_count}"
                )
            elif has_valid_prey:
                print(
                    f"  ** Frame {frame_count} (t={video_time:.2f}s): VALID PREY but "
                    f"below min_det_score ({frame_result.prey_detection.confidence:.3f}<{prey_min_score}) — NOT accumulated"
                )
            elif best_rodent:
                # Rodent detected by engine but didn't pass full pipeline
                reasons = []
                if not rodent_above:
                    reasons.append(f"below rodent threshold ({rodent_conf:.3f}<{rodent_threshold})")
                if not best_cat:
                    reasons.append("no cat detected")
                elif not cat_above:
                    reasons.append(f"cat below threshold ({cat_conf:.3f}<{cat_threshold})")
                if not spatial_match_str and cat_above and rodent_above:
                    reasons.append("no spatial match")

                if reasons:
                    reason_str = ", ".join(reasons)
                    print(
                        f"     Frame {frame_count} (t={video_time:.2f}s): rodent={rodent_conf:.3f} "
                        f"REJECTED: {reason_str}"
                    )

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")

    finally:
        cap.release()
        writer.release()
        csv_file.close()
        if show_preview:
            cv2.destroyAllWindows()
        engine.cleanup()

    # ── Summary ──────────────────────────────────────────────────────────
    elapsed = time.time() - wall_start
    avg_inference = sum(inference_times) / len(inference_times) if inference_times else 0

    print("\n" + "=" * 70)
    print("  DIAGNOSIS SUMMARY")
    print("=" * 70)

    print(f"\nProcessing:")
    print(f"  Frames processed:  {processed_count}/{total_frames}")
    print(f"  Wall time:         {elapsed:.1f}s")
    print(f"  Avg inference:     {avg_inference:.1f}ms")

    print(f"\nDetection Counts (engine threshold = {engine_threshold}):")
    print(f"  Frames with cat (any, >= {engine_threshold}):    {stats['frames_with_cat_any']}/{processed_count}")
    print(f"  Frames with cat (>= {cat_threshold}):          {stats['frames_with_cat_above_thresh']}/{processed_count}")
    print(f"  Frames with rodent (any, >= {engine_threshold}): {stats['frames_with_rodent_any']}/{processed_count}")
    print(f"  Frames with rodent (>= {rodent_threshold}):     {stats['frames_with_rodent_above_thresh']}/{processed_count}")
    print(f"  Frames with VALID prey (spatial OK):  {stats['frames_with_valid_prey']}/{processed_count}")
    print(f"  Frames prey ACCUMULATED (>= {prey_min_score}):  {stats['frames_prey_accumulated']}/{processed_count}")
    print(f"  Max rodent confidence:                {stats['max_rodent_conf']:.4f}")
    print(f"  Max accumulated score:                {stats['max_accum_score']:.4f}/{prey_score_threshold}")

    print(f"\nState Machine (video-time based):")
    for sn in ["IDLE", "MONITORING", "VERIFYING", "CONFIRMED"]:
        count = stats["state_frames"].get(sn, 0)
        pct = 100 * count / processed_count if processed_count > 0 else 0
        print(f"  {sn:12s}: {count:5d} ({pct:.1f}%)")

    confirmed_events = tracker.confirmed_events
    print(f"\nPrey Confirmed Events: {len(confirmed_events)}")
    for i, ev in enumerate(confirmed_events):
        print(f"  Event {i+1}: t={ev['video_time']:.2f}s, score={ev['accumulated_score']:.2f}, "
              f"count={ev['detection_count']}, "
              f"prey={ev['prey_detection'].class_name}({ev['prey_detection'].confidence:.3f})")

    # ── Confidence distributions ─────────────────────────────────────────
    if stats["cat_confidences"]:
        cat_arr = np.array(stats["cat_confidences"])
        print(f"\nCat Confidence Distribution:")
        print(f"  Min={cat_arr.min():.3f}  Max={cat_arr.max():.3f}  "
              f"Mean={cat_arr.mean():.3f}  Median={np.median(cat_arr):.3f}")

    if stats["rodent_confidences"]:
        rod_arr = np.array(stats["rodent_confidences"])
        print(f"\nRodent Confidence Distribution:")
        print(f"  Min={rod_arr.min():.3f}  Max={rod_arr.max():.3f}  "
              f"Mean={rod_arr.mean():.3f}  Median={np.median(rod_arr):.3f}")
        # Histogram
        bins = [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.0]
        hist, _ = np.histogram(rod_arr, bins=bins)
        print(f"  Histogram:")
        for i in range(len(bins) - 1):
            bar = "#" * min(hist[i], 50)
            marker = ""
            if bins[i] <= engine_threshold < bins[i+1]:
                marker = " <-- engine"
            if bins[i] <= rodent_threshold < bins[i+1]:
                marker = " <-- rodent thresh"
            if bins[i] <= prey_min_score < bins[i+1]:
                marker += " <-- min_det_score"
            print(f"    {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]:4d} {bar}{marker}")
    else:
        print(f"\n  !! NO rodent detections in entire video !!")
        print(f"  Possible causes:")
        print(f"    - Model cannot detect prey-in-mouth at this distance/angle")
        print(f"    - Engine threshold too high ({engine_threshold})")
        print(f"    - Quantization destroyed rodent features")
        print(f"    - Try: --engine-threshold 0.05 or even 0.01")

    # ── Diagnosis ────────────────────────────────────────────────────────
    print(f"\n--- DIAGNOSIS ---")
    issues = []

    if stats["frames_with_rodent_any"] == 0:
        issues.append(
            "CRITICAL: Model produced ZERO rodent detections. The model itself "
            "cannot see the rodent at all. This is a model/training issue, not a "
            "threshold issue. Try running with --engine-threshold 0.01 to confirm."
        )
    elif stats["frames_with_rodent_above_thresh"] == 0 and stats["frames_with_rodent_any"] > 0:
        issues.append(
            f"Rodent detected {stats['frames_with_rodent_any']} times but ALL below "
            f"rodent threshold ({rodent_threshold}). "
            f"Max confidence was {stats['max_rodent_conf']:.3f}. "
            f"Consider lowering rodent threshold."
        )
    elif stats["frames_with_valid_prey"] == 0 and stats["frames_with_rodent_above_thresh"] > 0:
        issues.append(
            f"Rodent passed threshold {stats['frames_with_rodent_above_thresh']} times "
            f"but NONE passed spatial validation. Cat and rodent boxes don't overlap enough. "
            f"Check the annotated video for box positions."
        )
    elif stats["frames_prey_accumulated"] == 0 and stats["frames_with_valid_prey"] > 0:
        issues.append(
            f"Valid prey matches found ({stats['frames_with_valid_prey']} frames) but "
            f"all below min_detection_score ({prey_min_score}). "
            f"Max rodent was {stats['max_rodent_conf']:.3f}. "
            f"Consider lowering min_detection_score."
        )
    elif stats["max_accum_score"] < prey_score_threshold and stats["frames_prey_accumulated"] > 0:
        issues.append(
            f"Prey accumulated {stats['frames_prey_accumulated']} times but "
            f"score ({stats['max_accum_score']:.2f}) never reached threshold "
            f"({prey_score_threshold}). Detections too sporadic or scores too low."
        )

    if len(confirmed_events) == 0 and stats["frames_prey_accumulated"] > 0:
        issues.append(
            f"Prey accumulated but never confirmed. "
            f"Max score={stats['max_accum_score']:.2f}/{prey_score_threshold}. "
            f"Need more consistent detections within {prey_window}s window."
        )

    if not issues:
        if len(confirmed_events) > 0:
            issues.append("Detection pipeline worked correctly — prey was confirmed.")
        else:
            issues.append("No clear single point of failure identified. Review CSV for details.")

    for i, issue in enumerate(issues):
        print(f"  {i+1}. {issue}")

    print(f"\nOutput files:")
    print(f"  Video: {output_path}")
    print(f"  CSV:   {csv_path}")
    print("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline diagnosis for prey detection debugging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_video_diagnosis.py video.mp4
  python test_video_diagnosis.py video.mp4 --engine-threshold 0.05
  python test_video_diagnosis.py video.mp4 --show-all-scores --show
  python test_video_diagnosis.py video.mp4 --skip 2 --max-frames 500
        """
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Output video path (auto-generated if not set)")
    parser.add_argument("--engine-threshold", type=float, default=None,
                        help="Engine confidence threshold (default: from config, typically 0.10)")
    parser.add_argument("--show-all-scores", action="store_true",
                        help="Show pixel dimensions of detection boxes")
    parser.add_argument("--show", action="store_true",
                        help="Show preview window (requires display)")
    parser.add_argument("--skip", type=int, default=1,
                        help="Process every Nth frame (default: 1)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Maximum frames to process")

    args = parser.parse_args()

    success = run_diagnosis(
        video_path=args.video,
        output_path=args.output,
        engine_threshold=args.engine_threshold,
        show_all_scores=args.show_all_scores,
        show_preview=args.show,
        skip_frames=args.skip,
        max_frames=args.max_frames,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
