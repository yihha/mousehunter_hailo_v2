"""
Video Encoder and Evidence Recorder

Provides H.264 MP4 encoding via ffmpeg and orchestrates
pre-roll + post-roll evidence capture in background threads.
"""

import gc
import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np

from .circular_buffer import BufferedFrame

logger = logging.getLogger(__name__)

# Check ffmpeg availability at import time
FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None

if not FFMPEG_AVAILABLE:
    logger.warning("ffmpeg not found - video encoding unavailable, will use legacy JPEG mode")


class VideoEncoder:
    """
    Low-level H.264 video encoder using ffmpeg subprocess.

    Pipes raw RGB24 frames to ffmpeg stdin for encoding to MP4.
    """

    ENCODE_TIMEOUT = 120  # seconds

    @staticmethod
    def encode_frames(
        frames: list[BufferedFrame],
        output_path: Path,
        framerate: float = 30.0,
    ) -> bool:
        """
        Encode a list of buffered frames to an H.264 MP4 file.

        Args:
            frames: List of BufferedFrame objects with frame_data (RGB numpy arrays)
            output_path: Path for the output .mp4 file
            framerate: Video framerate

        Returns:
            True if encoding succeeded, False otherwise
        """
        if not FFMPEG_AVAILABLE:
            logger.error("ffmpeg not available, cannot encode video")
            return False

        if not frames:
            logger.warning("No frames to encode")
            return False

        # Get dimensions from first frame
        h, w = frames[0].frame_data.shape[:2]

        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(framerate),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        logger.info(
            f"Encoding {len(frames)} frames ({w}x{h} @ {framerate}fps) -> {output_path}"
        )

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Pipe all frames to ffmpeg stdin
            for frame in frames:
                try:
                    raw_bytes = frame.frame_data.astype(np.uint8).tobytes()
                    proc.stdin.write(raw_bytes)
                except (BrokenPipeError, OSError):
                    logger.error("ffmpeg pipe broken during frame write")
                    break

            proc.stdin.close()

            # Wait for ffmpeg to finish
            _, stderr = proc.communicate(timeout=VideoEncoder.ENCODE_TIMEOUT)

            if proc.returncode != 0:
                logger.error(
                    f"ffmpeg exited with code {proc.returncode}: "
                    f"{stderr.decode('utf-8', errors='replace')[-500:]}"
                )
                return False

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Video encoded successfully: {output_path} ({file_size_mb:.1f}MB)")
            return True

        except subprocess.TimeoutExpired:
            logger.error(f"ffmpeg timed out after {VideoEncoder.ENCODE_TIMEOUT}s")
            proc.kill()
            proc.wait()
            return False
        except Exception as e:
            logger.error(f"Video encoding failed: {e}", exc_info=True)
            return False


class EvidenceRecorder:
    """
    Orchestrates evidence recording with pre-roll + post-roll.

    On trigger, captures pre-roll frames from the circular buffer,
    then spawns a background thread to collect post-roll frames
    and encode everything to an H.264 MP4.

    Falls back to legacy JPEG mode if ffmpeg is unavailable.
    """

    def __init__(
        self,
        framerate: float = 30.0,
        post_roll_seconds: float = 15.0,
        output_dir: str | Path = "runtime/recordings",
        evidence_format: str = "video",
    ):
        """
        Initialize the evidence recorder.

        Args:
            framerate: Camera framerate for post-roll timing
            post_roll_seconds: Seconds of post-event footage to capture
            output_dir: Base directory for evidence output
            evidence_format: "video" for H.264 MP4, "frames" for legacy JPEGs
        """
        self.framerate = framerate
        self.post_roll_seconds = post_roll_seconds
        self.output_dir = Path(output_dir)
        self.evidence_format = evidence_format

        self._on_complete_callbacks: list[Callable[[Path, bool], None]] = []
        self._encoding_thread: threading.Thread | None = None

        # Auto-fallback to frames if ffmpeg unavailable
        if self.evidence_format == "video" and not FFMPEG_AVAILABLE:
            logger.warning("ffmpeg unavailable, falling back to legacy JPEG evidence mode")
            self.evidence_format = "frames"

    def on_complete(self, callback: Callable[[Path, bool], None]) -> None:
        """
        Register a callback for when evidence encoding completes.

        Args:
            callback: Function(evidence_dir: Path, success: bool)
        """
        self._on_complete_callbacks.append(callback)

    def trigger_evidence_save(
        self,
        event_name: str,
        pre_frames: list[BufferedFrame],
        camera_get_main_frame: Callable,
    ) -> Path:
        """
        Trigger evidence recording (non-blocking).

        Returns the evidence directory immediately. A background thread
        handles post-roll capture and video encoding.

        Args:
            event_name: Name for the evidence directory
            pre_frames: Pre-roll frames from the circular buffer
            camera_get_main_frame: Callable that returns (frame, timestamp) tuple

        Returns:
            Path to the evidence directory (created immediately)
        """
        # Create evidence directory (event_name already includes timestamp)
        evidence_dir = self.output_dir / event_name
        evidence_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Evidence triggered: {evidence_dir} "
            f"(pre={len(pre_frames)} frames, post={self.post_roll_seconds}s, "
            f"format={self.evidence_format})"
        )

        # Check if a previous recording is still in progress
        if self._encoding_thread is not None and self._encoding_thread.is_alive():
            logger.warning(
                "Previous evidence recording still in progress, "
                "starting new recording in parallel"
            )

        # Spawn background thread for post-roll + encoding
        self._encoding_thread = threading.Thread(
            target=self._background_record,
            args=(evidence_dir, pre_frames, camera_get_main_frame),
            name="EvidenceRecorderThread",
            daemon=True,
        )
        self._encoding_thread.start()

        return evidence_dir

    def _background_record(
        self,
        evidence_dir: Path,
        pre_frames: list[BufferedFrame],
        camera_get_main_frame: Callable,
    ) -> None:
        """
        Background thread: collect post-roll, encode video, fire callbacks.

        Args:
            evidence_dir: Output directory for evidence
            pre_frames: Pre-roll frames already captured
            camera_get_main_frame: Callable returning (frame, timestamp)
        """
        success = False
        try:
            # 1. Collect post-roll frames
            post_frames = self._collect_post_roll(camera_get_main_frame)

            # 2. Combine pre + post frames
            all_frames = list(pre_frames) + post_frames

            if not all_frames:
                logger.warning("No frames available for evidence recording")
                return

            # 3. Encode or save
            if self.evidence_format == "video":
                output_path = evidence_dir / "evidence.mp4"
                success = VideoEncoder.encode_frames(
                    all_frames, output_path, self.framerate
                )
                if not success:
                    # Fallback: save as JPEGs if encoding fails
                    logger.warning("Video encoding failed, falling back to JPEG evidence")
                    self._save_frames_as_jpeg(all_frames, evidence_dir)
            else:
                self._save_frames_as_jpeg(all_frames, evidence_dir)
                success = True

            # 4. Free frame memory
            del all_frames
            del post_frames
            del pre_frames
            gc.collect()

            logger.info(f"Evidence recording complete: {evidence_dir} (success={success})")

        except Exception as e:
            logger.error(f"Evidence recording failed: {e}", exc_info=True)
        finally:
            # Fire completion callbacks
            for callback in self._on_complete_callbacks:
                try:
                    callback(evidence_dir, success)
                except Exception as e:
                    logger.error(f"Evidence complete callback error: {e}")

    def _collect_post_roll(
        self,
        camera_get_main_frame: Callable,
    ) -> list[BufferedFrame]:
        """
        Collect post-roll frames by polling the camera.

        Args:
            camera_get_main_frame: Callable returning (frame, timestamp)

        Returns:
            List of BufferedFrame objects for the post-roll period
        """
        post_frames: list[BufferedFrame] = []
        target_interval = 1.0 / self.framerate
        end_time = time.monotonic() + self.post_roll_seconds
        frame_number = 0

        logger.info(f"Collecting post-roll frames for {self.post_roll_seconds}s...")

        while time.monotonic() < end_time:
            loop_start = time.monotonic()

            try:
                frame, timestamp = camera_get_main_frame()
                if frame is not None:
                    post_frames.append(BufferedFrame(
                        timestamp=timestamp or datetime.now(),
                        frame_data=frame.copy(),
                        frame_number=frame_number,
                    ))
                    frame_number += 1
            except Exception as e:
                logger.debug(f"Post-roll frame capture error: {e}")

            # Maintain framerate
            elapsed = time.monotonic() - loop_start
            sleep_time = target_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        logger.info(f"Post-roll complete: {len(post_frames)} frames captured")
        return post_frames

    @staticmethod
    def _save_frames_as_jpeg(
        frames: list[BufferedFrame],
        output_dir: Path,
    ) -> int:
        """
        Save frames as individual JPEG files (legacy fallback).

        Args:
            frames: List of BufferedFrame objects
            output_dir: Directory to save into

        Returns:
            Number of frames saved
        """
        try:
            from PIL import Image
        except ImportError:
            logger.error("PIL not available, cannot save JPEG frames")
            return 0

        saved = 0
        for i, frame in enumerate(frames):
            try:
                filename = f"frame_{i:04d}_{frame.timestamp.strftime('%H%M%S_%f')}.jpg"
                filepath = output_dir / filename
                img = Image.fromarray(frame.frame_data)
                img.save(filepath, "JPEG", quality=85)
                saved += 1
            except Exception as e:
                logger.error(f"Failed to save frame {i}: {e}")

        logger.info(f"Saved {saved} JPEG frames to {output_dir}")
        return saved
