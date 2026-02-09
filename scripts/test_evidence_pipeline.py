#!/usr/bin/env python3
"""
Evidence Pipeline Integration Test — Run on Raspberry Pi

Tests the full CircularOutput2 + H264Encoder + PyavOutput evidence pipeline
that was introduced in Session 7 to replace the raw-frame deque (2.7GB -> 19MB).

Tests:
  1. Camera starts with hardware H.264 encoder + CircularOutput2
  2. Lores snapshot capture (RGB888 640x640)
  3. Evidence MP4 save (pre-roll + post-roll via open_output/close_output)
  4. MP4 file validity (non-zero, playable header check)
  5. Thread safety (open_output from main, close_output from timer thread)
  6. Evidence serialization guard (reject parallel saves)
  7. Completion callback fires
  8. Memory stays flat during evidence save
  9. sdnotify importable

Usage:
  python scripts/test_evidence_pipeline.py

Requires: Pi hardware with PiCamera 3, picamera2, PyavOutput (pyav)
"""

import gc
import logging
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_evidence")

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
results: list[tuple[str, str, str]] = []


def record(name: str, passed: bool, detail: str = ""):
    status = PASS if passed else FAIL
    results.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))


def record_skip(name: str, reason: str):
    results.append((name, SKIP, reason))
    print(f"  [{SKIP}] {name} — {reason}")


def get_rss_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback to /proc on Linux
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024  # kB -> MB
        except Exception:
            pass
    return -1


def is_valid_mp4(path: Path) -> tuple[bool, str]:
    """Check if an MP4 file has a valid ftyp or moov box."""
    if not path.exists():
        return False, "file does not exist"
    size = path.stat().st_size
    if size < 8:
        return False, f"file too small ({size} bytes)"
    with open(path, "rb") as f:
        header = f.read(12)
    # MP4 starts with a box: [size:4][type:4]
    # Common first boxes: ftyp, moov, mdat, free
    if len(header) < 8:
        return False, "could not read header"
    box_type = header[4:8]
    known_boxes = [b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide"]
    if box_type in known_boxes:
        return True, f"valid MP4 ({size / 1024:.1f} KB, first box: {box_type.decode()})"
    return False, f"unknown header box: {box_type!r} ({size} bytes)"


def probe_mp4_with_ffprobe(path: Path) -> tuple[bool, str]:
    """Use ffprobe to verify MP4 is fully playable."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "format=duration,size", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            duration = parts[0] if parts else "?"
            return True, f"ffprobe OK (duration={duration}s)"
        return False, f"ffprobe failed: {result.stderr.strip()}"
    except FileNotFoundError:
        return False, "ffprobe not installed"
    except subprocess.TimeoutExpired:
        return False, "ffprobe timed out"


# ──────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("  MouseHunter Evidence Pipeline Test")
    print("  Validates CircularOutput2 + H264Encoder + PyavOutput")
    print("=" * 65)
    print()

    # ── Test 0: Dependency checks ────────────────────────────────────
    print("--- Dependency Checks ---")

    # sdnotify
    try:
        import sdnotify
        record("sdnotify import", True, f"version: {getattr(sdnotify, '__version__', 'unknown')}")
    except ImportError:
        record("sdnotify import", False, "pip install sdnotify")

    # picamera2
    try:
        from picamera2 import Picamera2
        record("picamera2 import", True)
    except ImportError:
        record("picamera2 import", False, "not available — cannot run hardware tests")
        print("\nCannot proceed without picamera2. Exiting.")
        _print_summary()
        return

    # H264Encoder
    try:
        from picamera2.encoders import H264Encoder
        record("H264Encoder import", True)
    except ImportError:
        record("H264Encoder import", False)
        print("\nCannot proceed without H264Encoder. Exiting.")
        _print_summary()
        return

    # CircularOutput2
    try:
        from picamera2.outputs import CircularOutput2
        record("CircularOutput2 import", True)
    except ImportError:
        record("CircularOutput2 import", False, "upgrade picamera2")
        print("\nCannot proceed without CircularOutput2. Exiting.")
        _print_summary()
        return

    # PyavOutput
    try:
        from picamera2.outputs import PyavOutput
        record("PyavOutput import", True)
    except ImportError:
        record("PyavOutput import", False, "pip install av (or upgrade picamera2)")
        print("\nCannot proceed without PyavOutput. Exiting.")
        _print_summary()
        return

    # PIL
    try:
        from PIL import Image
        record("PIL/Pillow import", True)
    except ImportError:
        record("PIL/Pillow import", False, "pip install Pillow")

    print()

    # ── Test 1: Camera start with H.264 encoder ──────────────────────
    print("--- Test 1: Camera + H.264 Encoder Startup ---")

    with tempfile.TemporaryDirectory(prefix="mh_test_") as tmpdir:
        output_dir = Path(tmpdir)

        try:
            picam2 = Picamera2()
            config = picam2.create_video_configuration(
                main={"size": (1920, 1080), "format": "YUV420"},
                lores={"size": (640, 640), "format": "RGB888"},
                controls={"FrameRate": 30},
            )
            picam2.configure(config)
            record("camera configure (YUV420 main + RGB888 lores)", True)
        except Exception as e:
            record("camera configure", False, str(e))
            _print_summary()
            return

        encoder = H264Encoder(bitrate=5_000_000, repeat=True)
        circular = CircularOutput2(buffer_duration_ms=10000)  # 10s for test

        try:
            picam2.start_recording(encoder, circular)
            record("start_recording (H264 + CircularOutput2)", True)
        except Exception as e:
            record("start_recording", False, str(e))
            picam2.close()
            _print_summary()
            return

        # Let buffer fill for a few seconds
        print("  ... filling circular buffer (5s) ...")
        time.sleep(5)

        # ── Test 2: Lores snapshot ───────────────────────────────────
        print()
        print("--- Test 2: Lores Snapshot (RGB888 640x640) ---")

        try:
            frame = picam2.capture_array("lores")
            shape = frame.shape
            is_rgb = len(shape) == 3 and shape[2] == 3
            is_640 = shape[0] == 640 and shape[1] == 640
            record(
                "capture_array('lores') shape",
                is_rgb and is_640,
                f"shape={shape}, dtype={frame.dtype}",
            )
        except Exception as e:
            record("capture_array('lores')", False, str(e))

        # Save as JPEG
        try:
            img = Image.fromarray(frame)
            snap_path = output_dir / "test_snapshot.jpg"
            img.save(snap_path, "JPEG", quality=85)
            snap_size = snap_path.stat().st_size
            record(
                "JPEG snapshot save",
                snap_size > 1000,
                f"{snap_size / 1024:.1f} KB",
            )
        except Exception as e:
            record("JPEG snapshot save", False, str(e))

        # ── Test 2b: Main stream is YUV420 (NOT RGB) ────────────────
        try:
            main_frame = picam2.capture_array("main")
            main_shape = main_frame.shape
            # YUV420 returns 2D array: (height * 3 // 2, width) — NOT 3-channel RGB
            is_2d = len(main_shape) == 2
            expected_h = 1080 * 3 // 2  # 1620
            record(
                "capture_array('main') is YUV420 (2D)",
                is_2d,
                f"shape={main_shape} ({'2D YUV420' if is_2d else '3D — UNEXPECTED'})",
            )
        except Exception as e:
            record("capture_array('main') YUV420 check", False, str(e))

        # ── Test 3: Evidence MP4 save ────────────────────────────────
        print()
        print("--- Test 3: Evidence MP4 Save (pre-roll + post-roll) ---")

        mem_before = get_rss_mb()
        evidence_dir = output_dir / "prey_test_001"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = evidence_dir / "evidence.mp4"

        completion_event = threading.Event()
        completion_result: dict = {}

        post_roll_seconds = 5  # short for testing

        try:
            # Open output — flushes pre-roll buffer into file
            circular.open_output(PyavOutput(str(mp4_path)))
            record("circular.open_output(PyavOutput(...))", True)
        except Exception as e:
            record("circular.open_output", False, str(e))
            picam2.stop_recording()
            picam2.close()
            _print_summary()
            return

        # Simulate post-roll in background thread (mirrors camera_service.py)
        def post_roll_thread():
            try:
                time.sleep(post_roll_seconds)
                circular.close_output()
                completion_result["success"] = True
                completion_result["thread"] = threading.current_thread().name
            except Exception as e:
                completion_result["success"] = False
                completion_result["error"] = str(e)
            finally:
                completion_event.set()

        bg_thread = threading.Thread(
            target=post_roll_thread,
            name="TestPostRollThread",
            daemon=True,
        )
        bg_thread.start()

        print(f"  ... recording post-roll ({post_roll_seconds}s) ...")
        completed = completion_event.wait(timeout=post_roll_seconds + 10)

        record(
            "post-roll thread completed",
            completed and completion_result.get("success", False),
            f"thread={completion_result.get('thread', '?')}"
            + (f", error={completion_result.get('error')}" if not completion_result.get("success") else ""),
        )

        # ── Test 4: MP4 validity ─────────────────────────────────────
        print()
        print("--- Test 4: MP4 File Validity ---")

        valid, detail = is_valid_mp4(mp4_path)
        record("MP4 header check", valid, detail)

        probe_ok, probe_detail = probe_mp4_with_ffprobe(mp4_path)
        if "not installed" in probe_detail:
            record_skip("ffprobe validation", probe_detail)
        else:
            record("ffprobe validation", probe_ok, probe_detail)

        # ── Test 5: Thread safety (close from different thread) ──────
        print()
        print("--- Test 5: Thread Safety ---")
        record(
            "open_output (main) / close_output (bg thread)",
            completed and completion_result.get("success", False),
            "cross-thread open/close succeeded" if completion_result.get("success") else "FAILED",
        )

        # ── Test 6: Evidence serialization guard ─────────────────────
        print()
        print("--- Test 6: Evidence Serialization Guard ---")

        # Start a new evidence save
        mp4_path_2 = evidence_dir / "evidence_2.mp4"
        try:
            circular.open_output(PyavOutput(str(mp4_path_2)))
            # Try to open a SECOND output while first is active — should fail or be rejected
            mp4_path_3 = evidence_dir / "evidence_3.mp4"
            second_rejected = False
            try:
                circular.open_output(PyavOutput(str(mp4_path_3)))
                # If it didn't raise, close it
                circular.close_output()
                # CircularOutput2 might silently close previous — check if documented
                record(
                    "parallel open_output rejected",
                    False,
                    "second open_output did NOT raise (may close previous implicitly)",
                )
            except Exception as e:
                second_rejected = True
                record("parallel open_output rejected", True, f"correctly raised: {type(e).__name__}")
            finally:
                if not second_rejected:
                    # Clean up first recording
                    try:
                        circular.close_output()
                    except Exception:
                        pass
        except Exception as e:
            record("serialization guard setup", False, str(e))

        # ── Test 7: Memory check ─────────────────────────────────────
        print()
        print("--- Test 7: Memory Usage ---")

        gc.collect()
        mem_after = get_rss_mb()
        if mem_before > 0 and mem_after > 0:
            mem_delta = mem_after - mem_before
            record(
                "memory delta during evidence save",
                mem_delta < 200,  # should be <50 MB, 200 is generous safety margin
                f"before={mem_before:.0f}MB, after={mem_after:.0f}MB, delta={mem_delta:+.0f}MB",
            )
        else:
            record_skip("memory delta", "psutil not available and /proc not readable")

        # ── Test 8: CameraService integration ────────────────────────
        print()
        print("--- Test 8: CameraService Integration ---")

        # Stop the raw picamera2 first
        try:
            picam2.stop_recording()
            picam2.close()
        except Exception:
            pass

        time.sleep(1)

        # Now test via CameraService (the actual class used in production)
        try:
            sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
            from mousehunter.camera.camera_service import CameraService, PICAMERA_AVAILABLE, PYAV_AVAILABLE

            svc_dir = output_dir / "svc_test"
            svc = CameraService(
                main_resolution=(1920, 1080),
                inference_resolution=(640, 640),
                framerate=30,
                buffer_seconds=10.0,
                output_dir=str(svc_dir),
                post_roll_seconds=5.0,
                evidence_format="video",
            )

            # Register completion callback
            cb_event = threading.Event()
            cb_results: dict = {}

            def on_complete(edir: Path, success: bool):
                cb_results["dir"] = str(edir)
                cb_results["success"] = success
                cb_event.set()

            svc.on_evidence_complete(on_complete)

            svc.start()
            record("CameraService.start()", True, f"hw_encoder={svc._h264_encoder is not None}")

            # Let buffer fill
            print("  ... filling buffer (5s) ...")
            time.sleep(5)

            # Snapshot test
            snap_bytes = svc.capture_snapshot_bytes()
            record(
                "CameraService.capture_snapshot_bytes()",
                snap_bytes is not None and len(snap_bytes) > 1000,
                f"{len(snap_bytes) / 1024:.1f} KB" if snap_bytes else "None returned",
            )

            # Trigger evidence
            edir = svc.trigger_evidence_save("test_detection")
            record(
                "CameraService.trigger_evidence_save()",
                edir is not None,
                str(edir) if edir else "None",
            )

            # Wait for completion callback
            print("  ... waiting for evidence completion (~10s) ...")
            cb_fired = cb_event.wait(timeout=20)
            record(
                "on_evidence_complete callback",
                cb_fired and cb_results.get("success", False),
                f"success={cb_results.get('success')}, dir={cb_results.get('dir', '?')}",
            )

            # Verify MP4
            if edir:
                svc_mp4 = edir / "evidence.mp4"
                sv, sd = is_valid_mp4(svc_mp4)
                record("CameraService evidence MP4 valid", sv, sd)

                sp_ok, sp_detail = probe_mp4_with_ffprobe(svc_mp4)
                if "not installed" in sp_detail:
                    record_skip("CameraService ffprobe", sp_detail)
                else:
                    record("CameraService ffprobe validation", sp_ok, sp_detail)

            # Status check
            status = svc.get_status()
            record(
                "CameraService.get_status()",
                status.get("hw_encoder") is True and status.get("started") is True,
                f"hw_encoder={status.get('hw_encoder')}, frames={status.get('frame_count')}",
            )

            svc.cleanup()
            record("CameraService.cleanup()", True)

        except Exception as e:
            record("CameraService integration", False, f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary ──────────────────────────────────────────────────────
    print()
    _print_summary()


def _print_summary():
    print("=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    total = len(results)
    passed = sum(1 for _, s, _ in results if PASS in s)
    failed = sum(1 for _, s, _ in results if FAIL in s)
    skipped = sum(1 for _, s, _ in results if SKIP in s)
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}  |  Skipped: {skipped}")
    print()

    if failed > 0:
        print("  FAILURES:")
        for name, status, detail in results:
            if FAIL in status:
                print(f"    - {name}: {detail}")
        print()
        print("  VERDICT: NOT READY — fix failures above before deploying")
    else:
        print("  VERDICT: ALL TESTS PASSED — evidence pipeline is production-ready")
    print("=" * 65)


if __name__ == "__main__":
    main()
