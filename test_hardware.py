#!/usr/bin/env python3
"""
MouseHunter Hardware & Model Test Suite

Comprehensive test script to verify all hardware components and the custom YOLO model.
Run on Raspberry Pi with Hailo-8L AI HAT+.

Usage:
    python test_hardware.py           # Run all tests
    python test_hardware.py --quick   # Skip slow tests
    python test_hardware.py --no-gpio # Skip GPIO/jammer test
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


class TestResult:
    """Store test results."""
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.skipped = False
        self.error = None
        self.details = []
        self.duration_ms = 0.0

    def __str__(self):
        if self.skipped:
            return f"[SKIP] {self.name}"
        elif self.passed:
            return f"[PASS] {self.name} ({self.duration_ms:.1f}ms)"
        else:
            return f"[FAIL] {self.name}: {self.error}"


class HardwareTester:
    """Unified hardware test suite."""

    def __init__(self, skip_gpio: bool = False, quick_mode: bool = False):
        self.skip_gpio = skip_gpio
        self.quick_mode = quick_mode
        self.results: list[TestResult] = []
        self.project_root = Path(__file__).parent
        self.model_path = self.project_root / "models" / "yolov8n_catprey.hef"

    def print_header(self, title: str):
        """Print section header."""
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")

    def print_subheader(self, title: str):
        """Print subsection header."""
        print(f"\n--- {title} ---")

    def run_test(self, name: str, test_func, *args, **kwargs) -> TestResult:
        """Run a single test and record result."""
        result = TestResult(name)
        print(f"\n[TEST] {name}...")

        start = time.perf_counter()
        try:
            test_func(result, *args, **kwargs)
            result.passed = True
        except SkipTest as e:
            result.skipped = True
            result.error = str(e)
            print(f"  SKIPPED: {e}")
        except Exception as e:
            result.error = str(e)
            print(f"  FAILED: {e}")

        result.duration_ms = (time.perf_counter() - start) * 1000
        self.results.append(result)

        if result.passed:
            print(f"  PASSED ({result.duration_ms:.1f}ms)")
        for detail in result.details:
            print(f"    - {detail}")

        return result

    def run_all(self):
        """Run all hardware tests."""
        self.print_header("MouseHunter Hardware Test Suite")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Model: {self.model_path}")
        print(f"Quick mode: {self.quick_mode}")
        print(f"Skip GPIO: {self.skip_gpio}")

        # 1. Prerequisites
        self.print_header("1. Prerequisites Check")
        self.run_test("Model file exists", self.test_model_exists)
        self.run_test("Config loads correctly", self.test_config_loads)
        self.run_test("Hailo device available", self.test_hailo_available)
        self.run_test("PiCamera2 available", self.test_picamera_available)
        self.run_test("GPIO available", self.test_gpio_available)

        # 2. Hailo Inference
        self.print_header("2. Hailo Inference Tests")
        self.run_test("Load HEF model", self.test_load_hef)
        self.run_test("HailoEngine initialization", self.test_hailo_engine_init)
        self.run_test("Single frame inference", self.test_single_inference)
        if not self.quick_mode:
            self.run_test("Inference benchmark (10 frames)", self.test_inference_benchmark)

        # 3. Camera
        self.print_header("3. Camera Tests")
        self.run_test("Camera initialization", self.test_camera_init)
        if not self.quick_mode:
            self.run_test("Camera capture frames", self.test_camera_capture)

        # 4. GPIO/Jammer
        self.print_header("4. GPIO/Jammer Tests")
        if self.skip_gpio:
            result = TestResult("Jammer GPIO control")
            result.skipped = True
            result.error = "Skipped by user request"
            self.results.append(result)
            print(f"\n[TEST] Jammer GPIO control...")
            print(f"  SKIPPED: --no-gpio flag set")
        else:
            self.run_test("Jammer GPIO control", self.test_jammer)

        # 5. Integration
        self.print_header("5. Integration Tests")
        self.run_test("PreyDetector initialization", self.test_prey_detector_init)
        if not self.quick_mode:
            self.run_test("Full detection pipeline", self.test_full_pipeline)

        # Summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        self.print_header("TEST SUMMARY")

        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.skipped)
        skipped = sum(1 for r in self.results if r.skipped)
        total = len(self.results)

        print(f"\nTotal:   {total}")
        print(f"Passed:  {passed}")
        print(f"Failed:  {failed}")
        print(f"Skipped: {skipped}")

        if failed > 0:
            print("\nFailed tests:")
            for r in self.results:
                if not r.passed and not r.skipped:
                    print(f"  - {r.name}: {r.error}")

        print(f"\n{'='*60}")
        if failed == 0:
            print("  ALL TESTS PASSED!")
        else:
            print(f"  {failed} TEST(S) FAILED")
        print(f"{'='*60}\n")

        return failed == 0

    # =========================================================================
    # Test implementations
    # =========================================================================

    def test_model_exists(self, result: TestResult):
        """Check if model file exists."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        result.details.append(f"Size: {size_mb:.1f} MB")

    def test_config_loads(self, result: TestResult):
        """Check if config loads correctly."""
        from mousehunter.config import inference_config

        result.details.append(f"Model path: {inference_config.model_path}")
        result.details.append(f"Classes: {inference_config.classes}")
        result.details.append(f"Thresholds: {inference_config.thresholds}")

        # Verify class mapping
        expected_classes = {"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"}
        if inference_config.classes != expected_classes:
            raise ValueError(f"Class mapping mismatch! Expected {expected_classes}, got {inference_config.classes}")

    def test_hailo_available(self, result: TestResult):
        """Check if Hailo device is available."""
        try:
            from picamera2.devices import Hailo
            result.details.append("picamera2.devices.Hailo: available")
        except ImportError:
            # Try low-level API
            try:
                from hailo_platform import VDevice
                vdevice = VDevice()
                result.details.append("hailo_platform.VDevice: available")
                del vdevice
            except Exception as e:
                raise RuntimeError(f"Hailo not available: {e}")

    def test_picamera_available(self, result: TestResult):
        """Check if PiCamera2 is available."""
        try:
            from picamera2 import Picamera2
            result.details.append("picamera2: available")
        except ImportError as e:
            raise SkipTest(f"picamera2 not installed: {e}")

    def test_gpio_available(self, result: TestResult):
        """Check if GPIO is available."""
        try:
            from gpiozero import OutputDevice
            result.details.append("gpiozero: available")
        except ImportError as e:
            raise SkipTest(f"gpiozero not installed: {e}")

    def test_load_hef(self, result: TestResult):
        """Test loading HEF model directly."""
        try:
            from hailo_platform import HEF
            hef = HEF(str(self.model_path))

            # Get input info
            input_info = hef.get_input_vstream_infos()[0]
            result.details.append(f"Input: {input_info.name}, shape={input_info.shape}")

            # Get output info
            output_infos = hef.get_output_vstream_infos()
            result.details.append(f"Outputs: {len(output_infos)} tensors")
            for info in output_infos[:3]:  # Show first 3
                result.details.append(f"  - {info.name}: {info.shape}")

            del hef
        except ImportError:
            raise SkipTest("hailo_platform not available")

    def test_hailo_engine_init(self, result: TestResult):
        """Test HailoEngine initialization."""
        from mousehunter.inference.hailo_engine import HailoEngine, HAILO_AVAILABLE

        result.details.append(f"HAILO_AVAILABLE: {HAILO_AVAILABLE}")

        engine = HailoEngine(
            model_path=self.model_path,
            confidence_threshold=0.5,
            classes={"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"},
        )

        result.details.append(f"Initialized: {engine._initialized}")
        result.details.append(f"Using Hailo: {engine._use_hailo}")

        if engine._input_shape:
            result.details.append(f"Input shape: {engine._input_shape}")

        engine.cleanup()

    def test_single_inference(self, result: TestResult):
        """Test single frame inference."""
        import numpy as np
        from mousehunter.inference.hailo_engine import HailoEngine

        engine = HailoEngine(
            model_path=self.model_path,
            confidence_threshold=0.3,  # Lower threshold to catch more
            classes={"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"},
        )

        # Create test frame (random noise)
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Run inference
        detection_result = engine.infer(test_frame)

        result.details.append(f"Inference time: {detection_result.inference_time_ms:.2f}ms")
        result.details.append(f"Detections: {len(detection_result.detections)}")

        for det in detection_result.detections[:5]:
            result.details.append(f"  - {det.class_name}: {det.confidence:.2f}")

        engine.cleanup()

    def test_inference_benchmark(self, result: TestResult):
        """Benchmark inference speed."""
        import numpy as np
        from mousehunter.inference.hailo_engine import HailoEngine

        engine = HailoEngine(
            model_path=self.model_path,
            confidence_threshold=0.5,
            classes={"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"},
        )

        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        # Warm up
        for _ in range(3):
            engine.infer(test_frame)

        # Benchmark
        times = []
        for i in range(10):
            start = time.perf_counter()
            engine.infer(test_frame)
            times.append((time.perf_counter() - start) * 1000)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        fps = 1000 / avg_time

        result.details.append(f"Avg: {avg_time:.2f}ms")
        result.details.append(f"Min: {min_time:.2f}ms")
        result.details.append(f"Max: {max_time:.2f}ms")
        result.details.append(f"FPS: {fps:.1f}")

        engine.cleanup()

    def test_camera_init(self, result: TestResult):
        """Test camera initialization."""
        try:
            from picamera2 import Picamera2

            camera = Picamera2()
            result.details.append("Camera created")

            # Get sensor info
            sensor_modes = camera.sensor_modes
            result.details.append(f"Sensor modes: {len(sensor_modes)}")

            camera.close()
            result.details.append("Camera closed")

        except ImportError:
            raise SkipTest("picamera2 not available")
        except Exception as e:
            if "Camera" in str(e) or "camera" in str(e):
                raise SkipTest(f"Camera not connected: {e}")
            raise

    def test_camera_capture(self, result: TestResult):
        """Test camera frame capture."""
        try:
            from picamera2 import Picamera2
            import numpy as np

            camera = Picamera2()

            # Configure for inference
            config = camera.create_video_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (640, 640), "format": "RGB888"},
            )
            camera.configure(config)
            camera.start()

            # Capture frames
            frames_captured = 0
            for i in range(5):
                frame = camera.capture_array("lores")
                if frame is not None and frame.shape == (640, 640, 3):
                    frames_captured += 1
                time.sleep(0.1)

            camera.stop()
            camera.close()

            result.details.append(f"Frames captured: {frames_captured}/5")
            if frames_captured < 3:
                raise RuntimeError("Too few frames captured")

        except ImportError:
            raise SkipTest("picamera2 not available")

    def test_jammer(self, result: TestResult):
        """Test jammer GPIO control."""
        try:
            from mousehunter.hardware.jammer import Jammer, GPIO_AVAILABLE

            result.details.append(f"GPIO_AVAILABLE: {GPIO_AVAILABLE}")

            jammer = Jammer(pin=17, active_high=True, max_on_duration=5.0)
            result.details.append(f"Jammer created on GPIO {jammer.pin}")

            # Test activation
            activated = jammer.activate()
            result.details.append(f"Activated: {activated}")
            result.details.append(f"Is active: {jammer.is_active}")

            time.sleep(0.5)

            # Test deactivation
            deactivated = jammer.deactivate(reason="Test")
            result.details.append(f"Deactivated: {deactivated}")

            jammer.cleanup()

        except ImportError:
            raise SkipTest("gpiozero not available")

    def test_prey_detector_init(self, result: TestResult):
        """Test PreyDetector initialization."""
        from mousehunter.inference.hailo_engine import HailoEngine
        from mousehunter.inference.prey_detector import PreyDetector

        engine = HailoEngine(
            model_path=self.model_path,
            confidence_threshold=0.5,
            classes={"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"},
        )

        detector = PreyDetector(
            engine=engine,
            thresholds={"cat": 0.55, "rodent": 0.45, "bird": 0.80, "leaf": 0.90},
            window_size=5,
            trigger_count=3,
            spatial_validation_enabled=True,
            box_expansion=0.25,
        )

        status = detector.get_status()
        result.details.append(f"State: {status['state']}")
        result.details.append(f"Window size: {status['window_size']}")
        result.details.append(f"Trigger count: {status['trigger_count']}")

        engine.cleanup()

    def test_full_pipeline(self, result: TestResult):
        """Test full detection pipeline with camera."""
        import numpy as np
        from mousehunter.inference.hailo_engine import HailoEngine
        from mousehunter.inference.prey_detector import PreyDetector

        engine = HailoEngine(
            model_path=self.model_path,
            confidence_threshold=0.5,
            classes={"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"},
        )

        detector = PreyDetector(
            engine=engine,
            thresholds={"cat": 0.55, "rodent": 0.45, "bird": 0.80, "leaf": 0.90},
            window_size=5,
            trigger_count=3,
        )

        # Process synthetic frames
        total_detections = 0
        for i in range(10):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            frame_result = detector.process_frame(frame)
            if frame_result and frame_result.detections:
                total_detections += len(frame_result.detections)

        result.details.append(f"Frames processed: 10")
        result.details.append(f"Total detections: {total_detections}")
        result.details.append(f"Window fill: {detector.window_fill}")
        result.details.append(f"State: {detector.state.name}")

        engine.cleanup()


class SkipTest(Exception):
    """Raised when a test should be skipped."""
    pass


def main():
    parser = argparse.ArgumentParser(description="MouseHunter Hardware Test Suite")
    parser.add_argument("--quick", action="store_true", help="Skip slow benchmark tests")
    parser.add_argument("--no-gpio", action="store_true", help="Skip GPIO/jammer tests")
    args = parser.parse_args()

    tester = HardwareTester(skip_gpio=args.no_gpio, quick_mode=args.quick)
    success = tester.run_all()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
