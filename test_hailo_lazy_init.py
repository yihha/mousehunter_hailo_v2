#!/usr/bin/env python3
"""Test Hailo inference with lazy initialization (init in worker thread)."""

import numpy as np
import threading
import time
from pathlib import Path

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

MODEL_PATH = Path("models/yolov8n_catprey.hef")


class HailoTesterLazyInit:
    """Test class that lazily initializes Hailo on first use (in calling thread)."""

    def __init__(self):
        print(f"[{threading.current_thread().name}] Creating HailoTester (lazy init - no Hailo resources yet)")
        self._hailo_initialized = False
        self._hef = None
        self._vdevice = None
        self._network_group = None
        self._input_params = None
        self._output_params = None
        self._input_name = None

    def _ensure_initialized(self):
        """Lazy init Hailo in the current thread."""
        if self._hailo_initialized:
            return

        thread_name = threading.current_thread().name
        print(f"[{thread_name}] Lazy-initializing Hailo...")

        self._hef = HEF(str(MODEL_PATH))

        input_vstream_infos = self._hef.get_input_vstream_infos()
        self._input_name = input_vstream_infos[0].name
        print(f"[{thread_name}] Input: {self._input_name}, shape={input_vstream_infos[0].shape}")

        # Create VDevice in THIS thread
        self._vdevice = VDevice()

        # Configure network in THIS thread
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()

        # Create vstream params in THIS thread
        self._input_params = InputVStreamParams.make(
            self._network_group, quantized=True, format_type=FormatType.UINT8
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        self._hailo_initialized = True
        print(f"[{thread_name}] Hailo initialization complete!")

    def infer(self, frame: np.ndarray) -> dict:
        """Run inference (lazy init happens on first call)."""
        thread_name = threading.current_thread().name

        # Lazy init in this thread
        self._ensure_initialized()

        # Add batch dimension
        if len(frame.shape) == 3:
            frame = np.expand_dims(frame, axis=0)

        # Force copy that owns data
        frame = np.array(frame, dtype=np.uint8, copy=True, order='C')

        print(f"[{thread_name}] Input: shape={frame.shape}, owns_data={frame.flags['OWNDATA']}")

        with self._network_group.activate(self._network_group_params):
            with InferVStreams(
                self._network_group, self._input_params, self._output_params
            ) as pipeline:
                input_dict = {self._input_name: frame}
                print(f"[{thread_name}] Calling infer...")
                return pipeline.infer(input_dict)

    def cleanup(self):
        if self._network_group:
            del self._network_group
        if self._vdevice:
            del self._vdevice
        if self._hef:
            del self._hef


def worker_thread(tester: HailoTesterLazyInit, num_inferences: int):
    """Worker thread that runs inferences (Hailo will init here)."""
    thread_name = threading.current_thread().name
    print(f"\n[{thread_name}] Starting worker thread")

    for i in range(num_inferences):
        print(f"\n[{thread_name}] --- Inference {i+1} ---")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        try:
            result = tester.infer(frame)
            print(f"[{thread_name}] SUCCESS! Output keys: {list(result.keys())}")
        except Exception as e:
            print(f"[{thread_name}] FAILED: {e}")
        time.sleep(0.1)

    print(f"\n[{thread_name}] Worker thread complete")


def test_lazy_init():
    print("=== Test: Lazy Init Hailo in Worker Thread ===")
    print("(Object created in main thread, Hailo init in worker thread)\n")

    # Create object in main thread (NO Hailo resources yet)
    tester = HailoTesterLazyInit()

    # Run inferences from worker thread (Hailo will init there)
    thread = threading.Thread(target=worker_thread, args=(tester, 5), name="WorkerThread")
    thread.start()
    thread.join()

    tester.cleanup()
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_lazy_init()
