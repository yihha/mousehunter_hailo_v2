#!/usr/bin/env python3
"""Test Hailo inference with cross-thread usage (like main app)."""

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

MODEL_PATH = Path("models/yolov8n.hef")


class HailoTester:
    """Test class that creates Hailo in main thread, uses from worker thread."""

    def __init__(self):
        print(f"[Main Thread {threading.current_thread().name}] Initializing Hailo...")
        self._hef = HEF(str(MODEL_PATH))

        self._input_vstream_infos = self._hef.get_input_vstream_infos()
        self._output_vstream_infos = self._hef.get_output_vstream_infos()

        self._input_name = self._input_vstream_infos[0].name
        self._input_shape = self._input_vstream_infos[0].shape

        # Create VDevice in main thread
        self._vdevice = VDevice()

        # Configure network in main thread
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()

        # Create vstream params in main thread
        self._input_params = InputVStreamParams.make(
            self._network_group, quantized=True, format_type=FormatType.UINT8
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        print(f"[Main Thread] Initialization complete.\n")

    def infer(self, frame: np.ndarray) -> dict:
        """Run inference (called from worker thread)."""
        thread_name = threading.current_thread().name

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
        del self._network_group
        del self._vdevice
        del self._hef


def worker_thread(tester: HailoTester, num_inferences: int):
    """Worker thread that runs inferences."""
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


def test_cross_thread():
    print("=== Test: Cross-Thread Hailo Usage ===")
    print("(Create in main thread, use from worker thread)\n")

    # Create Hailo in main thread
    tester = HailoTester()

    # Run inferences from a different thread
    thread = threading.Thread(target=worker_thread, args=(tester, 5), name="WorkerThread")
    thread.start()
    thread.join()

    tester.cleanup()
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_cross_thread()
