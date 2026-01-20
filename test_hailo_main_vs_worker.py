#!/usr/bin/env python3
"""Test: Does Hailo work in main thread vs worker thread?"""

import numpy as np
import threading
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


def run_inference_in_current_thread():
    """Initialize AND run inference entirely in the current thread."""
    thread_name = threading.current_thread().name
    print(f"\n[{thread_name}] === Starting test ===")
    print(f"[{thread_name}] Initializing Hailo...")

    try:
        hef = HEF(str(MODEL_PATH))
        input_vstream_infos = hef.get_input_vstream_infos()
        input_name = input_vstream_infos[0].name

        vdevice = VDevice()
        configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = vdevice.configure(hef, configure_params)[0]
        network_group_params = network_group.create_params()

        input_params = InputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
        output_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)

        print(f"[{thread_name}] Initialization complete, running inference...")

        # Create test frame
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        frame = np.expand_dims(frame, axis=0)
        frame = np.array(frame, dtype=np.uint8, copy=True, order='C')

        with network_group.activate(network_group_params):
            with InferVStreams(network_group, input_params, output_params) as pipeline:
                result = pipeline.infer({input_name: frame})

        print(f"[{thread_name}] SUCCESS! Output keys: {list(result.keys())}")

        del network_group
        del vdevice
        del hef

    except Exception as e:
        print(f"[{thread_name}] FAILED: {e}")


def test_main_thread():
    """Test 1: Everything in main thread."""
    print("\n" + "=" * 60)
    print("TEST 1: Everything in MainThread")
    print("=" * 60)
    run_inference_in_current_thread()


def test_worker_thread():
    """Test 2: Everything in worker thread."""
    print("\n" + "=" * 60)
    print("TEST 2: Everything in WorkerThread")
    print("=" * 60)
    thread = threading.Thread(target=run_inference_in_current_thread, name="WorkerThread")
    thread.start()
    thread.join()


if __name__ == "__main__":
    print("Testing Hailo in Main Thread vs Worker Thread")
    print("Both tests init AND infer in the SAME thread\n")

    test_main_thread()
    test_worker_thread()

    print("\n" + "=" * 60)
    print("CONCLUSION:")
    print("If Test 1 succeeds and Test 2 fails, Hailo only works in main thread")
    print("=" * 60)
