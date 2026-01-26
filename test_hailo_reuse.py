#!/usr/bin/env python3
"""Test Hailo inference with reused VDevice/network_group (like main app)."""

import numpy as np
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


class HailoTester:
    """Test class that reuses VDevice like the main app."""

    def __init__(self):
        print("Initializing Hailo...")
        self._hef = HEF(str(MODEL_PATH))

        self._input_vstream_infos = self._hef.get_input_vstream_infos()
        self._output_vstream_infos = self._hef.get_output_vstream_infos()

        print(f"Input: {self._input_vstream_infos[0].name}, shape={self._input_vstream_infos[0].shape}")
        print(f"Output: {self._output_vstream_infos[0].name}, shape={self._output_vstream_infos[0].shape}")

        self._input_name = self._input_vstream_infos[0].name
        self._input_shape = self._input_vstream_infos[0].shape

        # Create VDevice (reused)
        self._vdevice = VDevice()

        # Configure network (reused)
        configure_params = ConfigureParams.create_from_hef(
            self._hef, interface=HailoStreamInterface.PCIe
        )
        self._network_group = self._vdevice.configure(self._hef, configure_params)[0]
        self._network_group_params = self._network_group.create_params()

        # Create vstream params (reused)
        self._input_params = InputVStreamParams.make(
            self._network_group, quantized=True, format_type=FormatType.UINT8
        )
        self._output_params = OutputVStreamParams.make(
            self._network_group, quantized=False, format_type=FormatType.FLOAT32
        )

        print("Initialization complete.\n")

    def infer(self, frame: np.ndarray) -> dict:
        """Run inference on a single frame (like main app does)."""
        # Add batch dimension
        if len(frame.shape) == 3:
            frame = np.expand_dims(frame, axis=0)

        # Force copy that owns data
        frame = np.array(frame, dtype=np.uint8, copy=True, order='C')

        print(f"Input: shape={frame.shape}, dtype={frame.dtype}, "
              f"nbytes={frame.nbytes}, contiguous={frame.flags['C_CONTIGUOUS']}, "
              f"owns_data={frame.flags['OWNDATA']}")

        # This is how main app does it - create context each time
        with self._network_group.activate(self._network_group_params):
            with InferVStreams(
                self._network_group, self._input_params, self._output_params
            ) as pipeline:
                input_dict = {self._input_name: frame}
                print(f"Calling infer with key: {self._input_name}")
                return pipeline.infer(input_dict)

    def cleanup(self):
        del self._network_group
        del self._vdevice
        del self._hef


def test_multiple_inferences():
    print("=== Test: Multiple Inferences with Reused Context ===\n")

    tester = HailoTester()

    # Run multiple inferences like the main app would
    for i in range(5):
        print(f"\n--- Inference {i+1} ---")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        try:
            result = tester.infer(frame)
            print(f"SUCCESS! Output keys: {list(result.keys())}")
        except Exception as e:
            print(f"FAILED: {e}")

    tester.cleanup()
    print("\n=== Test Complete ===")


if __name__ == "__main__":
    test_multiple_inferences()
