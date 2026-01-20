#!/usr/bin/env python3
"""Minimal Hailo inference test to debug input format issues."""

import numpy as np
from pathlib import Path

# Import Hailo
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


def test_hailo_inference():
    print("=== Minimal Hailo Inference Test ===\n")

    # 1. Load HEF
    print(f"Loading model: {MODEL_PATH}")
    hef = HEF(str(MODEL_PATH))

    # 2. Get input/output info
    input_vstream_infos = hef.get_input_vstream_infos()
    output_vstream_infos = hef.get_output_vstream_infos()

    print("\nInput streams:")
    for info in input_vstream_infos:
        print(f"  Name: {info.name}")
        print(f"  Shape: {info.shape}")
        print(f"  Format: {info.format.type}")

    print("\nOutput streams:")
    for info in output_vstream_infos:
        print(f"  Name: {info.name}")
        print(f"  Shape: {info.shape}")

    input_name = input_vstream_infos[0].name
    input_shape = input_vstream_infos[0].shape
    print(f"\nUsing input: {input_name} with shape {input_shape}")

    # 3. Create virtual device
    print("\nCreating VDevice...")
    vdevice = VDevice()

    # 4. Configure network
    print("Configuring network...")
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_group = vdevice.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # 5. Create vstream params
    print("Creating vstream params...")
    input_params = InputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
    output_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)

    # 6. Create test input - try different formats
    print("\n--- Testing different input formats ---\n")

    # Format A: 3D array (H, W, C)
    input_3d = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    input_3d = np.ascontiguousarray(input_3d)
    print(f"3D input: shape={input_3d.shape}, dtype={input_3d.dtype}, nbytes={input_3d.nbytes}, contiguous={input_3d.flags['C_CONTIGUOUS']}")

    # Format B: 4D array (1, H, W, C)
    input_4d = np.expand_dims(input_3d, axis=0)
    input_4d = np.ascontiguousarray(input_4d)
    print(f"4D input: shape={input_4d.shape}, dtype={input_4d.dtype}, nbytes={input_4d.nbytes}, contiguous={input_4d.flags['C_CONTIGUOUS']}")

    # Try inference with each format
    with network_group.activate(network_group_params):
        with InferVStreams(network_group, input_params, output_params) as pipeline:

            # Try 3D first
            print("\n[Test 1] Trying 3D input (640, 640, 3)...")
            try:
                input_dict = {input_name: input_3d}
                print(f"  Input dict keys: {list(input_dict.keys())}")
                print(f"  Input data shape: {input_dict[input_name].shape}")
                result = pipeline.infer(input_dict)
                print(f"  SUCCESS! Output keys: {list(result.keys())}")
                for k, v in result.items():
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            except Exception as e:
                print(f"  FAILED: {e}")

            # Try 4D
            print("\n[Test 2] Trying 4D input (1, 640, 640, 3)...")
            try:
                input_dict = {input_name: input_4d}
                print(f"  Input dict keys: {list(input_dict.keys())}")
                print(f"  Input data shape: {input_dict[input_name].shape}")
                result = pipeline.infer(input_dict)
                print(f"  SUCCESS! Output keys: {list(result.keys())}")
                for k, v in result.items():
                    print(f"    {k}: type={type(v)}")
                    if isinstance(v, list):
                        print(f"      List length: {len(v)}")
                        if len(v) > 0:
                            first = v[0]
                            print(f"      First element type: {type(first)}")
                            # Convert nested list to numpy and show shape
                            arr = np.array(first) if isinstance(first, list) else first
                            if hasattr(arr, 'shape'):
                                print(f"      As numpy array: shape={arr.shape}, dtype={arr.dtype}")
                    elif hasattr(v, 'shape'):
                        print(f"      shape={v.shape}, dtype={v.dtype}")
            except Exception as e:
                import traceback
                print(f"  FAILED: {e}")
                traceback.print_exc()

            # Try with explicit batch_size parameter
            print("\n[Test 3] Trying 3D input with batch_size=1...")
            try:
                input_dict = {input_name: input_3d}
                result = pipeline.infer(input_dict, batch_size=1)
                print(f"  SUCCESS! Output keys: {list(result.keys())}")
            except Exception as e:
                print(f"  FAILED: {e}")

    print("\n=== Test Complete ===")

    # Cleanup
    del network_group
    del vdevice
    del hef


if __name__ == "__main__":
    test_hailo_inference()
