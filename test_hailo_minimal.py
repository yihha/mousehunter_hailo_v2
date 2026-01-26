#!/usr/bin/env python3
"""Absolute minimal Hailo test - copy of official example pattern."""

import numpy as np
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InferVStreams, InputVStreamParams, OutputVStreamParams, FormatType

# Load model
hef = HEF("models/yolov8n_catprey.hef")
print(f"Loaded HEF: {hef.get_input_vstream_infos()[0].name}, shape={hef.get_input_vstream_infos()[0].shape}")

# Setup device
vdevice = VDevice()
configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
network_group = vdevice.configure(hef, configure_params)[0]

# Get params
input_vstreams_params = InputVStreamParams.make(network_group, quantized=True, format_type=FormatType.UINT8)
output_vstreams_params = OutputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)
network_group_params = network_group.create_params()

# Get input name
input_vstream_info = hef.get_input_vstream_infos()[0]
input_name = input_vstream_info.name

# Create test data - 4D with batch dimension
test_input = np.random.randint(0, 255, (1, 640, 640, 3), dtype=np.uint8)
test_input = np.ascontiguousarray(test_input)
print(f"Input: shape={test_input.shape}, dtype={test_input.dtype}, contiguous={test_input.flags['C_CONTIGUOUS']}")

# Run inference
print("Running inference...")
with network_group.activate(network_group_params):
    with InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
        input_dict = {input_name: test_input}
        print(f"Input dict: key={input_name}, value.shape={input_dict[input_name].shape}")
        results = infer_pipeline.infer(input_dict)
        print(f"SUCCESS! Results: {type(results)}, keys={list(results.keys()) if isinstance(results, dict) else 'N/A'}")

print("Done!")
