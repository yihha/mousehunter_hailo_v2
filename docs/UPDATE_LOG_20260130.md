# Update Log - January 30, 2026

## V3 Model Architecture: INT8 Quantization Optimization

### Summary
Major refactoring to address INT8 quantization accuracy loss on Hailo-8L NPU. Simplified model from 4 classes to 2 classes (cat/rodent), reduced reg_max from 16 to 8, and implemented `force_range_out` fix to prevent quantization collapse.

---

## Root Cause Analysis

### Problem
Despite good GPU training metrics (mAP50 ~0.84 for rodent), the Hailo-deployed model showed severe accuracy degradation with many false negatives and poor localization.

### Key Findings

#### 1. INT8 Quantization Kills DFL Precision
- **DFL (Distribution Focal Loss)** uses 4 coordinates × 16 bins = 64 channels
- GPU training uses FP32 (23-bit mantissa precision)
- Hailo uses INT8 (only 256 discrete levels)
- Softmax probabilities in INT8 lose fine-grained distinctions
- Result: Bounding boxes become imprecise or completely wrong

#### 2. "Nullified Node" Pathology
- With only 4 classes, classification output dynamic range is narrow
- INT8 quantization can collapse this to near-zero range
- Hailo DFC reports: `INFO: output_range_out was nullified due to range constraints`
- Result: All class scores become ~0, no detections output

#### 3. NoIR Camera Spectral Domain Shift
- NoIR camera lacks IR-cut filter, causing pink/magenta tint
- Training data from normal cameras has different color distribution
- Calibration images must match deployment spectral characteristics
- Insufficient calibration images (had ~50, need 500-1024)

---

## Solutions Implemented

### 1. Simplified to 2 Classes (cat + rodent)
**Rationale**: Fewer classes = more quantization bits per class = better precision

**Before (v2)**:
```python
classes = {"0": "bird", "1": "cat", "2": "leaf", "3": "rodent"}
```

**After (v3)**:
```python
classes = {"0": "cat", "1": "rodent"}
```

**Files changed**: config.json, config.py, hailo_engine.py, prey_detector.py, detection.py, all test files

### 2. Reduced reg_max from 16 to 8
**Rationale**: Fewer DFL bins = less precision loss during INT8 quantization

- **Before**: 4 coords × 16 bins = 64 box channels
- **After**: 4 coords × 8 bins = 32 box channels

**Files changed**:
- config.json: Added `"reg_max": 8`
- config.py: Added `reg_max` field with default=8
- hailo_engine.py:
  - Added `reg_max` parameter to constructor
  - Updated `_decode_dfl()` to use configurable reg_max
  - Updated `_normalize_tensor_format()` to handle 32-channel tensors

### 3. Force Range Out Fix (compile_hailo.py)
**Rationale**: Prevent quantization from collapsing output ranges

```python
# In hailo_build/compile_hailo.py
model_script_lines = [
    'normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])',
    'model_optimization_flavor(optimization_level=2)',
    f'quantization_param([{cls_names_str}], force_range_out=[0.0, 1.0])'  # CRITICAL FIX
]
```

This forces the classification output layers to maintain [0.0, 1.0] range during quantization, preventing nullified nodes.

---

## New Files Created

### 1. YOLOv8_CatPrey_Training_for_Hailo_v3.ipynb
Training notebook for 2-class model with reg_max=8:
- Filters dataset to cat/rodent only
- Custom model YAML with `reg_max: 8` in Detect layer
- Exports to ONNX with correct output layer names
- Located in: `C:\ML Projects\mousehunter_solutions_v1\hailo_build\`

### 2. extract_calibration_frames.py
Script to extract frames from NoIR camera footage for calibration:
- Processes video files or directories
- Resizes to 640x640 (model input size)
- Extracts evenly-spaced frames
- Target: 500-1024 images from actual deployment environment
- Located in: `C:\ML Projects\mousehunter_solutions_v1\hailo_build\`

---

## Files Modified

| File | Changes |
|------|---------|
| config/config.json | 2 classes, reg_max=8, updated thresholds |
| src/mousehunter/config.py | Added reg_max field |
| src/mousehunter/inference/hailo_engine.py | reg_max support, 32-channel tensors |
| src/mousehunter/inference/prey_detector.py | Removed bird from PREY_CLASSES |
| src/mousehunter/inference/detection.py | Removed bird from prey detection |
| test_hardware.py | Updated class mapping |
| tests/conftest.py | Updated fixtures for 2 classes |
| tests/test_hailo_engine.py | Updated test assertions |
| hailo_build/compile_hailo.py | force_range_out, 2 classes, reg_max=8 |

---

## Git Commits

1. `6ab12af` - Upgrade to v3 model: 2 classes (cat/rodent) with reg_max=8

---

## TODO: Next Steps

### Step 1: Extract Calibration Frames (CRITICAL)
```bash
cd "C:\ML Projects\mousehunter_solutions_v1\hailo_build"
python extract_calibration_frames.py --input /path/to/noir_videos/ --output calibration_images --count 1000
```
**Requirements**:
- Use NoIR camera footage from actual deployment location
- Include diverse lighting (day, night, artificial light)
- Include typical scenes (with and without cats)
- Target: 500-1024 images

### Step 2: Retrain Model with V3 Architecture
1. Upload `YOLOv8_CatPrey_Training_for_Hailo_v3.ipynb` to Google Colab
2. Use GPU runtime
3. Train with:
   - 2 classes only (cat + rodent)
   - reg_max=8 in Detect layer
   - Filter existing dataset to remove bird/leaf
4. Export to ONNX

### Step 3: Compile for Hailo
```bash
cd "C:\ML Projects\mousehunter_solutions_v1\hailo_build"

# Copy calibration images to expected location
cp -r calibration_images calib_set

# Run Hailo DFC compiler
docker run -it --rm -v "${PWD}:/work" hailo_dfc python compile_hailo.py
```

### Step 4: Deploy and Test
```bash
# Copy new .hef to Pi
scp yolov8n_catprey.hef pi@raspberrypi:/path/to/mousehunter/models/

# Test on Pi
python test_live_detection.py --confidence 0.5
```

---

## Technical Reference

### V3 Model Architecture
```
Input:  (640, 640, 3) - RGB image
Output: 6 tensors in HWC format
  conv_bbox_0: (80, 80, 32) - DFL bbox, scale 0, stride 8  (was 64 channels)
  conv_cls_0:  (80, 80, 2)  - class scores, scale 0        (was 4 channels)
  conv_bbox_1: (40, 40, 32) - DFL bbox, scale 1, stride 16
  conv_cls_1:  (40, 40, 2)  - class scores, scale 1
  conv_bbox_2: (20, 20, 32) - DFL bbox, scale 2, stride 32
  conv_cls_2:  (20, 20, 2)  - class scores, scale 2
```

### DFL Decoding (reg_max=8)
- 32 channels = 4 coordinates × 8 bins (was 64 = 4 × 16)
- Apply softmax over 8 bins per coordinate
- Compute expected value: `sum(softmax * [0,1,2,...,7])`
- Result: [left, top, right, bottom] distances in grid cell units

### Class Mapping (V3)
```python
{"0": "cat", "1": "rodent"}  # Simplified from 4 classes
```

### Thresholds (V3)
```python
{"cat": 0.55, "rodent": 0.45}  # No bird/leaf thresholds needed
```

---

## Why These Changes Matter

| Issue | Solution | Impact |
|-------|----------|--------|
| INT8 loses DFL precision | reg_max 16→8 | 50% fewer bins to quantize |
| Class scores nullified | force_range_out | Maintains [0,1] output range |
| 4 classes spread thin | 2 classes only | 2x more bits per class |
| Wrong calibration domain | NoIR camera frames | Matches deployment spectral profile |
| Too few calibration images | 500-1024 target | Better quantization calibration |

---

*Log created: January 30, 2026*
