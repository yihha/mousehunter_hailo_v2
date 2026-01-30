# Update Log - January 28, 2026

> **⚠️ HISTORICAL DOCUMENT (V2 Model)**
>
> This log documents fixes for the **V2 model** which had 4 classes (bird/cat/leaf/rodent) and reg_max=16.
> The current **V3 model** uses only 2 classes (cat/rodent) with reg_max=8.
> See `UPDATE_LOG_20260130.md` for V3 model changes.

## Custom YOLOv8n Model Post-Processing Fix

### Summary
Fixed the post-processing pipeline for the custom-trained YOLOv8n model (cat/bird/leaf/rodent detection) running on Hailo-8L NPU. The bounding boxes are now correctly positioned on detected objects.

---

## Issues Identified

### 1. Bounding Box Coordinates Were Completely Wrong
- **Symptom**: Boxes were huge (covering entire frame) or positioned in wrong locations
- **Root Cause**: Multiple issues in the post-processing code

### 2. Tensor Format Mismatch
- **Expected**: NCHW format `[1, 64, 80, 80]` (PyTorch/ONNX standard)
- **Actual from Hailo**: HWC format `[80, 80, 64]`
- **Impact**: Code was misinterpreting tensor dimensions

### 3. Tensor Name Matching Failed
- **Expected names**: `bbox_scale0`, `cls_scale0`, etc.
- **Actual names from Hailo**: `yolov8n_catprey/conv41`, `conv42`, etc.
- **Impact**: Tensors weren't being matched to correct scales

### 4. DFL Coordinate Calculation Was Wrong
- **Original code**:
  ```python
  cx = (x + 0.5 - decoded[0] + decoded[2]) / 2 * stride / 640.0  # WRONG
  ```
- **Problem**: Mixed up the DFL distance interpretation

---

## Diagnostic Process

### Step 1: Analyzed Training Output (from Colab)
```
Model outputs 6 tensors:
  bbox_scale0: [1, 64, 80, 80]  - DFL box regression
  cls_scale0:  [1, 4, 80, 80]   - Class scores
  bbox_scale1: [1, 64, 40, 40]
  cls_scale1:  [1, 4, 40, 40]
  bbox_scale2: [1, 64, 20, 20]
  cls_scale2:  [1, 4, 20, 20]
```

### Step 2: Created Debug Script to Inspect Hailo Output
```python
# debug_tensors.py - revealed actual tensor format
Input shape: (640, 640, 3)
Outputs:
  yolov8n_catprey/conv41: shape=(80, 80, 64), format=UINT8
  yolov8n_catprey/conv42: shape=(80, 80, 4), format=UINT8
  ...
```

**Key Finding**: Hailo returns HWC format, not NCHW!

### Step 3: Added DFL Debug Output
```python
print(f"[DEBUG] DFL at grid({x},{y}): decoded={decoded}, score={max_score:.2f}")
print(f"[DEBUG]   center=({cx_pixels:.0f},{cy_pixels:.0f}), dists=L:{left_dist:.0f}...")
print(f"[DEBUG]   box=({x1_pixels:.0f},{y1_pixels:.0f})-({x2_pixels:.0f},{y2_pixels:.0f})")
```

**Verified**: Math was now correct, boxes positioned properly on actual cats.

---

## Fixes Implemented

### Fix 1: Tensor Format Normalization
Added `_normalize_tensor_format()` method to handle both NCHW and HWC:
```python
def _normalize_tensor_format(self, tensor, num_classes):
    if tensor.ndim == 4:  # NCHW [batch, C, H, W]
        tensor = tensor[0]
        tensor = np.transpose(tensor, (1, 2, 0))  # -> HWC
    elif tensor.ndim == 3:
        # Detect CHW vs HWC by checking dimensions
        if d0 in (64, num_classes) and d1 == d2:
            return np.transpose(tensor, (1, 2, 0))  # CHW -> HWC
        elif d2 in (64, num_classes) and d0 == d1:
            return tensor  # Already HWC
    return tensor
```

### Fix 2: Scale-Based Tensor Matching
Added `_get_scale_from_name_or_size()` to match tensors by spatial size:
```python
def _get_scale_from_name_or_size(self, name, spatial_size):
    # Try name pattern first (scale0, scale1, scale2)
    match = re.search(r'scale[_]?(\d+)', name)
    if match:
        return int(match.group(1))
    # Fall back to spatial size
    size_to_scale = {80: 0, 40: 1, 20: 2}
    return size_to_scale.get(spatial_size, 0)
```

### Fix 3: Corrected DFL Coordinate Calculation
```python
# DFL decoded values are distances from cell center (in grid cell units)
left_dist = decoded[0] * stride   # pixels
top_dist = decoded[1] * stride
right_dist = decoded[2] * stride
bottom_dist = decoded[3] * stride

# Box edges
x1_pixels = cx_pixels - left_dist
y1_pixels = cy_pixels - top_dist
x2_pixels = cx_pixels + right_dist
y2_pixels = cy_pixels + bottom_dist
```

### Fix 4: Added Camera Flip Options
For cameras mounted upside-down or rotated:
```bash
python test_live_detection.py --vflip        # Vertical flip
python test_live_detection.py --hflip        # Horizontal flip
python test_live_detection.py --flip         # Both (180° rotation)
```

---

## Validation Results

### Model Performance (from training)
| Class  | Precision | Recall | mAP50  |
|--------|-----------|--------|--------|
| cat    | 0.949     | 0.978  | 0.991  |
| rodent | 0.834     | 0.758  | 0.841  |
| bird   | 0.746     | 0.268  | 0.342  |
| leaf   | -         | -      | -      |

### Live Detection Results (after fix)
- Bounding boxes now correctly positioned on detected cats
- True positive detections have accurate box placement
- False positives remain (model quality issue, not code)

---

## Known Remaining Issues (Model Quality)

| Issue | Cause | Mitigation |
|-------|-------|------------|
| Pink couch detected as "cat" | Training data bias | Raise threshold to 0.6+ |
| Computer mouse detected as "bird" | Only 11 bird training images | Retrain with more data |
| Some boxes larger than needed | Model DFL output characteristics | Model retraining |

These are **model training issues**, not post-processing bugs.

---

## Files Changed

1. `src/mousehunter/inference/hailo_engine.py`
   - Added `_normalize_tensor_format()` method
   - Added `_get_scale_from_name_or_size()` method
   - Rewrote `_postprocess_yolo_raw()` with correct tensor handling
   - Fixed DFL coordinate calculation

2. `test_live_detection.py`
   - Added `--vflip`, `--hflip`, `--flip` command line options
   - Uses libcamera Transform for hardware-accelerated flipping

3. `debug_tensors.py` (new, for diagnostics)
   - Script to inspect raw Hailo tensor output

---

## Git Commits

1. `48d57f6` - Fix YOLOv8 post-processing for Hailo tensor format
2. `17f8a3a` - Add camera flip options for upside-down mounted camera
3. `7abe0e0` - Add debug output for DFL decoding diagnostics
4. `1e7302f` - Comment out debug prints for production use

---

## Recommendations

### Immediate
```bash
# Use higher confidence threshold to reduce false positives
python test_live_detection.py --confidence 0.65
```

### Future Model Improvements
1. Add negative examples (furniture without cats)
2. Increase bird training images (currently only 11)
3. Add more diverse backgrounds and distances
4. Consider data augmentation for better generalization

---

## Technical Details

### Hailo Tensor Output Format
```
Input:  (640, 640, 3) - RGB image
Output: 6 tensors in HWC format
  conv41: (80, 80, 64) - DFL bbox, scale 0, stride 8
  conv42: (80, 80, 4)  - class scores, scale 0
  conv52: (40, 40, 64) - DFL bbox, scale 1, stride 16
  conv53: (40, 40, 4)  - class scores, scale 1
  conv62: (20, 20, 64) - DFL bbox, scale 2, stride 32
  conv63: (20, 20, 4)  - class scores, scale 2
```

### DFL (Distribution Focal Loss) Decoding
- 64 channels = 4 coordinates × 16 bins
- Apply softmax over 16 bins per coordinate
- Compute expected value: `sum(softmax * [0,1,2,...,15])`
- Result: [left, top, right, bottom] distances in grid cell units
- Multiply by stride to get pixel distances

### Class Mapping
```python
{0: "bird", 1: "cat", 2: "leaf", 3: "rodent"}
```

---

*Log created: January 28, 2026*
