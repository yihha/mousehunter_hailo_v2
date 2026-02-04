# Update Log - February 1, 2026

## Session Summary: ONNX Export for Hailo with PyTorch 2.9

### Objective
Export trained YOLOv8n model (2 classes, reg_max=8) to ONNX format with 6 separate output tensors for Hailo-8L compilation.

---

## Training Results (Successful)

Training completed successfully on Google Colab with PyTorch 2.9:

```
Model: YOLOv8n with PatchedDetect (reg_max=8)
Dataset: cat_prey_training_v3 (pre-filtered 2 classes)
Classes: cat (0), rodent (1)

Results:
  mAP50: 0.907
  cat mAP50: 0.978
  rodent mAP50: 0.796

Weights saved: 5.8 MB (best.pt)
Location: runs/detect/runs/detect/catprey_v4_reg8/weights/best.pt
```

---

## ONNX Export Problem

### The Issue
PyTorch 2.9 introduced a new `torch.export`-based ONNX exporter that breaks our custom wrapper approach.

### What We Need
6 separate output tensors for `hailo_engine.py`:
- `bbox_scale0`: [1, 32, 80, 80] - DFL bbox, stride 8
- `cls_scale0`: [1, 2, 80, 80] - class scores
- `bbox_scale1`: [1, 32, 40, 40] - DFL bbox, stride 16
- `cls_scale1`: [1, 2, 40, 40] - class scores
- `bbox_scale2`: [1, 32, 20, 20] - DFL bbox, stride 32
- `cls_scale2`: [1, 2, 20, 20] - class scores

### What Ultralytics Produces
Single combined decoded output:
- `output0`: [1, 6, 8400] - already decoded boxes (4) + class scores (2)

This defeats the purpose of reg_max=8 since DFL decoding already happened.

---

## Approaches Tried

### 1. Custom YOLOv8ForHailo Wrapper (FAILED)
```python
class YOLOv8ForHailo(nn.Module):
    def forward(self, x):
        # Extract raw cv2/cv3 outputs
        for i, feat in enumerate(features):
            bbox = self.cv2[i](feat)
            cls = self.cv3[i](feat)
            outputs.append(bbox)
            outputs.append(cls)
        return tuple(outputs)
```
**Result**: PyTorch works (correct 6 outputs), but ONNX export produces 0.37 MB file (missing weights).

**Root Cause**: PyTorch 2.9's new exporter doesn't properly trace custom wrappers.

### 2. JIT Trace Before Export (FAILED)
```python
traced_model = torch.jit.trace(hailo_model_cpu, dummy_input)
torch.onnx.export(traced_model, ...)
```
**Error**: `TorchExportError: Exporting a ScriptModule is not supported`

PyTorch 2.9's new exporter explicitly rejects JIT traced models.

### 3. Legacy Exporter Environment Variable (FAILED)
```python
os.environ["PYTORCH_ONNX_USE_LEGACY"] = "1"
torch.onnx.export(model, ...)
```
**Result**: Environment variable ignored, still uses new exporter, still 0.37 MB.

### 4. Patch Detect.forward() In-Place (FAILED)
```python
def raw_output_forward(self, x):
    outputs = []
    for i in range(len(self.cv2)):
        bbox = self.cv2[i](x[i])
        cls = self.cv3[i](x[i])
        outputs.append(bbox)
        outputs.append(cls)
    return tuple(outputs)

detect.__class__.forward = raw_output_forward
```
**Result**: PyTorch inference correct (6 outputs with right shapes), but ONNX still 0.37 MB.

### 5. ONNX Graph Surgery (CURRENT APPROACH - IN PROGRESS)
Strategy:
1. Use Ultralytics export (works, 10.8 MB with all weights)
2. Run ONNX shape inference to find intermediate tensor shapes
3. Find cv2 (bbox) and cv3 (cls) final conv outputs
4. Modify graph to expose these as outputs instead of combined decoded output

**Status**: Shape inference finds 267 tensors, but exact name matching for cv2/cv3 outputs is failing.

---

## Key Findings

### 1. Ultralytics ONNX Export Works
```
Size: 10.83 MB (correct)
Output: [1, 6, 8400] (combined decoded format)
```
All weights are present. The graph contains our target tensors.

### 2. Target Tensors Exist in Graph
From earlier run that found tensors:
```
/model.22/cv2.0/cv2.0.2/Conv_output_0: [1, 32, 80, 80]  <- bbox scale 0
/model.22/cv2.1/cv2.1.2/Conv_output_0: [1, 32, 40, 40]  <- bbox scale 1
/model.22/cv2.2/cv2.2.2/Conv_output_0: [1, 32, 20, 20]  <- bbox scale 2
/model.22/cv3.0/cv3.0.2/Conv_output_0: [1, 2, 80, 80]   <- cls scale 0
/model.22/cv3.1/cv3.1.2/Conv_output_0: [1, 2, 40, 40]   <- cls scale 1
/model.22/cv3.2/cv3.2.2/Conv_output_0: [1, 2, 20, 20]   <- cls scale 2
```

### 3. Model Has Correct reg_max=8
```
Classes: 2, reg_max: 8
Box channels: 32 (4 * 8)
```
The trained model has the right architecture.

---

## Current State

### What's Working
- Training with reg_max=8: SUCCESS
- Model validation: mAP50 = 0.907
- Ultralytics ONNX export: 10.83 MB (weights correct)
- Shape inference: Finds 267 tensors

### What's Not Working
- Exact tensor name matching in graph surgery
- Need to debug why `/model.22/cv2.0/cv2.0.2/Conv_output_0` isn't being found

### Next Steps (Tomorrow)
1. Run cell-24 with fuzzy matching to see actual tensor names
2. Debug tensor name format in the ONNX graph
3. Complete graph surgery to expose 6 raw outputs
4. Verify with ONNX Runtime
5. Test with Hailo DFC compilation

---

## Files Modified Today

| File | Status | Notes |
|------|--------|-------|
| `hailo_build/YOLOv8_CatPrey_Training_for_Hailo_v4.ipynb` | Updated | Cell-24 with graph surgery approach |
| `docs/UPDATE_LOG_20260201.md` | Created | This file |

---

## Technical Reference

### Expected ONNX Output Structure for Hailo
```
Input:  images [1, 3, 640, 640]
Outputs (6):
  bbox_scale0: [1, 32, 80, 80]  # 4 coords × 8 bins, stride 8
  cls_scale0:  [1, 2, 80, 80]   # 2 classes
  bbox_scale1: [1, 32, 40, 40]  # stride 16
  cls_scale1:  [1, 2, 40, 40]
  bbox_scale2: [1, 32, 20, 20]  # stride 32
  cls_scale2:  [1, 2, 20, 20]
```

### hailo_engine.py Expectations
- Identifies tensors by shape (spatial dimensions 80/40/20)
- Identifies bbox vs cls by channel count (32 vs 2)
- Applies DFL decoding with configurable reg_max

### Notebook Location
```
C:\ML Projects\mousehunter_solutions_v1\hailo_build\YOLOv8_CatPrey_Training_for_Hailo_v4.ipynb
```

---

## Session 2: ONNX Graph Surgery Fix (SUCCESS)

### Problem Resolved
The graph surgery in cell-24 was failing to find tensors despite them being present. Root cause unclear (possibly Colab state issue), but diagnostic confirmed all 6 tensors exist with expected names.

### Solution Applied
1. **Ran diagnostic script** to dump all ONNX tensor names
2. **Confirmed tensor names match** what cell-24 expected:
   ```
   /model.22/cv2.0/cv2.0.2/Conv_output_0: [1, 32, 80, 80]  ← bbox scale 0
   /model.22/cv3.0/cv3.0.2/Conv_output_0: [1, 2, 80, 80]   ← cls scale 0
   ...etc
   ```
3. **Created working ONNX** using Option B (direct graph surgery on existing file)

### Final ONNX File
```
File: yolov8n_catrodent_reg8_6outputs_20260201-104632.onnx
Size: 10.84 MB
Location: Google Drive/Colab Notebooks/cat_prey/yolo_v8n_models_v4/
```

**Outputs (verified):**
| Output | Shape | Purpose |
|--------|-------|---------|
| /model.22/cv2.0/cv2.0.2/Conv_output_0 | [1, 32, 80, 80] | bbox scale 0 |
| /model.22/cv3.0/cv3.0.2/Conv_output_0 | [1, 2, 80, 80] | cls scale 0 |
| /model.22/cv2.1/cv2.1.2/Conv_output_0 | [1, 32, 40, 40] | bbox scale 1 |
| /model.22/cv3.1/cv3.1.2/Conv_output_0 | [1, 2, 40, 40] | cls scale 1 |
| /model.22/cv2.2/cv2.2.2/Conv_output_0 | [1, 32, 20, 20] | bbox scale 2 |
| /model.22/cv3.2/cv3.2.2/Conv_output_0 | [1, 2, 20, 20] | cls scale 2 |

### Notebook Updated
Updated `YOLOv8_CatPrey_Training_for_Hailo_v4.ipynb` cell-24 with more robust tensor finding:
- **Strategy A**: Exact name matching (fastest)
- **Strategy B**: Pattern-based fuzzy matching
- **Strategy C**: Shape-based fallback (most robust)

This ensures future training runs won't fail due to tensor naming issues.

### Next Steps
1. Download ONNX from Google Drive
2. Prepare calibration images (extract_calibration_frames.py)
3. Run Hailo DFC compilation (compile_hailo.py)
4. Deploy .hef to Raspberry Pi

---

## Session 3: Hailo Compilation (SUCCESS)

### Calibration Data Prepared
Extracted frames from video footage for calibration:
```
Total calibration images: 567
Location: hailo_build/calib_imgs/
Source: Video from deployment location (non-NoIR camera)
```

**Note on camera mismatch**: Calibration images were from a standard camera, but deployment uses NoIR camera with IR illuminator. Since IR illumination produces grayscale images, the color profile difference should be minimal. Monitor for any quantization issues in night conditions.

### Compilation Process

#### Issue Encountered: force_range_out Layer Naming
The `force_range_out` quantization parameter failed because Hailo's internal layer naming doesn't match ONNX output names:
- **ONNX name**: `/model.22/cv3.0/cv3.0.2/Conv_output_0`
- **Tried**: `yolov8n_catprey/model.22/cv3.0/cv3.0.2/Conv`
- **Error**: `Layer not found in model`

**Solution**: Disabled `force_range_out` and compiled without it. Compilation succeeded with good quantization metrics.

#### Compilation Results
```
Hailo DFC: hailo_dfc Docker container
Target: Hailo-8L (hailo8l)
Compilation time: ~25 minutes total
```

**Quantization Quality (SNR - higher is better):**
| Output Layer | SNR | Status |
|--------------|-----|--------|
| output_layer1 (bbox scale 0) | 27.85 dB | ✓ Good |
| output_layer2 (cls scale 0) | 36.53 dB | ✓ Excellent |
| output_layer3 (bbox scale 1) | 27.08 dB | ✓ Good |
| output_layer4 (cls scale 1) | 34.87 dB | ✓ Excellent |
| output_layer5 (bbox scale 2) | 28.34 dB | ✓ Good |
| output_layer6 (cls scale 2) | 32.19 dB | ✓ Good |

All outputs >25 dB indicates acceptable quantization noise levels.

**Distillation Loss (lower is better):**
- All layers: 0.03-0.08 range
- Total: ~0.49
- Status: ✓ Good accuracy retention

**Model Partitioning:**
- Contexts: 3 (normal for YOLOv8 on Hailo-8L)
- No nullified nodes warnings
- Successful mapping to all clusters

### Final HEF File
```
File: yolov8n_catprey.hef
Location: hailo_build/
Input: 640x640 RGB (uint8)
Outputs: 6 raw detection tensors
```

### Important Notes

1. **force_range_out was NOT applied** - but compilation succeeded without nullified nodes. Monitor detection quality; if classification outputs are weak, investigate correct Hailo layer names.

2. **To find correct Hailo layer names** (if needed later):
   ```python
   # Add after runner.translate_onnx_model():
   print(runner.get_hn_dict())
   ```

3. **Hailo internal layer naming** differs from ONNX:
   - Hailo shows end nodes as: `'/model.22/cv3.0/cv3.0.2/Conv'`
   - The exact internal scope format is unclear

4. **Config files are consistent** - all set for V3 model:
   - `config.json`: classes = {"0": "cat", "1": "rodent"}, reg_max = 8
   - `hailo_engine.py`: defaults match config
   - `prey_detector.py`: PREY_CLASSES = {"rodent"}

---

## Next Steps: Testing on Raspberry Pi

### 1. Copy HEF to Raspberry Pi
```bash
scp hailo_build/yolov8n_catprey.hef pi@<PI_IP>:/path/to/mousehunter_hailo_v2/models/
```

### 2. Quick Sanity Test
```bash
# On Raspberry Pi
cd /path/to/mousehunter_hailo_v2
python test_live_detection.py --confidence 0.5
```

### 3. What to Watch For

**Good signs:**
- Detections with confidence 0.4-0.9
- Bounding boxes correctly positioned on cats/rodents
- No tensor shape errors

**Bad signs (indicate quantization issues):**
- All confidences near 0 → classification outputs collapsed
- All confidences near 1.0 → sigmoid/normalization issue
- Boxes in wrong positions → DFL decoding or stride issue

### 4. If Detection Quality is Poor
1. Check if outputs are all zeros (nullified)
2. Re-enable `force_range_out` with correct layer names
3. Consider adding more calibration images from NoIR camera

---

## Files Modified Today

| File | Status | Notes |
|------|--------|-------|
| `hailo_build/YOLOv8_CatPrey_Training_for_Hailo_v4.ipynb` | Updated | Robust 3-strategy tensor finding |
| `hailo_build/compile_hailo.py` | Updated | Disabled force_range_out |
| `hailo_build/yolov8n_catprey.hef` | Created | Final compiled model |
| `scripts/dump_onnx_tensors.py` | Created | Diagnostic for ONNX tensor names |
| `docs/UPDATE_LOG_20260201.md` | Updated | This file |

---

## V3 Model Summary

| Property | Value |
|----------|-------|
| Architecture | YOLOv8n |
| Classes | 2 (cat=0, rodent=1) |
| reg_max | 8 |
| Training mAP50 | 0.907 |
| Input size | 640x640 RGB |
| Output format | 6 raw tensors (DFL bbox + class scores) |
| Target hardware | Hailo-8L |
| Quantization | INT8, optimization level 2 |

---

## Session 4: System Testing & Score Accumulation Implementation

### Testing on Raspberry Pi

#### Hardware Test Results
All 14 hardware tests passed:
- Hailo-8L NPU: Connected and working
- Model loading: 6 tensors correctly identified
- Camera (PiCamera 3 NoIR): Dual stream configured (1080p main, 640x640 inference)
- GPIO/Jammer: Working on GPIO 17
- Inference speed: ~70ms average (14.2 FPS on NPU)

#### Video Inference Test
Tested with sample video at different thresholds:
```
Threshold 0.2: cat=108, rodent=15 detections
Threshold 0.3: cat=106, rodent=8 detections (missed some rodents)
```
Confidence distribution healthy: 0.22 - 0.91 range.

### New Feature: Time-Based Score Accumulation

Replaced frame-based sliding window (3-of-5) with score accumulation approach:

**Problem with old approach:**
- Frame-based counting missed sporadic detections
- Gaps between detections exceeded window size

**New approach (based on radar tracking best practices):**
```
Mode: score_accumulation
Window: 3.0 seconds
Score threshold: 0.9
Min detection score: 0.20
Cat lost reset: 1.5 seconds
```

**How it works:**
1. Cat detected → State: IDLE → MONITORING
2. Each rodent detection adds its confidence to accumulated score
3. Score ≥ 0.9 within window → State: CONFIRMED
4. No cat for 1.5s → Reset to IDLE

### Bug Fixes Applied

| Issue | File | Fix |
|-------|------|-----|
| Version said "v2" | `__init__.py`, `main.py` | Updated to v3.0.0 |
| Default thresholds mismatch | `config.py` | Aligned with config.json |
| Tests failing with new mode | `conftest.py` | Added `prey_confirmation_mode="frame_count"` for legacy tests |
| `_update_state` method renamed | `test_prey_detector.py` | Changed to `_update_state_frame_count` |
| Verbose debug logging | `hailo_engine.py` | Changed `logger.info` to `logger.debug` |
| `PreyDetectionEvent` attribute error | `main.py` | Updated to use new attributes |

### Live Detection Test Results

System successfully detected and triggered lockdown:
```
State: IDLE -> MONITORING (cat detected)
State: MONITORING -> VERIFYING (prey detected)
State: VERIFYING -> CONFIRMED
PREY CONFIRMED! score=1.00/0.9, detections=2 in 3.0s window, prey=rodent (0.45)
PREY DETECTED! rodent (0.45) with cat (0.60), score: 1.00/0.9 (2 detections in 3.0s)
JAMMER ACTIVATED - Cat flap BLOCKED
```

**Note:** This was a FALSE POSITIVE - no actual rodent present. Environment differs from training data.

### Configuration Recommendations for Deployment

Current settings may be too sensitive. Consider adjusting after real-world testing:

```json
"thresholds": {
    "cat": 0.50,
    "rodent": 0.30    // Raise from 0.20 to reduce false positives
},
"prey_confirmation": {
    "min_detection_score": 0.30  // Match rodent threshold
}
```

### Files Modified

| File | Changes |
|------|---------|
| `src/mousehunter/__init__.py` | Version → 3.0.0 |
| `src/mousehunter/config.py` | Added score accumulation config, aligned thresholds |
| `src/mousehunter/main.py` | Version string, fixed event attributes |
| `src/mousehunter/inference/hailo_engine.py` | Debug logging to DEBUG level |
| `src/mousehunter/inference/prey_detector.py` | Complete rewrite with score accumulation |
| `config/config.json` | Added prey_confirmation section |
| `tests/conftest.py` | New fixtures for score accumulation mode |
| `tests/test_prey_detector.py` | Fixed tests, added TestScoreAccumulation class |
| `test_video_inference.py` | Created - video testing with annotations |

### Git Commits

```
65ade40 Fix PreyDetectionEvent attribute names in main.py callback
230ec4f Change verbose frame debug logging from INFO to DEBUG level
50d3df9 Update version to v3, align config defaults, fix tests for score accumulation
2212b0b Implement time-based score accumulation for prey confirmation
9ddf816 Add video inference test and debug diagnostics for tensor matching
```

---

## Deployment Checklist

### Before Deployment
- [x] All 64 unit tests passing
- [x] All Python files compile without errors
- [x] Version updated to 3.0.0
- [x] Configuration consistent across all files
- [x] Debug logging reduced to DEBUG level
- [x] State machine transitions verified

### Deployment Steps
1. Pull latest code: `git pull`
2. Verify config: `cat config/config.json | grep -A5 classes`
3. Run hardware test: `python test_hardware.py`
4. Start service: `sudo systemctl start mousehunter`
5. Monitor logs: `sudo journalctl -u mousehunter -f`

### Data Collection for Model Improvement
- [ ] Collect false positive images from `runtime/evidence/`
- [ ] Collect true positive images when actual prey detected
- [ ] Record deployment environment images for calibration
- [ ] Note lighting conditions (day/night/IR)

### Threshold Tuning (After Initial Deployment)
If too many false positives:
- Raise `thresholds.rodent` from 0.20 to 0.30 or 0.35
- Raise `min_detection_score` to match

If missing real detections:
- Lower `thresholds.rodent`
- Lower `score_threshold` from 0.9 to 0.7

---

## Current System Configuration

```
MouseHunter v3.0.0
Model: YOLOv8n custom (2 classes: cat=0, rodent=1)
reg_max: 8
Thresholds: cat=0.50, rodent=0.20
Confirmation: score_accumulation (3.0s window, 0.9 threshold)
Spatial validation: enabled (0.25 box expansion)
Hardware: Raspberry Pi 5 + Hailo-8L + PiCamera 3 NoIR
```

---

## Session 5: Deployment Debugging & Training Data Capture (February 5, 2026)

### Deployment Issue: Telegram Bot Unresponsive

After initial deployment, the Telegram bot stopped responding. Investigation revealed:

**Root Cause:** Application crashed when camera wasn't available, causing systemd restart loop (37 restarts observed).

**Original Code:**
```python
def _start_detection_thread(self) -> None:
    if self._camera is None or self._detector is None:
        raise RuntimeError("Camera or detector not initialized")
```

**Fix Applied:** Made detection thread optional, allowing Telegram bot and API to run without camera:
```python
def _start_detection_thread(self) -> None:
    missing = []
    if self._camera is None:
        missing.append("camera")
    if self._detector is None:
        missing.append("detector")
    if missing:
        logger.warning(
            f"Detection disabled: {', '.join(missing)} not available. "
            "Telegram bot and API will still run."
        )
        return
```

This enables debugging Telegram/API functionality without requiring camera hardware.

### False Positive Investigation

User reported prey detection (rodent 37%) when only human was present. Analysis:
- Cat-as-anchor strategy requires cat detection (50%+) before prey can trigger lockdown
- Low rodent threshold (0.20) can cause false positives on unexpected objects
- Recommendation: Raise `thresholds.rodent` to 0.30-0.40 after collecting deployment data

### New Feature: Training Data Capture

Implemented comprehensive training data capture system for YOLO model improvement:

**Three Capture Modes:**
1. **Periodic**: Every 30 minutes, captures environment images regardless of detections
2. **Cat-Only**: When cat detected for 2+ seconds without prey (requires cat_only_delay_seconds cooldown)
3. **Near-Miss**: When verifying state resets to idle with accumulated score > 0 (potential false negatives)

**New Files:**
- `src/mousehunter/storage/training_data.py` - Complete capture module

**Configuration Added to config.json:**
```json
"training_data": {
    "_comment": "Capture images for YOLO model training/improvement",
    "enabled": false,
    "periodic_interval_minutes": 30,
    "capture_cat_only": true,
    "cat_only_delay_seconds": 2.0,
    "capture_near_miss": true,
    "include_detections_json": true,
    "max_images_per_day": 100,
    "use_inference_resolution": true,
    "remote_path": "MouseHunter/training",
    "local_dir": "runtime/training_data"
}
```

**Features:**
- Daily capture limit (max 100 images/day)
- Automatic cloud upload via rclone (if configured)
- Metadata JSON with detection info for each capture
- Cooldown between captures to avoid duplicates
- Resolution options: inference (640x640) or main (1920x1080)

**Integration with Prey Detector:**
Added callback infrastructure for cat-only and near-miss events:
- `on_cat_only()` - Called when cat present without prey for configured delay
- `on_near_miss()` - Called when verifying resets without confirmation

### Bug Fixes

| Issue | File | Fix |
|-------|------|-----|
| Camera crash | `main.py` | Detection thread now optional |
| Unused imports | `training_data.py` | Removed `shutil`, `BytesIO` |
| Null pointer risk | `training_data.py` | Added defensive check for `cat_detection.confidence` |
| Attribute conflict | `training_data.py` | Renamed `capture_cat_only` to `cat_only_enabled` |

### Git Commits

```
697cfb9 Update log with Session 4: Testing & Score Accumulation Implementation
```

Note: Training data capture implementation was completed in this session.

### Cloud Storage Notes

- rclone automatically creates remote directories if they don't exist
- Configure with `rclone config` to create remote (e.g., "gdrive")
- Set `rclone_remote` in config to enable uploads
- Images upload asynchronously to avoid blocking detection loop

---

## Current System Configuration (v3.0.0)

```
MouseHunter v3.0.0
Model: YOLOv8n custom (2 classes: cat=0, rodent=1)
reg_max: 8
Thresholds: cat=0.50, rodent=0.20
Confirmation: score_accumulation (3.0s window, 0.9 threshold)
Spatial validation: enabled (0.25 box expansion)
Training data capture: available (disabled by default)
Cloud storage: rclone-based (optional)
Hardware: Raspberry Pi 5 + Hailo-8L + PiCamera 3 NoIR
```

---

*Log updated: February 5, 2026 (Session 5 - Deployment Debugging & Training Data)*
