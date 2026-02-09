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

---

## Session 6: Annotated Evidence System + False Positive Investigation (February 6, 2026)

### Major Feature: Annotated Key Frames + H.264 Video Evidence

Replaced the 416-JPEG evidence dump (~120MB per event) with:
1. **Annotated keyframe** (`keyframe_trigger.jpg`) - 640x640 JPEG with bounding boxes drawn
2. **H.264 MP4 video** (`evidence.mp4`) - pre-roll + post-roll encoded via ffmpeg

Cloud upload reduced from ~120MB to ~10MB per event.

#### New Files Created

**`src/mousehunter/camera/frame_annotator.py`**
- `annotate_frame(frame, detections)` → numpy array with bounding boxes drawn
- `annotate_frame_to_jpeg(frame, detections, quality)` → JPEG bytes
- Color scheme: green=cat, red=rodent, yellow=unknown
- Cached DejaVuSans-Bold font with PIL default fallback
- Label placement: above box normally, below if near frame top

**`src/mousehunter/camera/video_encoder.py`**
- `VideoEncoder`: pipes raw RGB24 frames to ffmpeg → libx264 H.264 MP4
  - Command: `-f rawvideo -pix_fmt rgb24 -s WxH -r FPS -i pipe:0 -c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p -movflags +faststart`
  - 120s encode timeout, stderr capture for error logging
- `EvidenceRecorder`: orchestrates pre-roll + post-roll in background daemon thread
  - `trigger_evidence_save()` returns immediately (non-blocking)
  - Background thread: collect post-roll → combine → encode → fire callbacks → gc.collect()
  - Auto-fallback to legacy JPEG mode if ffmpeg unavailable
- `FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None`

#### Files Modified

**`src/mousehunter/camera/circular_buffer.py`**
- Added `get_pre_roll_frames(seconds)` method
- Thin wrapper around existing `get_frames(since=...)`
- Returns references (not deep copies) to keep frames alive after buffer eviction

**`src/mousehunter/camera/camera_service.py`**
- Added `post_roll_seconds` and `evidence_format` constructor parameters
- Creates `self._evidence_recorder = EvidenceRecorder(...)` alongside existing buffer
- `trigger_evidence_save()` routes: video→EvidenceRecorder, frames→legacy buffer
- Added `on_evidence_complete(callback)` method
- Updated factory function with new params

**`src/mousehunter/config.py`**
- Added to `RecordingConfig`: `post_roll_seconds: float` (default 15.0), `evidence_format: str` (default "video")

**`config/config.json`**
- Added `"post_roll_seconds": 15` and `"evidence_format": "video"` to recording section

**`src/mousehunter/main.py`**
- Camera init: passes new config, registers `_handle_evidence_complete` callback
- `_handle_prey_confirmed()`: creates annotated frame, saves `keyframe_trigger.jpg`, defers cloud upload for video mode
- New `_handle_evidence_complete(evidence_dir, success)`: fires when encoding done, schedules cloud upload
- New `_schedule_cloud_upload(evidence_dir)`: extracted helper with defensive guards
- Added `self._evidence_events: dict[str, object] = {}` for per-directory event tracking (race condition fix)

**`src/mousehunter/camera/__init__.py`**
- Added exports: `VideoEncoder`, `EvidenceRecorder`, `FFMPEG_AVAILABLE`, `annotate_frame`, `annotate_frame_to_jpeg`

#### Evidence Directory Structure (New)

```
runtime/recordings/prey_20260206_143022/
  evidence.mp4          # H.264 video (pre-roll + post-roll)
  keyframe_trigger.jpg  # Annotated 640x640 detection frame
  metadata.json         # Detection metadata (created by cloud_storage)
```

#### Threading Architecture

```
DetectionThread:
  _handle_prey_confirmed()
    → annotate_frame()                    [fast, ~10ms]
    → camera.trigger_evidence_save()      [returns immediately]
    → save keyframe_trigger.jpg           [fast, ~20ms]
    → schedule _execute_lockdown()        [async on MainThread]

EvidenceRecorderThread (background, daemon):
    → collect post-roll frames for 15s
    → pipe all frames to ffmpeg subprocess (~10-30s)
    → call _handle_evidence_complete()
       → schedule cloud upload on MainThread
```

#### Bugs Found and Fixed During Review

| # | Issue | Fix |
|---|-------|-----|
| 1 | `annotate_frame_to_jpeg` crashes if PIL unavailable | Added `PIL_AVAILABLE` guard with `raise RuntimeError` |
| 2 | Label renders off-screen at frame top | `label_y = y1 - 20 if y1 >= 20 else y2 + 2` |
| 3 | No warning on re-trigger during active recording | Added `is_alive()` check with warning log |
| 4 | `_schedule_cloud_upload` missing defensive guard | Added `not self._cloud_storage` to early return |
| 5 | Race condition on `_last_prey_event` for deferred uploads | Added `self._evidence_events` dict mapping evidence_dir→event, stored at trigger time, popped at callback time |

#### Git Commit

```
6041067 Add Telegram notification when cat is detected
```

Note: The evidence system changes were included in the same push.

---

### False Positive Investigation: Human Detected as Cat

#### Observation
The custom YOLOv8n model detected a human (person's legs in white pants) as "cat" with 60.4% confidence in the cat flap corridor.

**Evidence:** `test_video/cat_only_20260206_172311.jpg` + `.json`
- class_id=0, class_name="cat", confidence=0.604
- bbox: bottom-right quadrant of frame (legs area)
- detection_state: "MONITORING"

#### Root Cause Analysis

This is **NOT a code bug** - it's a model architecture limitation.

The custom YOLOv8n model has only 2 classes: cat (0) and rodent (1). There is:
- **No background class** / "other" / "unknown" category
- **No way to express "this is neither cat nor rodent"**

YOLOv8 uses sigmoid (not softmax) per-class scores, so classes are independent. But when the model sees something with cat-like features (warm body, leg-like shapes, similar proportions at certain angles), it has no option but to assign it to one of its two classes.

#### Why Fine-Tuning Lost Human Detection

When fine-tuning from COCO pretrained weights:
1. **Backbone** (feature extraction layers): retained from COCO, still "sees" human features
2. **Classification head**: completely replaced — 80-class output → 2-class output
3. The new 2-class head was trained exclusively on cat/rodent images with no background examples
4. Result: backbone activates on human-like features, head forces output to cat or rodent

This is standard transfer learning behavior, not a bug.

#### Recommended Solutions

**Quick fix (no retraining):**
- Raise `thresholds.cat` from 0.50 to **0.65-0.70** to reject low-confidence cat detections
- The false positive was only 60.4% — a higher threshold eliminates it

**Medium-term (model improvement):**
- Add **background/negative images** to the training set (YOLO supports this natively with label-free images)
- Collect hard negatives using the existing training data capture system (cat_only mode already captures these cases)
- Include 10-20% background images: empty corridor, human legs, other non-target objects

**Long-term (architecture):**
- Add a 3rd class ("background" or "other") if background images alone aren't sufficient
- This is generally not needed — YOLO handles background images well without an explicit class

#### Key Insight: YOLO Background Training
YOLO natively supports background images: if an image has no `.txt` label file (or an empty one), the model treats all predictions on that image as false positives during training. This teaches the model "not everything is a detection." No code changes needed — just add unlabeled images to the training dataset.

---

### Current System Configuration (v3.0.0)

```
MouseHunter v3.0.0
Model: YOLOv8n custom (2 classes: cat=0, rodent=1)
reg_max: 8
Thresholds: cat=0.50, rodent=0.20
  ⚠ Consider raising cat threshold to 0.65-0.70 to reduce human false positives
Confirmation: score_accumulation (3.0s window, 0.9 threshold)
Spatial validation: enabled (0.25 box expansion)
Evidence: H.264 video + annotated keyframe (NEW)
  - post_roll_seconds: 15
  - evidence_format: "video" (fallback: "frames")
Training data capture: available (disabled by default)
Cloud storage: rclone-based (optional)
Hardware: Raspberry Pi 5 + Hailo-8L + PiCamera 3 NoIR
```

---

### Action Items

- [ ] Raise cat confidence threshold to 0.65-0.70 (quick false positive fix)
- [ ] Collect background/negative images via training data capture system
- [ ] Include hard negatives (human, empty corridor) in next training round
- [ ] Verify H.264 evidence recording works on Pi (ffmpeg installed)
- [ ] Monitor cloud upload sizes with new video evidence format

---

*Log updated: February 6, 2026 (Session 6 - Annotated Evidence System + False Positive Investigation)*

---

## Session 7: System Freeze Fix — 6 Stability Fixes (February 8, 2026)

### The Incident

On Feb 8, 2026, the MouseHunter system (Pi 5, 8GB RAM) completely froze after a prey detection event, requiring a hard reboot. The system was running Firefox alongside MouseHunter when a prey detection triggered the full evidence pipeline.

### Root Cause Analysis (from journal logs)

Six contributing factors identified:

| # | Factor | Impact |
|---|--------|--------|
| 1 | **Memory exhaustion** | CircularVideoBuffer held 450 raw frames (2.7 GB). Evidence recording copied 712 frames (4.3 GB). Combined ~7 GB just for video, on top of OS + Firefox + Hailo. |
| 2 | **SD card swap thrashing** | 511 MB swap on SD caused a death spiral instead of clean OOM kill |
| 3 | **Event loop starvation** | 2.5 min Telegram polling gap, jammer 93s late on 300s timer, state tick 102s late |
| 4 | **rclone thread pool exhaustion** | Two 300s upload timeouts blocked default executor threads |
| 5 | **Audio blocking** | `audio.play()` called synchronously in async `_execute_lockdown()`, ALSA device probing blocked |
| 6 | **No LOCKDOWN timeout failsafe** | State machine relied solely on jammer `is_active` check |

### Fixes Applied

#### Fix 1: Replace Raw Frame Buffer with CircularOutput2 (camera_service.py) — LARGEST CHANGE

**Before**: Background thread captures both main (1920x1080 RGB, 6 MB/frame) and lores streams. Main frames copied into Python deque (2.7 GB). Evidence recorder copies 712 frames (4.3 GB) + pipes to ffmpeg.

**After**: Hardware H.264 encoder encodes main stream on the ISP. `CircularOutput2` stores ~19 MB of compressed H.264 in a time-based ring buffer. Evidence saved via `open_output(PyavOutput("evidence.mp4"))` → background thread sleeps for post-roll → `close_output()`.

Key changes:
- Main stream format: `RGB888` → `YUV420` (required for hardware H.264 encoder)
- Removed `CircularVideoBuffer` dependency from camera_service (legacy file kept for fallback)
- Removed `EvidenceRecorder` dependency from camera_service (legacy file kept for fallback)
- Capture loop now only captures lores stream for inference (no main frame copies)
- Snapshots (`capture_snapshot_bytes`, `get_main_frame`, `save_snapshot`) use on-demand `capture_array("lores")` (RGB888)
- Evidence serialization guard: `self._evidence_recording` flag prevents parallel saves
- `on_evidence_complete` callbacks now owned directly by CameraService

**Memory impact**: ~7 GB → ~19 MB for the entire video pipeline.

**picamera2 API used**:
```python
from picamera2.encoders import H264Encoder
from picamera2.outputs import CircularOutput2, PyavOutput

encoder = H264Encoder(bitrate=5_000_000, repeat=True)
circular = CircularOutput2(buffer_duration_ms=15000)
picam2.start_recording(encoder, circular)

# On trigger:
circular.open_output(PyavOutput("evidence.mp4"))  # flushes pre-roll
time.sleep(post_roll_seconds)                      # in background thread
circular.close_output()                             # finalizes MP4
```

#### Fix 2: Systemd Watchdog + Memory Limits (mousehunter.service, main.py)

**Service file changes**:
- `Type=simple` → `Type=notify` (enables sd-notify protocol)
- Added `WatchdogSec=30` (systemd kills if no heartbeat in 30s)
- Added `MemoryMax=6G` (hard cgroup limit — SIGKILL if exceeded, leaves 2 GB for OS)
- Added `MemoryHigh=5G` (soft limit — triggers memory pressure, slows allocation)

**main.py changes**:
- Import `sdnotify` (pure-python) with `systemd.daemon` fallback
- Send `READY=1` after all initialization complete
- Send `WATCHDOG=1` heartbeat every ~1s in main loop (after each `_state_machine_tick()`)

**New dependency**: `pip install sdnotify`

#### Fix 3: rclone Upload Hardening (cloud_storage.py)

- Added dedicated `ThreadPoolExecutor(max_workers=2, thread_name_prefix="rclone")` — isolates rclone from the default asyncio executor
- Reduced default timeout from 300s to 120s
- Added `_upload_in_progress` guard: skips new uploads if one is already running (prevents stacking)

#### Fix 4: LOCKDOWN Timeout Failsafe (main.py)

Added hard timeout in `_state_machine_tick()`:
- Computes `elapsed` since `_lockdown_start`
- If `elapsed >= lockdown_duration_seconds * 2` (default: 600s), force-deactivates jammer and transitions to COOLDOWN
- Prevents infinite LOCKDOWN if jammer check is delayed or jammer auto-off fails

#### Fix 5: Audio Non-Blocking (main.py)

Wrapped `self._audio.play()` in `run_in_executor` with `asyncio.wait_for(timeout=5.0)`:
- Prevents ALSA device probing from blocking the event loop
- Catches `TimeoutError` and `Exception` gracefully
- Feb 8 logs showed "ALSA: Couldn't open audio device: Unknown error 524"

#### Fix 6: Serialize Evidence Recordings (video_encoder.py)

Changed parallel recording behavior to rejection:
- **Before**: "starting new recording in parallel" (warning, then proceed)
- **After**: "SKIPPING new recording" (warning, return early with directory path)
- Keyframe is still saved by main.py before `trigger_evidence_save` is called
- Defense-in-depth: with Fix 1, evidence recording no longer accumulates raw frames

### YUV420 Snapshot Bug — FIXED

**Issue**: `capture_array("main")` with YUV420 stream returns a 2D array of shape `(height * 3 // 2, stride)` — raw YUV420 planar data, NOT RGB. `Image.fromarray()` on this produces garbled greyscale images with scrambled UV data at the bottom. picamera2 does NOT auto-convert; `capture_image("main")` also raises `RuntimeError` for YUV420.

**Affected methods**: `capture_snapshot_bytes()`, `save_snapshot()`, `get_main_frame()`.

**Fix applied**: All three methods now use `capture_array("lores")` which returns RGB888 at 640x640. This resolution is adequate for all callers (Telegram notifications, /photo command, REST API). The mock camera was also updated to return a realistic YUV420-shaped 2D array for the main stream.

**Alternative considered but rejected**: Converting with `cv2.cvtColor(frame, cv2.COLOR_YUV420p2RGB)` — adds OpenCV dependency for snapshots and the full 1080p resolution is unnecessary for notification images.

### Files Modified

| File | Change Type | Description |
|------|------------|-------------|
| `src/mousehunter/camera/camera_service.py` | **Rewritten** | CircularOutput2 + hw H.264 encoder, lores-only capture loop |
| `src/mousehunter/main.py` | Modified | sd-notify watchdog, LOCKDOWN failsafe, audio non-blocking |
| `src/mousehunter/storage/cloud_storage.py` | Modified | Dedicated executor, reduced timeout, upload guard |
| `src/mousehunter/camera/video_encoder.py` | Modified | Reject parallel recordings |
| `systemd/mousehunter.service` | Modified | Type=notify, WatchdogSec, MemoryMax/MemoryHigh |
| `src/mousehunter/camera/__init__.py` | Modified | Updated docstring |

### Remaining TODO

- [x] ~~**Verify `capture_array("main")` format on Pi**~~ — confirmed: returns YUV420 2D array, NOT RGB
- [x] ~~Change `capture_snapshot_bytes()`, `save_snapshot()`, `get_main_frame()` to use lores stream~~ — DONE
- [ ] **Install sdnotify**: `pip install sdnotify` in the venv on Pi
- [ ] **Test evidence MP4 on Pi**: verify CircularOutput2 → PyavOutput produces valid MP4
- [ ] **Test /photo command**: verify Telegram snapshot still works with new capture method
- [ ] **Monitor memory**: `htop` during prey detection — should stay flat at ~1-2 GB
- [ ] **Test systemd watchdog**: `sudo journalctl -u mousehunter | grep WATCHDOG`
- [ ] **Overnight stability test**: run 12+ hours with Firefox closed

### Deployment Steps

```bash
# On Pi
cd ~/mousehunter_hailo_v2
git pull
pip install sdnotify  # new dependency
sudo systemctl daemon-reload  # reload service file changes
sudo systemctl restart mousehunter
sudo journalctl -u mousehunter -f  # watch logs
```

---

*Log updated: February 8, 2026 (Session 7 - System Freeze Fix + YUV420 snapshot fix)*
