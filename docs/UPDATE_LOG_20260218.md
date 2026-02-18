# Update Log - February 18, 2026

## Session 14: Two-Stage Cascaded Zoom Detection

### Objective
Fix prey detection failure at distance. The system detects prey correctly when the cat is close to the camera (confirmed by phone video diagnosis at t=2.14s), but fails in production because the Pi camera is mounted far from the cat entry point. In the real event (Feb 18, 18:04), the cat was small in frame (~15-20% of 640x640) and the rodent was ~10px — below YOLOv8's detection limit (~12-16px at stride-8).

---

## 1. Root Cause Analysis

### Problem
After Sessions 11 and 13 fixed threshold tuning and postprocessing (multi-label + class-aware NMS), the detection pipeline works correctly on close-range footage. But production logs from the Feb 18 event showed:
- Cat detected at confidence 0.60 (right at the old threshold boundary)
- Rodent never detected — too few pixels in the 640x640 lores frame
- The cat occupied ~15-20% of the frame, making the rodent in its mouth approximately 10px

### Physical Limitation
YOLOv8's smallest detection head (P3, stride-8) produces an 80x80 feature map from 640x640 input. Each grid cell covers 8x8 pixels. Objects need to be at least ~12-16px to produce a meaningful activation. A 10px rodent is below this practical limit — no amount of threshold tuning can detect what the network cannot see.

### Solution
**Two-stage cascaded detection**: When a cat is detected on the 640x640 lores stream but no prey is found, capture from the 1920x1080 main stream, crop the region around the cat, and resize to 640x640 for a second inference pass. This provides ~3-5x effective zoom, making a 10px rodent into 30-50px — well within detection range.

Also lower the cat threshold from 0.60 to 0.55. The live system had cat at exactly 0.60; any confidence jitter drops it below threshold and shortens the detection window. (Initially lowered to 0.50, raised to 0.55 after live false positive testing.)

---

## 2. Detection Pipeline (After This Change)

```
Camera lores (640x640 RGB) ──→ Stage 1: Hailo inference (~10ms)
                                   │
                                   ├─ Cat found + prey found → Score accumulation (normal path)
                                   │
                                   ├─ Cat found + NO prey → Stage 2: Zoom detection
                                   │     │
                                   │     ├─ Capture main stream (1920x1080 YUV420)
                                   │     ├─ Convert YUV420 → RGB
                                   │     ├─ Crop cat region (with 50% padding, squared)
                                   │     ├─ Resize crop to 640x640
                                   │     ├─ Hailo inference on zoomed crop (~10ms)
                                   │     ├─ Cat confirmed in zoom? (filters false positives)
                                   │     ├─ Prey spatially near cat in zoom? (filters noise)
                                   │     └─ Both pass → Score accumulation
                                   │
                                   └─ No cat → Skip (idle)
```

### Coordinate Mapping
Both streams come from the same sensor/ISP with the same ScalerCrop applied, then scaled to their target resolution. Normalized coordinates [0,1] map 1:1 between lores and main — no special coordinate transformation needed.

### Crop Calculation
```
cat_bbox = (x, y, w, h) normalized [0,1]
main_frame = 1920x1080

cx = cat_bbox.center_x * 1920      # Center in pixels
cy = cat_bbox.center_y * 1080
cat_w_px = cat_bbox.width * 1920
cat_h_px = cat_bbox.height * 1080

crop_w = cat_w_px * (1 + 2 * 0.5)  # 50% padding each side → 2x cat width
crop_h = cat_h_px * (1 + 2 * 0.5)
crop_size = max(crop_w, crop_h)     # Make square

x1 = clamp(cx - crop_size/2, 0, 1920)
y1 = clamp(cy - crop_size/2, 0, 1080)
x2 = clamp(cx + crop_size/2, 0, 1920)
y2 = clamp(cy + crop_size/2, 0, 1080)

cropped = main_frame[y1:y2, x1:x2]
resized = cv2.resize(cropped, (640, 640))
```

---

## 3. Implementation Details

### config/config.json — Threshold + zoom config
```json
"thresholds": {
    "cat": 0.55,        // was 0.60 (initially 0.50, raised after false positive testing)
    "rodent": 0.15
},
"zoom_detection": {
    "enabled": true,
    "crop_padding": 0.5,
    "_comment": "Two-stage: crop from 1080p main stream when cat detected without prey"
},
```

### config.py — New InferenceConfig fields
```python
zoom_detection_enabled: bool    # default from config.json (False if missing)
zoom_crop_padding: float        # default 0.5
```
Also updated hardcoded cat threshold fallback from 0.60 to 0.55 for consistency.

### camera_service.py — `capture_main_frame_rgb()`
New method that captures the main stream (YUV420) and converts to RGB:
```python
def capture_main_frame_rgb(self) -> np.ndarray | None:
    yuv = self._camera.capture_array("main")       # YUV420: (1620, 1920)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)  # RGB: (1080, 1920, 3)
    return rgb
```
Mock mode returns random RGB data with correct dimensions.

### prey_detector.py — Zoom support in `process_frame()`
Extended signature with optional callback:
```python
def process_frame(
    self,
    frame: np.ndarray,
    timestamp: datetime | None = None,
    zoom_frame_provider: Callable[[Detection], np.ndarray | None] | None = None,
) -> DetectionFrame:
```

After Stage 1, if cat found but no prey:
- Calls `zoom_frame_provider(cat_detection)` to get 640x640 zoomed crop
- Runs `engine.infer(zoom_frame)` for Stage 2
- Searches zoom results for prey using `_get_best_detection()` (applies per-class thresholds)
- **Cat confirmation required** — zoom crop must also detect a cat (filters false positives)
- **Spatial validation applied** within zoom frame (prey must be near cat in crop)
- Creates `SpatialMatch(intersection_type="zoom")` for logging/status
- Zoom prey feeds into score accumulation normally

### main.py — Zoom provider + crop logic
New `_get_zoom_frame(cat_detection)` method on `MouseHunterController`:
- Captures main frame RGB via camera service
- Maps cat normalized bbox → pixel coords on 1080p frame
- Pads by `zoom_crop_padding` (default 50% each side)
- Makes crop square, clamps to image bounds
- Resizes to 640x640

Modified `_detection_loop()`:
- Reads `zoom_detection_enabled` from config at startup
- Creates method reference `zoom_provider = self._get_zoom_frame` if enabled
- Passes `zoom_frame_provider=zoom_provider` to `process_frame()`

---

## 4. Performance Impact

| Step | Time | When |
|------|------|------|
| Stage 1 inference | ~8-10ms | Every frame |
| Main stream capture + YUV→RGB | ~3-5ms | Only when cat found, no prey |
| Crop + resize | ~1ms | Only when cat found, no prey |
| Stage 2 inference | ~8-10ms | Only when cat found, no prey |
| **Total (no cat)** | **~10ms** | Baseline unchanged |
| **Total (cat, no prey)** | **~25ms** | Within 33ms frame budget (30fps) |

Memory: ~6MB transient per zoom call (1920x1080x3 RGB), released immediately after inference. Negligible vs 6GB memory limit.

---

## 5. Known Limitations

### Zoom prey bbox coordinates
The prey Detection from Stage 2 has bbox coordinates in the zoom crop's coordinate space (normalized 0-1 within the 640x640 crop), not the original lores frame. If this prey triggers confirmation, the keyframe annotation will draw the prey bbox in the wrong position on the lores frame. This is acceptable because:
1. The prey is ~10px in the lores frame — too small to annotate meaningfully
2. The cat bbox annotation (from Stage 1) is still correct
3. The primary evidence is the MP4 video, not the keyframe annotation
4. The detection confidence and class name are reported correctly

### Edge clamping makes non-square crops
If the cat is near a frame edge, clamping produces a non-square crop that gets resized to 640x640 with slight aspect ratio distortion. YOLOv8 tolerates this via data augmentation during training.

---

## 6. Threshold Summary (Current State)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Engine confidence | 0.10 | Coarse noise filter in YOLO postprocess |
| Cat per-class | **0.55** | Minimum to consider a cat detection valid (was 0.60, then 0.50, settled on 0.55) |
| Rodent per-class | 0.15 | Lets weak prey-in-mouth detections through |
| min_detection_score | 0.20 | Real per-frame gate for score accumulation |
| Score threshold | 0.9 | Accumulated confidence needed to confirm |
| min_detection_count | 3 | Minimum separate frames with prey |
| Window | 5.0s | Time window for accumulation |
| Cat lost reset | 5.0s | Reset if no cat for this long |

---

## 7. Files Changed

| File | Change |
|------|--------|
| `config/config.json` | Cat threshold 0.60→0.55, added `zoom_detection` section |
| `src/mousehunter/config.py` | Added `zoom_detection_enabled` and `zoom_crop_padding` fields, updated cat threshold fallback to 0.55 |
| `src/mousehunter/camera/camera_service.py` | Added `capture_main_frame_rgb()` method (YUV420→RGB conversion) |
| `src/mousehunter/inference/prey_detector.py` | Added `zoom_frame_provider` parameter to `process_frame()`, Stage 2 zoom logic |
| `src/mousehunter/main.py` | Added `_get_zoom_frame()` method, modified `_detection_loop()` to wire up zoom provider |

---

## 8. Deployment

```bash
git pull && sudo systemctl restart mousehunter
```

Verify:
```bash
journalctl -u mousehunter -f | grep -iE "zoom|two-stage"
```

Expected log output:
- Startup: `Two-stage zoom detection ENABLED (padding=0.5)`
- When prey detected via zoom: `Zoom detection: rodent (0.XX) found in zoomed crop around cat`

---

## 9. Post-Deployment: False Positive Fix (Cat Confirmation in Zoom)

### Problem Discovered
Live testing revealed that the zoom feature **amplified false positives**. When a human walked in front of the camera:
1. Stage 1: Human misclassified as cat at 0.60 confidence (model has no background class)
2. Stage 2: Zoom cropped the human body from 1080p → resized to 640x640
3. Model found "rodent" at 0.40, 0.35, 0.50 in the zoomed human body
4. Score accumulated: 1.25 > 0.9 threshold, 3 frames → **full LOCKDOWN on a human**

Before zoom, this false cat detection was harmless (no prey found → MONITORING → timeout → IDLE). The zoom feature turned a benign false positive into a destructive one.

### Cat Threshold Adjustment
Raised cat threshold from 0.50 → 0.55. The original 0.50 was too aggressive — humans triggered at 0.50. However, 0.55 alone wasn't enough: the next test still triggered with human at 0.60 confidence.

### Root Cause
The model (2 classes: cat, rodent, no background class) is forced to classify all detected objects as either cat or rodent. Humans at distance resemble cats to the model. This is a **training data problem** — the model needs background/negative images (humans, furniture, empty scenes) with empty annotation files to learn to suppress detections on non-target content.

### Fix: Cat Confirmation + Spatial Validation in Zoom
Two safeguards added to `prey_detector.py` Stage 2 zoom logic:

1. **Cat confirmation**: After running inference on the zoomed crop, require a cat detection above threshold in the zoom frame. If the zoomed "cat" region doesn't show a cat, the original detection was likely a false positive. This works because:
   - A real cat zoomed in is **more** detectable (fills more of the frame, higher confidence)
   - A human zoomed in is **less** cat-like (skin, clothing, proportions visible)

2. **Spatial validation**: Apply the same spatial intersection check (expanded cat bbox) within the zoom frame. Prey must be near the cat in the crop, filtering random noise detections.

### Updated Detection Pipeline
```
Stage 2: Zoom detection (revised)
  ├─ Capture main stream → crop cat region → resize 640x640
  ├─ Hailo inference on zoomed crop
  ├─ Cat detected in zoom? (NEW CHECK)
  │     ├─ NO → "likely false positive" → skip prey search
  │     └─ YES → search for prey
  │           ├─ Prey spatially near cat in zoom? (NEW CHECK)
  │           │     ├─ NO → skip
  │           │     └─ YES → score accumulation
  │           └─ No prey → skip
```

### Live Test Result
After deploying the fix:
- Human walks in front of camera → Stage 1: cat detected (MONITORING)
- Stage 2: zoom crops human → **no cat in zoom** → prey search skipped
- No score accumulation, no lockdown
- Cat lost after 5.1s → clean reset to IDLE

### Zoom Log Messages (Changed to INFO Level)
- `Zoom: no cat confirmed in zoomed crop, skipping prey search (likely false positive)`
- `Zoom: rodent (0.XX) not spatially near cat in zoom crop, skipping`
- `Zoom detection: rodent (0.XX) found in zoomed crop around cat (zoom cat: 0.XX)`

### Updated Threshold Summary

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Engine confidence | 0.10 | Coarse noise filter in YOLO postprocess |
| Cat per-class | **0.55** | Minimum to consider a cat detection valid (was 0.50→0.55) |
| Rodent per-class | 0.15 | Lets weak prey-in-mouth detections through |
| min_detection_score | 0.20 | Real per-frame gate for score accumulation |
| Score threshold | 0.9 | Accumulated confidence needed to confirm |
| min_detection_count | 3 | Minimum separate frames with prey |
| Window | 5.0s | Time window for accumulation |
| Cat lost reset | 5.0s | Reset if no cat for this long |

### Files Changed

| File | Change |
|------|--------|
| `config/config.json` | Cat threshold 0.50→0.55 |
| `src/mousehunter/config.py` | Updated cat threshold fallback to 0.55 |
| `src/mousehunter/inference/prey_detector.py` | Cat confirmation + spatial validation in zoom, zoom rejection logs changed to INFO level |

### Industry Best Practice Note
The false positive issue is a common problem for custom object detectors without a background class. Standard fix: add 1-10% background/negative images (humans, empty scenes) with empty annotation files to the training set. The training data capture system (cat_only, near_miss callbacks) already collects these images for future retraining.

---

*Commits: `744fabe` (zoom feature), `944f24d` (update log), `5d0e27f` (cat threshold 0.55), `d268988` (cat confirmation safeguard)*
*Log created: February 18, 2026*
