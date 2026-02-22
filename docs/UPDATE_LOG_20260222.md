# Update Log - February 22, 2026

## Session 17: Disable Zoom Detection — Refocus on Image Quality

### Objective
Disable two-stage zoom detection to eliminate false positives, refocusing the improvement strategy on IR illumination and sensor upgrades rather than software workarounds.

---

## 1. Context & Decision

### Problem with Zoom Detection in Production
The two-stage zoom detection (Session 14) was designed to detect small prey at distance by cropping the cat region from the 1080p main stream and re-running inference at ~3-5x effective zoom.

In practice, it caused **repeated false positive lockdowns**:
- Humans falsely detected as cat (model has no background class)
- Zoom crops human body → model finds "rodent" patterns → score accumulates → lockdown
- Cat confirmation + spatial validation (Session 14 fixes) reduced but did not eliminate the issue
- The fundamental problem: a 2-class model will always find cats/rodents in whatever it sees — more inference passes = more false positive surface area

### Why Disable Rather Than Fix Further
- **Root cause is elsewhere**: Session 16 proved the real detection gap is low-light image quality, not pixel count. At close range (0.5-2m from cat flap), prey is 50-100+ px in 640x640 — well above YOLOv8's detection limit.
- **Engineering effort vs. payoff**: Properly fixing zoom requires model retraining with negative samples (humans, empty scenes). That effort is better spent on IR illumination, which addresses the actual root cause.
- **Risk reduction**: Every false lockdown erodes trust in the system. Disabling zoom immediately eliminates this failure mode.
- **Zoom can return later**: The config toggle (`inference.zoom_detection.enabled`) preserves all zoom code. Once the model is retrained with negatives, zoom can be re-enabled safely.

---

## 2. Changes Made

### Config Change
- `config.json`: `inference.zoom_detection.enabled` set from `true` to `false`

### Code Verification
Confirmed all code paths work correctly with zoom disabled:
- `main.py:515` — `zoom_provider` set to `None` when disabled
- `prey_detector.py:289` — `zoom_frame_provider is not None` check short-circuits, entire zoom block skipped
- `camera_service.py:capture_main_frame_rgb()` — never called (only used by zoom)
- Score accumulation, spatial validation, evidence pipeline — all function identically
- `intersection_type` in status API will only report "overlap", "proximity", or "disabled" (never "zoom")
- No code paths assume zoom is always on; the feature was designed as optional from the start

### No Code Changes Required
The zoom implementation was properly guarded behind config checks. Only the config value needed to change.

---

## 3. Current Detection Pipeline (Without Zoom)

```
Camera lores (640x640 RGB) → HailoEngine inference (~10ms)
  → Multi-label + class-aware NMS (all classes above threshold per cell)
  → Per-class thresholds: cat=0.55, rodent=0.15
  → Spatial validation: prey must intersect/be near cat
  → Score accumulation: ≥0.9 score AND ≥3 detections in 5s window
  → Prey confirmed → LOCKDOWN
```

Single-stage inference only. Main stream (1080p YUV420) used only for evidence recording, not detection.

---

## 4. Next Steps (Unchanged from Session 16)

1. **Priority 1: 850nm IR illumination** — Address the actual root cause (low-light image quality). Test with current Pi Camera 3 NoIR first.
2. **Priority 2: IMX462 STARVIS sensor** — If IR alone is insufficient, upgrade to a sensor with 0.001 lux sensitivity and 2x NIR QE.
3. **Priority 3: Model retraining with negatives** — Once false positives are reduced at the model level, zoom detection can be safely re-enabled.

---

*Session: February 22, 2026*
*Config-only change — zoom detection disabled*
