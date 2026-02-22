# Update Log - February 22, 2026

## Session 17: Disable Zoom Detection + Restore Thresholds — Refocus on Image Quality

### Objective
Disable two-stage zoom detection and restore detection thresholds to the pre-zoom validated levels, eliminating false positives and refocusing the improvement strategy on IR illumination and sensor upgrades rather than software workarounds.

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

### 2.1 Zoom Detection Disabled
- `config.json`: `inference.zoom_detection.enabled` set from `true` to `false`

Code verification confirmed all paths work correctly with zoom disabled:
- `main.py:515` — `zoom_provider` set to `None` when disabled
- `prey_detector.py:289` — `zoom_frame_provider is not None` check short-circuits, entire zoom block skipped
- `camera_service.py:capture_main_frame_rgb()` — never called (only used by zoom)
- Score accumulation, spatial validation, evidence pipeline — all function identically
- No code paths assume zoom is always on; the feature was designed as optional from the start

### 2.2 Thresholds Restored to Pre-Zoom Validated Levels

After disabling zoom, a false positive prey detection still occurred. Investigation revealed that thresholds had been lowered during the zoom sessions (Sessions 11-14) and never restored:

| Setting | Pre-zoom (validated Feb 18) | During zoom era | Restored |
|---------|---------------------------|-----------------|----------|
| Cat threshold | **0.60** | 0.55 | **0.60** |
| Rodent threshold | 0.30 → 0.15 (Session 11) | 0.15 | **0.25** |

**Why these caused false positives without zoom:**
- Cat at 0.55 let humans pass as "cat" (model has no background class)
- Multi-label NMS (Session 13) allows same grid cell to emit BOTH cat AND rodent
- A noisy cell producing cat@0.55 + rodent@0.15 trivially passes spatial validation (same location)
- Three such frames in 5 seconds → false lockdown

**Reference point:** The security camera video test on Feb 18 (post multi-label NMS fix, pre-zoom) successfully detected prey with cat=0.60, rodent=0.15, and no false positives. Rodent raised to 0.25 rather than the original 0.15 for additional false positive margin — at close range with IR illumination, real prey-in-mouth will produce confidence well above 0.25.

**Files changed:**
- `config/config.json` — thresholds updated
- `src/mousehunter/config.py` — fallback defaults updated to match
- `src/mousehunter/inference/prey_detector.py` — fallback defaults updated to match

### 2.3 Multi-label NMS: Kept (Correct but Understood)

The multi-label + class-aware NMS (Session 13) is **technically correct** for YOLOv8 (BCE loss = independent class probabilities) and **necessary** for prey-in-mouth detection (cat and rodent occupy same grid cells). However, it shifts the false-positive burden to the thresholds:

- **Before Session 13 (argmax):** A cell could only emit its best class. A cell that weakly fires for both cat and rodent would only emit cat (stronger signal). This was an "accidental" false positive filter — it worked by discarding information.
- **After Session 13 (multi-label):** Same cell emits both. Co-located cat+rodent trivially passes spatial validation. Thresholds must be high enough to filter noise.

**Decision:** Keep multi-label NMS. The argmax approach systematically killed prey-in-mouth detections, which is the system's most important detection case. The correct compensation is higher per-class thresholds (applied above) and long-term model retraining with negative samples.

---

## 3. Current Detection Pipeline

```
Camera lores (640x640 RGB) → HailoEngine inference (~10ms)
  → Multi-label + class-aware NMS (all classes above threshold per cell)
  → Per-class thresholds: cat=0.60, rodent=0.25
  → Spatial validation: prey must intersect/be near cat (box_expansion=0.25)
  → Score accumulation: ≥0.9 score AND ≥3 detections in 5s window
  → min_detection_score: 0.20 (per-frame gate for accumulation)
  → Prey confirmed → LOCKDOWN
```

Single-stage inference only. Main stream (1080p YUV420) used only for evidence recording, not detection. Zoom code preserved behind config toggle for future re-enabling.

---

## 4. Threshold Summary (Current State)

| Threshold | Value | Purpose |
|-----------|-------|---------|
| Engine confidence | 0.10 | Coarse noise filter in YOLO postprocess |
| Cat per-class | **0.60** | Minimum to consider a cat detection valid (restored from 0.55) |
| Rodent per-class | **0.25** | Per-class gate (raised from 0.15 for false positive margin) |
| min_detection_score | 0.20 | Real per-frame gate for score accumulation |
| Score threshold | 0.9 | Accumulated confidence needed to confirm |
| min_detection_count | 3 | Minimum separate frames with prey |
| Window | 5.0s | Time window for accumulation |
| Cat lost reset | 5.0s | Reset if no cat for this long |

---

## 5. Next Steps (Unchanged from Session 16)

1. **Priority 1: 850nm IR illumination** — Address the actual root cause (low-light image quality). Test with current Pi Camera 3 NoIR first.
2. **Priority 2: IMX462 STARVIS sensor** — If IR alone is insufficient, upgrade to a sensor with 0.001 lux sensitivity and 2x NIR QE.
3. **Priority 3: Model retraining with negatives** — Once false positives are reduced at the model level, zoom detection can be safely re-enabled and thresholds can be lowered for better recall.

---

*Session: February 22, 2026*
*Commits: `f7b7e08` (zoom disabled), `be2e60e` (thresholds restored)*
