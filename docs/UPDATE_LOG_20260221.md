# Update Log - February 21, 2026

## Session 16: Camera & Detection Performance Analysis — IMX462 Evaluation + Next Steps

### Objective
Evaluate upgrading the Pi Camera 3 NoIR (IMX708) to InnoMaker IMX462 STARVIS camera module, and identify the most impactful improvements for prey detection in the system's actual operating environment.

---

## 1. Physical Setup Context

### Environment
- Camera mounted **close to the cat flap**, at the end of a corridor
- Corridor: ~7-8 meters long, ~1.7 meters wide
- **Low ambient light** even during daytime — corridor acts as a tunnel with poor natural lighting
- Cat approaches from up to 8m away, but prey detection matters most at close range (0.5-2m from camera at the flap)

### Key Evidence: Security Camera Comparison
- A separate security camera (also near the cat flap) captured video of a prey event
- Running that video through the Pi's YOLOv8n model → **prey detected successfully**
- The Pi Camera 3 NoIR captured the same event → **total miss**
- Security cameras typically use STARVIS-class sensors (IMX290/291/307/327/462)

### Root Cause Conclusion
The model works. The pixel count is sufficient at close range. **The bottleneck is image quality in low light.** The Pi Camera 3 NoIR produces noisy/dark frames in the corridor's ambient lighting that the model cannot extract prey features from. The security camera's superior low-light sensor produced clean enough frames for the same model to succeed.

---

## 2. IMX462 STARVIS Camera Evaluation

### Sensor Comparison

| Feature | Pi Camera 3 NoIR (IMX708) | InnoMaker IMX462 STARVIS |
|---------|--------------------------|--------------------------|
| Resolution | 4608x2592 (12 MP) | 1920x1080 (2 MP) |
| Pixel size | 1.4 um | 2.9 um |
| Sensor size | 1/2.43" | 1/2.8" |
| Min illumination | ~1 lux | 0.001 lux |
| NIR QE at 850nm | Standard BSI | ~2x (STARVIS BSI) |
| Analog gain range | 0-~42 dB | 0-72 dB |
| Max FPS (1080p) | 30 (full-res crop) / 60 | 60 (RAW10) / 50 (RAW12) |
| HDR | Single-frame | DOL-HDR (120 dB dynamic range) |
| Autofocus | Yes (PDAF) | No (manual M12 mount) |
| Pi 5 support | Official, fully tuned ISP | Unofficial (imx290 driver, uncalibrated PiSP) |
| Lens | Fixed | Interchangeable M12 mount |
| IR-cut filter | None (NoIR variant) | None |

### Software Compatibility Assessment

**No code changes required** — all components are sensor-agnostic:

| Component | Compatible | Notes |
|-----------|------------|-------|
| camera_service.py (picamera2) | Yes | Abstracts sensor differences, same dual-stream config |
| config.json (1920x1080 / 640x640 / 30fps) | Yes | IMX462 supports all these natively |
| main.py zoom detection | Yes | Same 1080p main stream crop pipeline |
| hailo_engine.py | Yes | Receives same 640x640 RGB uint8 input |
| prey_detector.py | Yes | Detection logic is sensor-agnostic |
| Evidence recording (H.264 + CircularOutput2) | Yes | Same YUV420 stream, same encoder pipeline |
| Telegram snapshots | Yes | Uses lores RGB888, unchanged |

### Pi 5 Integration Requirements

1. **Adapter cable**: 15-to-22 pin **Type B** (opposite-side contacts) FPC cable required
   - CRITICAL: Wrong cable type (same-side contacts) swaps 3.3V and GND → **boot failure**
2. **config.txt**:
   ```
   camera_auto_detect=0
   dtoverlay=imx290,clock-frequency=74250000,cam0
   ```
3. **PiSP tuning file** (no official one exists for Pi 5):
   ```bash
   sudo cp /usr/share/libcamera/ipa/rpi/pisp/uncalibrated.json \
           /usr/share/libcamera/ipa/rpi/pisp/imx290.json
   ```
4. **Stock lens replacement**: The included 148° diagonal (118° horizontal) M12 lens is too wide for a 1.7m corridor. Replace with ~70-90° HFOV M12 lens for better pixel density on target.

### Pi 5 Known Risks

- Unofficial driver (piggybacks on imx290 kernel driver)
- Uncalibrated ISP tuning — color, white balance, noise reduction are generic
- Forum reports of I2C errors, "Dequeue timer expired", intermittent detection failures
- No sensor ID register — cannot auto-detect, requires manual dtoverlay
- May break on future OS/kernel updates

### Verdict for Our Setup

The resolution disadvantage (2 MP vs 12 MP) is **irrelevant** — at 0.5-2m range, even 1080p gives abundant pixels on the prey. The uncalibrated ISP is a secondary concern when the current camera produces **unusable dark frames**. The IMX462's ~10-100x better low-light sensitivity directly addresses our actual failure mode. The security camera comparison is effectively a proof of concept for STARVIS sensors in this exact scenario.

**Recommended only if IR illumination alone (with current camera) proves insufficient.**

---

## 3. Prioritized Improvement Plan

### Priority 1: Add 850nm IR Illumination (Test with Current Camera First)

**Rationale**: Cheapest validation (~$10-15), zero software risk, zero driver risk. The Pi Camera 3 NoIR does respond to 850nm IR — just with ~half the QE of IMX462. In a 1.7m-wide corridor at close range, a decent IR flood light may produce adequate frames.

**Action**:
1. Mount 850nm IR flood light at/near the cat flap
2. Verify Pi Camera 3 NoIR frame quality improves in live conditions
3. If frames are clean enough for detection → problem solved, no camera swap needed
4. If still insufficient → validates IMX462 upgrade as necessary

**Expected impact**: Could solve the detection gap entirely if the IR provides sufficient illumination for the IMX708 sensor.

### Priority 2: IMX462 Camera Upgrade (If IR Alone Is Insufficient)

**Rationale**: Proven by security camera comparison. STARVIS sensor + IR illumination transforms the dark corridor into effectively daylight conditions for the camera.

**Action**:
1. Obtain IMX462 + correct Type B adapter cable + 70-90° HFOV M12 lens
2. Configure Pi 5 (config.txt, tuning file)
3. Test frame quality in corridor conditions
4. Capture new calibration images with IMX462 for potential model recompilation

**Expected impact**: Eliminates low-light detection failures. Combined with IR, provides consistent imaging regardless of ambient light.

### Priority 3: Retrain Model with Background/Negative Images

**Rationale**: The other major failure mode — humans misclassified as cats (no background class) → false lockdowns, amplified by zoom detection. Training data capture system (cat_only, near_miss callbacks) is already collecting suitable images.

**Action**:
1. Collect negative images: humans, furniture, empty corridor scenes
2. Create empty annotation files for each (YOLO format for "no objects")
3. Add 5-10% negative images to training set
4. Retrain YOLOv8n, re-export ONNX, recompile HEF
5. Consider adding more prey-in-mouth samples if available

**Expected impact**: Dramatically reduces false positives. Could allow lowering cat threshold (0.55 → 0.50) for better recall without false lockdown risk.

### Priority 4: YOLOv8s Model Upgrade

**Rationale**: YOLOv8s (11.2M params) has ~3x the capacity of YOLOv8n (3.2M params). Better feature extraction from marginal/noisy frames — directly relevant to our low-light operating conditions. Even with improved lighting, handles edge cases (partially occluded prey, unusual angles) better.

**Action**:
1. Verify Hailo-8L (8 TOPS) can run YOLOv8s at 640x640 within ~33ms frame budget
2. Train YOLOv8s with PatchedDetect (reg_max=8) on same dataset
3. Compile HEF, deploy, benchmark inference time on Pi
4. If inference exceeds 33ms, consider reducing framerate to 15fps (still adequate for detection)

**Expected impact**: Better accuracy across all conditions, especially on hard cases (prey-in-mouth, partial occlusion, noisy frames).

---

## 4. Solutions Evaluated and Deprioritized

The following were considered but are **not the primary bottleneck** for our setup:

| Solution | Why Not Priority |
|----------|-----------------|
| Higher inference resolution (640→1024) | At close range, 640px is sufficient — problem is frame quality, not pixel count |
| SAHI tiled inference | Solves small-object detection at distance — not our failure mode at the cat flap |
| Camera repositioning | Camera is already at the cat flap — as close as practical |
| Dual-camera setup | Significant complexity; single camera with good lighting should suffice |
| Pi HQ Camera (IMX477) | Better than IMX708 in low light (larger pixels) but not STARVIS-class; CS-mount lens flexibility is nice but IMX462 M12 mount also provides this |
| Motion-gated high-res capture | Mode switching latency (~100-200ms); cat at close range moves fast through flap |

---

## 5. Summary

The system's prey detection pipeline (thresholds, multi-label NMS, zoom detection, score accumulation) is **working correctly** — proven by successful detection on security camera footage. The gap is in the imaging hardware: the Pi Camera 3 NoIR produces insufficient image quality in the corridor's low ambient lighting.

The recommended approach is incremental:
1. Try IR illumination first (cheapest, lowest risk)
2. Upgrade sensor only if needed (IMX462 STARVIS)
3. Improve model robustness in parallel (negative images, larger model)

---

*Analysis session: February 21, 2026*
*No code changes — evaluation and planning only*
