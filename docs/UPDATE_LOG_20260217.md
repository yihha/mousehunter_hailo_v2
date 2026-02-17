# Update Log - February 17, 2026

## Session 11: Prey Detection Pipeline — Detect-Low-Confirm-High Pattern

### Objective
Investigate and fix missed prey detection events where a cat carrying prey was clearly visible but the system failed to detect the rodent. Implement a "detect low, confirm high" pattern to allow weak prey-in-mouth detections to accumulate over time.

---

## 1. Root Cause Analysis

### Problem
Cat was detected without issues, but prey (rodent in cat's mouth) was completely missed during daytime with somewhat dark lighting.

### Investigation
Traced the full detection pipeline through three gates:

| Gate | Component | Old Value | Function |
|------|-----------|-----------|----------|
| Gate 1 | Engine threshold (`HailoEngine`) | 0.25 | Filter YOLO grid cells in postprocessing |
| Gate 2 | Per-class threshold (`PreyDetector`) | 0.25 | Per-class confidence filter |
| Gate 3 | Min detection score (accumulator) | 0.20 | Per-frame gate for score accumulation |

### Root Cause
The `confidence_threshold` property in `config.py` returned the **rodent threshold** (0.25), which was passed to `HailoEngine` as the engine-level filter. This killed weak prey-in-mouth detections (typically 0.20–0.34 confidence) at the engine level **before** they could reach the score accumulation system.

The `min_detection_score` of 0.20 was effectively **dead code** — no detection below 0.25 could ever reach it because the engine already filtered it out.

### YOLO Engine Detail
Each of the 8400 grid cells outputs only ONE class via `argmax(class_scores)`. The engine threshold applies to `max(class_scores)` per cell. With only 2 classes (cat/rodent) and no background class, every cell above 0.10 outputs either cat or rodent — making the engine threshold critical for what reaches downstream.

---

## 2. Solution: Detect-Low-Confirm-High Pattern

### Architecture
```
Frame → HailoEngine (gate 1: 0.10) → PreyDetector per-class (gate 2: 0.15)
  → Score Accumulator (gate 3: min_score 0.20)
  → Dual Confirmation: accumulated_score ≥ 0.9 AND detection_count ≥ 3
```

### New Threshold Values

| Parameter | Before | After | Rationale |
|-----------|--------|-------|-----------|
| `engine_confidence_threshold` | 0.25 (implicit) | **0.10** | Noise filter only — let weak signals through |
| Rodent per-class threshold | 0.25 | **0.15** | Low gate to allow prey-in-mouth detections |
| `min_detection_score` | 0.20 (dead code) | **0.20** | Now the real per-frame gate (works because engine is 0.10) |
| `prey_window_seconds` | 3.0 | **5.0** | More time to accumulate weak detections |
| `prey_score_threshold` | 0.9 | **0.9** | Unchanged — high bar for cumulative evidence |
| `prey_min_detection_count` | (none) | **3** | NEW — prevents single high-confidence noise frame from triggering |
| `reset_on_cat_lost_seconds` | 1.5 | **5.0** | Cat briefly leaving frame doesn't reset accumulation |

### Key Design Decision
User proposed removing engine threshold entirely for rodent once cat is confirmed. Analysis showed this isn't practical because:
1. Engine threshold is **class-agnostic** (applies to max score across all classes per grid cell)
2. With only 2 classes and no background class, every cell outputs cat or rodent
3. Processing all 8400 grid cells with no filter would impact performance

Settled on 0.10 as a compromise — low enough to pass weak prey detections, high enough to filter pure noise.

---

## 3. Implementation Details

### config.py — New `engine_confidence_threshold` field
```python
engine_confidence_threshold: float = Field(
    default=_json_config.get("inference", {}).get("engine_confidence_threshold", 0.10),
    description="Engine-level threshold for YOLO postprocessing.",
)

@property
def confidence_threshold(self) -> float:
    """Engine-level confidence threshold for YOLO postprocessing."""
    return self.engine_confidence_threshold  # Was: self.thresholds.get("rodent", 0.25)
```

### prey_detector.py — Dual confirmation check
```python
# Check if both thresholds reached (score AND count)
if (accumulated >= self.prey_score_threshold
        and detection_count >= self.prey_min_detection_count):
    # CONFIRMED — prey detected with high cumulative confidence
```

### Fallback Default Consistency
Fixed inconsistent fallback defaults across all three layers:

| Parameter | config.json | config.py fallback | prey_detector.py default |
|-----------|------------|-------------------|------------------------|
| rodent threshold | 0.15 | 0.15 | 0.15 |
| prey_window_seconds | 5.0 | 5.0 | 5.0 |
| reset_on_cat_lost_seconds | 5.0 | 5.0 | 5.0 |
| prey_min_detection_count | 3 | 3 | 3 |

---

## 4. Tests

### New test: `test_confirmation_requires_both_score_and_count`
Verifies dual confirmation:
- High score (0.95) with only 2 detection frames → does NOT confirm (count < 3)
- After adding more entries so count reaches 4 → DOES confirm (score ≥ 0.9 AND count ≥ 3)

### Updated tests
- `test_confirmation_mode_set` — checks new window (5.0) and min_detection_count (3)
- `test_get_status_includes_score_fields` — checks for `min_detection_count` in status

---

## 5. Files Changed

| File | Change |
|------|--------|
| `config/config.json` | Added `engine_confidence_threshold: 0.10`, lowered rodent to 0.15, window to 5.0s, added `min_detection_count: 3` |
| `src/mousehunter/config.py` | Added `engine_confidence_threshold` field, `prey_min_detection_count` field, fixed `confidence_threshold` property, fixed all fallback defaults |
| `src/mousehunter/inference/prey_detector.py` | Added `prey_min_detection_count` param, dual confirmation logic, updated defaults, logging, factory function, get_status |
| `tests/conftest.py` | Updated `prey_detector_score_accumulation` fixture with new values |
| `tests/test_prey_detector.py` | Updated existing tests, added `test_confirmation_requires_both_score_and_count` |

---

## 6. Expected Impact

### Should now detect
- Prey-in-mouth with confidence 0.20–0.34 (previously killed at engine level)
- Weak detections in dark/low-contrast lighting that produce 0.15–0.25 confidence
- Brief glimpses that accumulate over 5 seconds across multiple frames

### False positive protection
- Engine threshold 0.10 still filters pure noise
- Per-class rodent threshold 0.15 rejects very weak signals
- Per-frame gate 0.20 prevents noise from entering accumulation
- Dual confirmation: cumulative score ≥ 0.9 AND ≥ 3 separate frames
- These together make it extremely unlikely for random noise to trigger a false positive

---

*Commit: `54c123c` — Pushed to origin/main*
*Log created: February 17, 2026*

---

## Session 12: Telegram Notification Reliability Fix (February 17, 2026)

### Objective
Fix silent Telegram notification failures — sometimes timeout, sometimes no response at all with no error logged.

---

## 1. Root Causes Identified

Five issues found causing notification loss:

| # | Issue | Severity | Impact |
|---|-------|----------|--------|
| 1 | No explicit timeout on Telegram API calls | HIGH | Default 20s read timeout × 3 retries = 60s+ blocking the event loop |
| 2 | Cat notification future is fire-and-forget | HIGH | Exceptions silently swallowed — no log, no error, no trace |
| 3 | No `RetryAfter` handling in retry logic | MEDIUM | Telegram rate-limit responses not retried properly; max retry wait too low (10s) |
| 4 | `notify_sync()` future never checked | LOW | Same fire-and-forget pattern as #2 |
| 5 | No clear startup success/failure log | LOW | User has no way to know if notifications are working until one is missed |

---

## 2. Fixes Applied

### Fix 1: Explicit API Timeouts (telegram_bot.py)

Added to all three Telegram API call sites (`send_message`, `send_photo`, text-only `send_message` in `send_alert`):

```python
await self._app.bot.send_message(
    chat_id=self.chat_id,
    text=text,
    read_timeout=10,
    write_timeout=10,
    connect_timeout=5,
)
```

**Effect**: Each attempt limited to ~10s instead of 20s default. With 3 retries, worst case is ~30s + backoff instead of 60s+.

### Fix 2: Done Callback on Cat Notification Future (main.py)

Added `_notification_callback()` method and attached it to the cat notification future:

```python
future = asyncio.run_coroutine_threadsafe(
    self._send_cat_notification(image_bytes),
    self._main_loop,
)
future.add_done_callback(self._notification_callback)
```

Now any exception from the coroutine is logged instead of silently swallowed.

### Fix 3: RetryAfter Handling + Increased Max Wait (telegram_bot.py)

- Imported `RetryAfter` from `telegram.error`
- Added to `_RETRY_EXCEPTIONS` tuple
- Increased max retry backoff from 10s to 30s (Telegram rate limits can request up to 30s)

### Fix 4: Outer `asyncio.wait_for()` Safety Net (main.py)

Wrapped both `_send_cat_notification()` and `_execute_lockdown()` Telegram sends:

```python
await asyncio.wait_for(
    self._telegram.send_alert(message, image_bytes, include_buttons=True),
    timeout=30,
)
```

Prevents the entire main event loop from blocking even if retry + API timeout gets stuck. `asyncio.TimeoutError` caught with clear log message.

### Fix 5: Startup Notification Logging (telegram_bot.py)

```python
# Success:
logger.info("Telegram notifications working - startup message sent")

# Failure:
logger.warning(
    "Telegram notifications FAILED - startup message not delivered: ... "
    "Notifications may not work until network is restored."
)
```

### Bonus Fix: `notify_sync()` Done Callback (telegram_bot.py)

Added `_notify_sync_callback()` to the fire-and-forget future in the module-level `notify_sync()` function.

---

## 3. Files Changed

| File | Change |
|------|--------|
| `src/mousehunter/notifications/telegram_bot.py` | API timeouts (10s/10s/5s), RetryAfter import + retry tuple, max wait 10→30, startup logging, notify_sync callback |
| `src/mousehunter/main.py` | `_notification_callback()` method, done callback on cat notification, `asyncio.wait_for(timeout=30)` on both send_alert calls |

---

## 4. Timeout Budget Analysis

With all fixes, worst-case timing for a single notification:

| Layer | Per-attempt | Total (3 attempts) |
|-------|-------------|---------------------|
| connect_timeout | 5s | 15s |
| read_timeout | 10s | 30s |
| retry backoff (exponential, max=30) | — | ~33s (1+2+30) |
| **Outer wait_for** | — | **30s hard cap** |

The outer `asyncio.wait_for(timeout=30)` cancels everything after 30s regardless — the event loop is never blocked longer than that.

---

## 5. Test Results

All 65 tests pass, 1 skipped (Hailo hardware test — expected on dev machine).

---

*Commit: `f883ff5` — Pushed to origin/main*
*Log updated: February 17, 2026*
