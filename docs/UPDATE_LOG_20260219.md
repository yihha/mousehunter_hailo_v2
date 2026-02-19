# Update Log - February 19, 2026

## Session 15: Telegram Bot Polling Death Fix + Log Spam Reduction

### Objective
Fix Telegram bot becoming unresponsive to all commands (/unlock, /status, /help, etc.) after running for hours. Additionally, reduce log spam from zoom rejection messages that were rotating log files too fast.

---

## 1. Root Cause Analysis

### Telegram Polling Death
On Feb 19, the Telegram bot stopped responding to all commands. Investigation revealed:
- **Polling was dead for 8+ hours** — 10 pending commands found in Telegram queue dating back to 12:50, but service started at 22:33 the previous day
- **Zero `getUpdates` calls logged** in the entire current session — polling either never started or died very early
- **Outbound notifications worked fine** — prey alerts and cat notifications were sent successfully (these use direct `bot.send_message()`, not the polling updater)
- **No errors logged** — the polling failure was completely silent
- **Root cause**: `TelegramBot.start()` runs as `asyncio.create_task()` (main.py:151) — if polling fails, the exception is swallowed. PTB v22.5's internal polling tasks can die without notification.

### Zoom Log Spam
The "Zoom: no cat confirmed" log at INFO level fires ~6/sec when a cat is present and a human walks by (false cat detection → zoom → no cat confirmed). This generates ~36KB/min of log data, rotating the 10MB log files every ~4.5 hours and erasing valuable diagnostic entries like getUpdates calls and error messages.

---

## 2. Changes

### 2.1 Polling Health Monitor (`telegram_bot.py`)

**`_polling_error_callback()`** — sync callback passed to PTB's `start_polling(error_callback=...)`. Previously, `getUpdates` failures were silently swallowed by PTB; now they're logged at WARNING level.

**`check_polling_health()`** — async method that:
1. Checks `self._app.updater.running` property
2. If False and `self._started` is True, logs WARNING and restarts polling
3. On restart, uses `drop_pending_updates=False` to preserve any queued commands
4. Returns True if restart was needed
5. Tracks restart count for diagnostics

**MockTelegramApp** — added `running = True` to mock updater class so health checks don't crash in dev mode.

### 2.2 Command Handler Error Wrapping (`telegram_bot.py`)

Wrapped 4 handlers with try/except:
- `_handle_callback()` — inline button handler (Unlock/Scream/Ignore buttons)
- `_cmd_lock()` — /lock command
- `_cmd_unlock()` — /unlock command
- `_cmd_scream()` — /scream command

Previously, if `query.answer()`, `query.edit_message_text()`, or hardware calls threw, the exception propagated unhandled. Now they're caught, logged at ERROR, and a user-visible error reply is sent.

`_cmd_status()` and `_cmd_photo()` already had try/except — now all command handlers are consistent.

### 2.3 Main Loop Health Check (`main.py`)

**Periodic check in `_state_machine_tick()`**: Every 30 ticks (~30s), calls `self._telegram.check_polling_health()`. Logs WARNING if restart was needed.

**Startup task done callback**: Added `task.add_done_callback(self._telegram_task_callback)` on the telegram startup task (line 153). Catches startup failures that were previously silently swallowed by `asyncio.create_task()`. Follows existing pattern from `_lockdown_callback`.

### 2.4 Zoom Log Spam Reduction (`prey_detector.py`)

Changed two log lines from INFO to DEBUG:
- "Zoom: no cat confirmed in zoomed crop..." — fires ~6/sec when false cat detected
- "Zoom: {prey} not spatially near cat in zoom crop..." — fires on spatial validation rejection

The *successful* zoom detection log ("Zoom detection: rodent...") remains at INFO level — that's the valuable diagnostic entry.

---

## 3. Files Changed

| File | Changes |
|------|---------|
| `src/mousehunter/notifications/telegram_bot.py` | +`_polling_error_callback()`, +`check_polling_health()`, +`error_callback` to `start_polling()`, +try/except on 4 handlers, +mock `updater.running` |
| `src/mousehunter/main.py` | +`_telegram_health_check_counter`, +30s health check in `_state_machine_tick()`, +`_telegram_task_callback()`, +`add_done_callback` on telegram task |
| `src/mousehunter/inference/prey_detector.py` | 2x `logger.info` → `logger.debug` for zoom rejection logs |

---

## 4. Verification Plan

1. Deploy to Pi and restart service
2. Confirm `getUpdates` calls appear in log within 30s of startup
3. Send `/status` from Telegram — should get response
4. Verify zoom rejection logs are now at DEBUG level (won't appear at default INFO level)
5. Check that log rotation slows down (no longer rotating every 4-5 hours)
6. Simulate polling death: the health check should detect and restart within 30s

---

## 5. Logging Quality Improvements (Commit 2)

After the initial fix, a critical audit of logging quality across the entire codebase identified 5 additional issues. All were fixed in a second commit.

### 5.1 Hailo Inference Error Tracking (`hailo_engine.py`)

**Problem**: When the Hailo NPU inference fails, the error handler logs a single ERROR and returns an empty list. If the NPU enters a persistent failure state, the system is "blind" (no detections at all) with only a single error log entry — easy to miss.

**Fix**: Added `_consecutive_inference_errors` counter:
- First 3 errors: full ERROR with traceback (`exc_info=True`)
- At 10 consecutive: WARNING "system is blind" escalation
- Every 100th after: ERROR reminder (prevents log spam during sustained failure)
- On recovery: WARNING with count of errors recovered from
- Counter resets to 0 on successful inference

### 5.2 start_polling() Wrapping (`telegram_bot.py`)

**Problem**: In `start()`, the `start_polling()` call was outside try/except. If it throws (e.g., network down at boot), the entire `start()` method fails and the bot never initializes.

**Fix**: Wrapped `start_polling()` in try/except with a descriptive error message explaining that the bot can still send notifications but won't receive commands. The health check (from commit 1) will attempt restart in ~30s.

### 5.3 Cloud Metadata Parse Logging (`cloud_storage.py`)

**Problem**: In the evidence folder listing, metadata parse failures used bare `except Exception: pass` — any parse error was completely invisible.

**Fix**: Changed to `except Exception as e: logger.debug(f"Metadata parse failed for {folder.name}: {e}")`. Uses DEBUG because this is a non-critical path (listing old evidence), but at least failures are now discoverable.

### 5.4 Callback Error Identity (`prey_detector.py`)

**Problem**: All 3 callback error log sites used `logger.error(f"Callback error: {e}")` — impossible to tell which callback failed (cat_only, near_miss, or prey_detected) or see the traceback.

**Fix**: All 3 sites now log `logger.error(f"Callback {getattr(callback, '__name__', callback)} error: {e}", exc_info=True)`. The callback name and full traceback are now visible.

### 5.5 Periodic Heartbeat Log (`main.py`)

**Problem**: Between prey events (which may be hours or days apart), the log is essentially empty — no way to confirm the system is alive and healthy without SSH-ing in.

**Fix**: Added `_log_heartbeat()` method, called every 300 ticks (~5 minutes) in `_state_machine_tick()`. Single INFO line containing:
- Current state machine state
- Frame count (total + since last heartbeat)
- Average inference time (ms)
- Consecutive inference errors (if any)
- Cats seen since last heartbeat
- Total detection count and lockdown count
- Jammer status (LOCKED/unlocked)
- Telegram polling status (alive/DEAD/not-started) + restart count

Example output:
```
Heartbeat: state=IDLE, frames=54000 (+1500), inference=8.2ms, cats=12, detections=0, lockdowns=0, jammer=unlocked, telegram=alive
```

---

## 6. Files Changed (Commit 2)

| File | Changes |
|------|---------|
| `src/mousehunter/inference/hailo_engine.py` | +`_consecutive_inference_errors`, escalating error logging, recovery logging |
| `src/mousehunter/notifications/telegram_bot.py` | +try/except around `start_polling()` in `start()` |
| `src/mousehunter/storage/cloud_storage.py` | `except: pass` → `except: logger.debug(...)` for metadata parse |
| `src/mousehunter/inference/prey_detector.py` | 3x callback error logs now include callback name + `exc_info=True` |
| `src/mousehunter/main.py` | +`_log_heartbeat()`, +`_heartbeat_counter`, +`_cats_seen_since_heartbeat`, heartbeat every 300 ticks |

---

*Log created: February 19, 2026*
