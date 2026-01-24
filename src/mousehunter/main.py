"""
MouseHunter Main Controller

The central orchestrator implementing the state machine:
IDLE -> VERIFYING -> LOCKDOWN -> COOLDOWN -> IDLE

Coordinates:
- Camera capture and circular buffer
- Hailo-8L inference
- Prey detection with debouncing
- Jammer activation
- Audio deterrent
- Telegram notifications
- REST API server
"""

import asyncio
import logging
import signal
import sys
import threading
from datetime import datetime
from enum import Enum, auto
from typing import Callable

import numpy as np

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """Main system state machine states."""

    IDLE = auto()  # Normal monitoring
    VERIFYING = auto()  # Potential prey, verifying
    LOCKDOWN = auto()  # Prey confirmed, jammer active
    COOLDOWN = auto()  # Post-lockdown cooldown


class MouseHunterController:
    """
    Main controller orchestrating all system components.

    Implements the Detect-and-Deny logic loop with state machine
    for reliable prey detection and interdiction.
    """

    def __init__(self):
        """Initialize the controller."""
        self._state = SystemState.IDLE
        self._running = False
        self._main_loop: asyncio.AbstractEventLoop | None = None
        self._loop_ready = threading.Event()  # Signals when main loop is ready

        # Component instances (initialized in start())
        self._jammer = None
        self._audio = None
        self._camera = None
        self._detector = None
        self._telegram = None
        self._cloud_storage = None

        # Detection thread
        self._detection_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Background async tasks (for proper cancellation on shutdown)
        self._background_tasks: list[asyncio.Task] = []

        # State tracking
        self._last_state_change = datetime.now()
        self._lockdown_start: datetime | None = None
        self._cooldown_start: datetime | None = None
        self._detection_count = 0
        self._lockdown_count = 0
        self._last_prey_event = None  # Most recent prey detection event

        # Callbacks
        self._on_state_change_callbacks: list[Callable[[SystemState], None]] = []

        logger.info("MouseHunterController initialized")

    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state

    async def start(self) -> None:
        """Start the MouseHunter system."""
        logger.info("=== Starting MouseHunter System ===")

        # Store main event loop for cross-thread communication
        self._main_loop = asyncio.get_running_loop()

        # Import and set global loop reference for Telegram
        from mousehunter.notifications import telegram_bot
        telegram_bot.MAIN_LOOP = self._main_loop

        # Signal that main loop is ready for cross-thread communication
        self._loop_ready.set()

        # Load configuration
        from mousehunter.config import (
            setup_logging,
            ensure_runtime_dirs,
            telegram_config,
            api_config,
            jammer_config,
        )

        setup_logging()
        ensure_runtime_dirs()

        # Initialize components
        await self._init_components()

        # Register signal handlers
        self._setup_signal_handlers()

        self._running = True
        logger.info("System initialization complete")

        # Start background tasks (store for proper cancellation on shutdown)
        self._background_tasks = []

        # Start Telegram bot if enabled
        if telegram_config.enabled and self._telegram:
            task = asyncio.create_task(self._telegram.start(), name="telegram_bot")
            self._background_tasks.append(task)
            logger.info("Telegram bot task started")

        # Start API server if enabled
        if api_config.enabled:
            from mousehunter.api.server import start_server, set_components

            set_components(
                jammer=self._jammer,
                audio=self._audio,
                camera=self._camera,
                detector=self._detector,
            )
            task = asyncio.create_task(
                start_server(
                    host=api_config.host,
                    port=api_config.port,
                    jammer=self._jammer,
                    audio=self._audio,
                    camera=self._camera,
                    detector=self._detector,
                ),
                name="api_server",
            )
            self._background_tasks.append(task)
            logger.info(f"API server task started on {api_config.host}:{api_config.port}")

        # Start cloud storage periodic sync if enabled
        if self._cloud_storage and self._cloud_storage.enabled:
            from mousehunter.config import cloud_storage_config

            if cloud_storage_config.sync_interval_minutes > 0:
                task = asyncio.create_task(
                    self._cloud_sync_loop(cloud_storage_config.sync_interval_minutes),
                    name="cloud_sync",
                )
                self._background_tasks.append(task)
                logger.info(
                    f"Cloud sync task started (interval: {cloud_storage_config.sync_interval_minutes}min)"
                )

        # Start detection pipeline in background thread
        self._start_detection_thread()

        # Send startup notification
        if self._telegram:
            try:
                await self._telegram.send_message(
                    "MouseHunter Online\n"
                    f"State: {self._state.name}\n"
                    f"Jammer: GPIO {jammer_config.gpio_pin}\n"
                    "Prey detection active."
                )
            except Exception as e:
                logger.error(f"Failed to send startup notification: {e}")

        logger.info("=== MouseHunter System Running ===")

        # Main loop - wait for shutdown
        try:
            while self._running:
                await asyncio.sleep(1)
                await self._state_machine_tick()
        except asyncio.CancelledError:
            logger.info("Main loop cancelled")

        # Cleanup
        await self._shutdown()

    async def _init_components(self) -> None:
        """Initialize all hardware and software components."""
        logger.info("Initializing components...")

        # Hardware: Jammer
        try:
            from mousehunter.hardware.jammer import Jammer
            from mousehunter.config import jammer_config

            self._jammer = Jammer(
                pin=jammer_config.gpio_pin,
                active_high=jammer_config.active_high,
                max_on_duration=jammer_config.lockdown_duration_seconds,
            )
            logger.info("Jammer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize jammer: {e}")

        # Hardware: Audio
        try:
            from mousehunter.hardware.audio import AudioDeterrent
            from mousehunter.config import audio_config, PROJECT_ROOT
            from pathlib import Path

            sound_path = Path(audio_config.sound_file)
            if not sound_path.is_absolute():
                sound_path = PROJECT_ROOT / sound_path

            self._audio = AudioDeterrent(
                sound_file=sound_path if sound_path.exists() else None,
                volume=audio_config.volume,
                enabled=audio_config.enabled,
            )
            logger.info("Audio deterrent initialized")
        except Exception as e:
            logger.error(f"Failed to initialize audio: {e}")

        # Camera
        try:
            from mousehunter.camera.camera_service import CameraService
            from mousehunter.config import camera_config, recording_config

            self._camera = CameraService(
                main_resolution=camera_config.main_resolution,
                inference_resolution=camera_config.inference_resolution,
                framerate=camera_config.framerate,
                buffer_seconds=camera_config.buffer_seconds,
                vflip=camera_config.vflip,
                hflip=camera_config.hflip,
                output_dir=recording_config.output_dir,
            )
            logger.info("Camera service initialized")
        except Exception as e:
            logger.error(f"Failed to initialize camera: {e}")

        # Inference / Prey Detector
        try:
            from mousehunter.inference.prey_detector import PreyDetector
            from mousehunter.config import inference_config

            self._detector = PreyDetector(
                thresholds=inference_config.thresholds,
                window_size=inference_config.window_size,
                trigger_count=inference_config.trigger_count,
                spatial_validation_enabled=inference_config.spatial_validation_enabled,
                box_expansion=inference_config.box_expansion,
            )

            # Register prey detection callback
            self._detector.on_prey_confirmed(self._handle_prey_confirmed)
            logger.info("Prey detector initialized")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {e}")

        # Telegram Bot
        try:
            from mousehunter.notifications.telegram_bot import TelegramBot
            from mousehunter.config import telegram_config

            if telegram_config.enabled:
                self._telegram = TelegramBot(
                    bot_token=telegram_config.bot_token,
                    chat_id=telegram_config.chat_id,
                    enabled=telegram_config.enabled,
                )
                self._telegram.set_components(
                    jammer=self._jammer,
                    audio=self._audio,
                    camera=self._camera,
                )
                logger.info("Telegram bot initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram: {e}")

        # Cloud Storage
        try:
            from mousehunter.storage.cloud_storage import CloudStorage
            from mousehunter.config import cloud_storage_config, recording_config

            if cloud_storage_config.enabled:
                self._cloud_storage = CloudStorage(
                    rclone_remote=cloud_storage_config.rclone_remote,
                    remote_path=cloud_storage_config.remote_path,
                    evidence_dir=recording_config.evidence_dir,
                    enabled=cloud_storage_config.enabled,
                    delete_local_after_upload=cloud_storage_config.delete_local_after_upload,
                    retention_days_local=recording_config.max_age_days,
                    retention_days_cloud=cloud_storage_config.retention_days_cloud,
                    bandwidth_limit=cloud_storage_config.bandwidth_limit,
                )
                logger.info("Cloud storage initialized")
        except Exception as e:
            logger.error(f"Failed to initialize cloud storage: {e}")

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Signal {signum} received, initiating shutdown...")
            self._running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _start_detection_thread(self) -> None:
        """Start the detection pipeline in a background thread.

        Raises:
            RuntimeError: If camera or detector are not available
        """
        missing = []
        if self._camera is None:
            missing.append("camera")
        if self._detector is None:
            missing.append("detector")

        if missing:
            error_msg = f"Cannot start detection: {', '.join(missing)} not available"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        self._stop_event.clear()
        self._detection_thread = threading.Thread(
            target=self._detection_loop,
            name="DetectionThread",
            daemon=True,
        )
        self._detection_thread.start()
        logger.info("Detection thread started")

    def _detection_loop(self) -> None:
        """
        Main detection loop (runs in background thread).

        Captures frames and runs inference continuously.
        """
        logger.info("Detection loop starting")

        # Start camera
        self._camera.start()

        frame_count = 0
        while not self._stop_event.is_set():
            try:
                # Get frame from camera
                frame, timestamp = self._camera.get_inference_frame()

                if frame is None:
                    continue

                # Process through prey detector
                result = self._detector.process_frame(frame, timestamp)
                frame_count += 1

                # Log periodically
                if frame_count % 100 == 0:
                    logger.debug(
                        f"Detection loop: {frame_count} frames, "
                        f"state={self._state.name}, "
                        f"avg_inference={self._detector.engine.average_inference_time:.1f}ms"
                    )

            except Exception as e:
                logger.error(f"Detection loop error: {e}", exc_info=True)

        self._camera.stop()
        logger.info("Detection loop stopped")

    def _handle_prey_confirmed(self, event) -> None:
        """
        Handle prey confirmation from detector (called from detection thread).

        Transitions to LOCKDOWN state and triggers all interdiction actions.
        """
        logger.warning(
            f"PREY DETECTED! {event.prey_detection.class_name} "
            f"({event.prey_detection.confidence:.2f}) with cat "
            f"({event.cat_detection.confidence:.2f}), "
            f"window: {event.detections_in_window}/{event.window_size}"
        )

        self._detection_count += 1
        self._last_prey_event = event  # Store for Telegram notification

        # Capture evidence image
        image_bytes = None
        if event.frame is not None:
            try:
                from PIL import Image
                from io import BytesIO

                img = Image.fromarray(event.frame)
                buffer = BytesIO()
                img.save(buffer, "JPEG", quality=85)
                image_bytes = buffer.getvalue()
            except Exception as e:
                logger.error(f"Failed to encode evidence image: {e}")

        # Save evidence to disk
        evidence_path = None
        if self._camera:
            try:
                evidence_path = self._camera.trigger_evidence_save(
                    f"prey_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                logger.info(f"Evidence saved to: {evidence_path}")
            except Exception as e:
                logger.error(f"Failed to save evidence: {e}")

        # Schedule async actions on main loop
        # Wait for main loop to be ready (with timeout to avoid deadlock)
        if not self._loop_ready.wait(timeout=5.0):
            logger.error("Main loop not ready after 5s, cannot execute lockdown")
            return

        if self._main_loop:
            # Schedule cloud upload (async, non-blocking)
            if self._cloud_storage and evidence_path:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._cloud_storage.upload_evidence(
                            evidence_path,
                            prey_type=event.prey_detection.class_name,
                            confidence=event.prey_detection.confidence,
                            cat_confidence=event.cat_detection.confidence,
                        ),
                        self._main_loop,
                    )
                except Exception as e:
                    logger.error(f"Failed to schedule cloud upload: {e}")
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._execute_lockdown(image_bytes), self._main_loop
                )
                # Don't block waiting for result, but log any errors
                future.add_done_callback(self._lockdown_callback)
            except Exception as e:
                logger.error(f"Failed to schedule lockdown: {e}", exc_info=True)

    def _lockdown_callback(self, future) -> None:
        """Callback to log lockdown execution errors."""
        try:
            future.result()  # Raises if coroutine raised
        except Exception as e:
            logger.error(f"Lockdown execution failed: {e}", exc_info=True)

    async def _execute_lockdown(self, image_bytes: bytes | None = None) -> None:
        """Execute lockdown sequence (async)."""
        from mousehunter.config import jammer_config

        # Transition to LOCKDOWN
        self._transition_to(SystemState.LOCKDOWN)
        self._lockdown_start = datetime.now()
        self._lockdown_count += 1

        # 1. IMMEDIATE: Activate jammer
        if self._jammer:
            await self._jammer.activate_with_auto_off(jammer_config.lockdown_duration_seconds)
            logger.info("Jammer ACTIVATED - Cat flap BLOCKED")

        # 2. Trigger audio deterrent
        if self._audio:
            self._audio.play()
            logger.info("Audio deterrent triggered")

        # 3. Send Telegram notification
        if self._telegram:
            try:
                # Get prey details from the last detection event
                prey_info = getattr(self, '_last_prey_event', None)
                prey_type = prey_info.prey_detection.class_name if prey_info else "prey"
                prey_conf = prey_info.prey_detection.confidence if prey_info else 0.0

                message = (
                    f"PREY DETECTED: {prey_type.upper()}\n"
                    f"Confidence: {prey_conf:.0%}\n"
                    f"Cat flap locked for {jammer_config.lockdown_duration_seconds}s.\n"
                    f"Detection #{self._detection_count}"
                )
                await self._telegram.send_alert(message, image_bytes, include_buttons=True)
            except Exception as e:
                logger.error(f"Failed to send alert: {e}")

    async def _cloud_sync_loop(self, interval_minutes: int) -> None:
        """Periodic cloud sync and local cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(interval_minutes * 60)

                if self._cloud_storage:
                    # Sync any pending uploads
                    logger.info("Running periodic cloud sync...")
                    results = await self._cloud_storage.sync_all()
                    if results:
                        success = sum(1 for v in results.values() if v)
                        logger.info(f"Cloud sync: {success}/{len(results)} folders uploaded")

                    # Cleanup old local files
                    deleted = self._cloud_storage.cleanup_local()
                    if deleted > 0:
                        logger.info(f"Local cleanup: {deleted} folders deleted")

            except asyncio.CancelledError:
                logger.debug("Cloud sync loop cancelled")
                break
            except Exception as e:
                logger.error(f"Cloud sync error: {e}")

    async def _state_machine_tick(self) -> None:
        """Periodic state machine update."""
        from mousehunter.config import jammer_config

        if self._state == SystemState.LOCKDOWN:
            # Check if lockdown duration has elapsed
            if self._lockdown_start and self._jammer:
                if not self._jammer.is_active:
                    # Jammer auto-deactivated, transition to cooldown
                    logger.info("Lockdown complete, entering cooldown")
                    self._transition_to(SystemState.COOLDOWN)
                    self._cooldown_start = datetime.now()

        elif self._state == SystemState.COOLDOWN:
            # Check if cooldown has elapsed
            if self._cooldown_start:
                elapsed = (datetime.now() - self._cooldown_start).total_seconds()
                if elapsed >= jammer_config.cooldown_duration_seconds:
                    logger.info("Cooldown complete, returning to IDLE")
                    self._transition_to(SystemState.IDLE)
                    self._detector.reset()

        elif self._state == SystemState.VERIFYING:
            # Detector handles this internally
            pass

    def _transition_to(self, new_state: SystemState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.now()

        logger.info(f"State: {old_state.name} -> {new_state.name}")

        for callback in self._on_state_change_callbacks:
            try:
                callback(new_state)
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    async def _shutdown(self) -> None:
        """Clean shutdown of all components."""
        logger.info("Initiating shutdown...")

        # Cancel background tasks first (API server, Telegram)
        if self._background_tasks:
            logger.info(f"Cancelling {len(self._background_tasks)} background tasks...")
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for tasks to complete with timeout
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self._background_tasks, return_exceptions=True),
                    timeout=5.0,
                )
                logger.info("Background tasks cancelled successfully")
            except asyncio.TimeoutError:
                logger.warning("Background tasks did not cancel within 5s")
            self._background_tasks.clear()

        # Stop detection thread (before cleanup to avoid race conditions)
        self._stop_event.set()
        if self._detection_thread:
            logger.info("Waiting for detection thread to stop...")
            self._detection_thread.join(timeout=15.0)
            if self._detection_thread.is_alive():
                logger.error(
                    "Detection thread did not stop within 15s! "
                    "Proceeding with cleanup anyway (may cause errors)"
                )

        # Deactivate jammer (safety)
        if self._jammer:
            self._jammer.deactivate("Shutdown")
            self._jammer.cleanup()

        # Stop audio
        if self._audio:
            self._audio.cleanup()

        # Stop camera
        if self._camera:
            self._camera.cleanup()

        # Stop Telegram
        if self._telegram:
            await self._telegram.stop()

        # Cleanup detector
        if self._detector:
            self._detector.cleanup()

        logger.info("Shutdown complete")

    def on_state_change(self, callback: Callable[[SystemState], None]) -> None:
        """Register callback for state changes."""
        self._on_state_change_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get full system status."""
        return {
            "state": self._state.name,
            "running": self._running,
            "last_state_change": self._last_state_change.isoformat(),
            "detection_count": self._detection_count,
            "lockdown_count": self._lockdown_count,
            "jammer": self._jammer.get_status() if self._jammer else None,
            "audio": self._audio.get_status() if self._audio else None,
            "camera": self._camera.get_status() if self._camera else None,
            "detector": self._detector.get_status() if self._detector else None,
        }


# ==================== Entry Point ====================


async def app() -> None:
    """Main application entry point."""
    controller = MouseHunterController()
    await controller.start()


def main() -> None:
    """CLI entry point."""
    print("=== MouseHunter v2.0 ===")
    print("Cat Prey Detection & Interdiction System")
    print("Raspberry Pi 5 + Hailo-8L")
    print()

    try:
        asyncio.run(app())
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
