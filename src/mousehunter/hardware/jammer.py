"""
RFID Jammer Controller

Controls the DONGKER 134.2KHz RFID module via a 5V relay.
The jammer creates electromagnetic interference that prevents the SureFlap
cat flap from reading the cat's microchip, blocking entry.

Hardware:
- GPIO 17 (BCM) -> Relay IN
- Relay 5V -> DONGKER VCC
- GPIO HIGH = Relay CLOSED = Jammer ON = Door BLOCKED

Safety Features:
- Auto-off watchdog timer (max 60 seconds per activation by default)
- Graceful cleanup on shutdown
- Thread-safe state management
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Callable

logger = logging.getLogger(__name__)

# Conditional import for non-Pi development
try:
    from gpiozero import OutputDevice
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("gpiozero not available - running in simulation mode")


class MockOutputDevice:
    """Mock GPIO device for development/testing without hardware."""

    def __init__(self, pin: int, active_high: bool = True, initial_value: bool = False):
        self.pin = pin
        self.active_high = active_high
        self._value = initial_value
        logger.info(f"[MOCK] GPIO {pin} initialized (active_high={active_high})")

    def on(self) -> None:
        self._value = True
        logger.info(f"[MOCK] GPIO {self.pin} -> HIGH (Jammer ON)")

    def off(self) -> None:
        self._value = False
        logger.info(f"[MOCK] GPIO {self.pin} -> LOW (Jammer OFF)")

    def close(self) -> None:
        self._value = False
        logger.info(f"[MOCK] GPIO {self.pin} closed")

    @property
    def value(self) -> bool:
        return self._value


class Jammer:
    """
    RFID Jammer controller with safety features.

    Controls a relay connected to a 134.2kHz RFID reader module.
    When activated, the reader's carrier wave interferes with the
    SureFlap's polling, preventing microchip recognition.
    """

    def __init__(
        self,
        pin: int = 17,
        active_high: bool = True,
        max_on_duration: float = 600.0,  # 10 minutes max
    ):
        """
        Initialize the jammer controller.

        Args:
            pin: BCM GPIO pin number (default 17)
            active_high: True if HIGH activates relay (default True)
            max_on_duration: Maximum time jammer can be ON before auto-off (seconds)
        """
        self.pin = pin
        self.active_high = active_high
        self.max_on_duration = max_on_duration

        # State tracking
        self._is_active = False
        self._activation_time: datetime | None = None
        self._lock = threading.Lock()
        self._watchdog_task: asyncio.Task | None = None

        # Callbacks for state change notifications
        self._on_activate_callbacks: list[Callable[[], None]] = []
        self._on_deactivate_callbacks: list[Callable[[], None]] = []

        # Initialize GPIO
        if GPIO_AVAILABLE:
            self._device = OutputDevice(pin, active_high=active_high, initial_value=False)
        else:
            self._device = MockOutputDevice(pin, active_high=active_high, initial_value=False)

        logger.info(
            f"Jammer initialized on GPIO {pin} (active_high={active_high}, "
            f"max_duration={max_on_duration}s)"
        )

    @property
    def is_active(self) -> bool:
        """Check if jammer is currently active."""
        return self._is_active

    @property
    def activation_time(self) -> datetime | None:
        """Get the time when jammer was activated."""
        return self._activation_time

    @property
    def time_remaining(self) -> float:
        """Get remaining time before auto-off (seconds). Returns 0 if inactive."""
        if not self._is_active or not self._activation_time:
            return 0.0
        elapsed = (datetime.now() - self._activation_time).total_seconds()
        return max(0.0, self.max_on_duration - elapsed)

    def activate(self, refresh_timer: bool = False) -> bool:
        """
        Activate the jammer (block cat flap).

        Args:
            refresh_timer: If True, reset the activation time if already active.

        Returns:
            True if activation was successful, False if already active
        """
        with self._lock:
            if self._is_active:
                if refresh_timer:
                    self._activation_time = datetime.now()
                    logger.info("Jammer timer refreshed")
                    return True
                remaining = self.time_remaining
                logger.info(f"Jammer already active ({remaining:.1f}s remaining)")
                return False

            self._device.on()
            self._is_active = True
            self._activation_time = datetime.now()

            logger.info(f"JAMMER ACTIVATED - Cat flap BLOCKED")

            # Notify callbacks
            for callback in self._on_activate_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Activation callback error: {e}")

            return True

    def deactivate(self, reason: str = "Manual") -> bool:
        """
        Deactivate the jammer (unblock cat flap).

        Args:
            reason: Reason for deactivation (for logging)

        Returns:
            True if deactivation was successful, False if already inactive
        """
        with self._lock:
            if not self._is_active:
                logger.debug("Jammer already inactive")
                return False

            self._device.off()

            # Cancel any pending watchdog task
            if self._watchdog_task and not self._watchdog_task.done():
                self._watchdog_task.cancel()
                self._watchdog_task = None

            # Calculate duration
            duration = 0.0
            if self._activation_time:
                duration = (datetime.now() - self._activation_time).total_seconds()

            self._is_active = False
            self._activation_time = None

            logger.info(f"JAMMER DEACTIVATED - Reason: {reason} (was active {duration:.1f}s)")

            # Notify callbacks
            for callback in self._on_deactivate_callbacks:
                try:
                    callback()
                except Exception as e:
                    logger.error(f"Deactivation callback error: {e}")

            return True

    async def activate_with_auto_off(self, duration: float | None = None) -> bool:
        """
        Activate jammer and schedule automatic deactivation.

        Args:
            duration: Custom duration (uses max_on_duration if None)

        Returns:
            True if activation was successful
        """
        if not self.activate(refresh_timer=True):
            return False

        duration = duration or self.max_on_duration

        # Cancel existing watchdog
        if self._watchdog_task and not self._watchdog_task.done():
            self._watchdog_task.cancel()

        # Start new watchdog
        self._watchdog_task = asyncio.create_task(self._auto_off_watchdog(duration))
        logger.info(f"Auto-off scheduled in {duration:.1f}s")

        return True

    async def _auto_off_watchdog(self, duration: float) -> None:
        """Background task to automatically deactivate jammer."""
        try:
            await asyncio.sleep(duration)
            self.deactivate(reason="Auto-off timer")
        except asyncio.CancelledError:
            logger.debug("Auto-off watchdog cancelled")

    def on_activate(self, callback: Callable[[], None]) -> None:
        """Register callback for jammer activation."""
        self._on_activate_callbacks.append(callback)

    def on_deactivate(self, callback: Callable[[], None]) -> None:
        """Register callback for jammer deactivation."""
        self._on_deactivate_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get detailed jammer status."""
        return {
            "is_active": self._is_active,
            "gpio_pin": self.pin,
            "activation_time": self._activation_time.isoformat() if self._activation_time else None,
            "time_remaining_seconds": self.time_remaining,
            "max_duration_seconds": self.max_on_duration,
        }

    def cleanup(self) -> None:
        """Clean up GPIO resources."""
        with self._lock:
            if self._is_active:
                self.deactivate(reason="Cleanup/Shutdown")
            self._device.close()
            logger.info("Jammer GPIO cleaned up")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# Import config here to avoid circular imports
def _create_default_jammer() -> Jammer:
    """Create jammer instance from config."""
    try:
        from mousehunter.config import jammer_config

        return Jammer(
            pin=jammer_config.gpio_pin,
            active_high=jammer_config.active_high,
            max_on_duration=jammer_config.lockdown_duration_seconds,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return Jammer()


# Global jammer instance (lazy initialization)
_jammer_instance: Jammer | None = None


def get_jammer() -> Jammer:
    """Get or create the global jammer instance."""
    global _jammer_instance
    if _jammer_instance is None:
        _jammer_instance = _create_default_jammer()
    return _jammer_instance




def test_jammer() -> None:
    """Test the jammer hardware (CLI entry point)."""
    import time

    logging.basicConfig(level=logging.INFO)
    print("=== Jammer Hardware Test ===")
    print(f"GPIO Available: {GPIO_AVAILABLE}")

    # Load from config instead of hardcoding
    j = _create_default_jammer()
    print(f"Loaded from config: GPIO pin={j.pin}, active_high={j.active_high}, max_duration={j.max_on_duration}s")

    try:
        input("Press Enter to ACTIVATE jammer (block cat flap)...")
        j.activate()
        print(f"Jammer active: {j.is_active}")
        print(f"Status: {j.get_status()}")

        time.sleep(2)

        input("Press Enter to DEACTIVATE jammer (unblock cat flap)...")
        j.deactivate(reason="Manual test")
        print(f"Jammer active: {j.is_active}")

    except KeyboardInterrupt:
        print("\nTest interrupted")
    finally:
        j.cleanup()
        print("Test complete")


if __name__ == "__main__":
    test_jammer()
