"""
Audio Deterrent Controller

Plays a startling sound (hawk screech, loud noise) through a USB audio
adapter to deter the cat from entering with prey.

Requirements:
- USB Audio Adapter (DAC) connected to Pi 5
- Active (powered) speaker connected to 3.5mm output
- pygame for audio playback
"""

import asyncio
import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Conditional import for headless development
# Note: pygame.mixer.init() is deferred to first use for thread safety
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError as e:
    PYGAME_AVAILABLE = False
    logger.warning(f"pygame not available: {e} - audio will be simulated")

# Module-level lock for pygame.mixer operations (pygame is not thread-safe)
_pygame_lock = threading.Lock()
_pygame_initialized = False


def _ensure_pygame_initialized() -> bool:
    """Initialize pygame.mixer if not already done (thread-safe)."""
    global _pygame_initialized
    if not PYGAME_AVAILABLE:
        return False

    with _pygame_lock:
        if not _pygame_initialized:
            try:
                pygame.mixer.init()
                _pygame_initialized = True
                logger.info("pygame mixer initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pygame mixer: {e}")
                return False
    return True


class AudioDeterrent:
    """
    Audio deterrent system for startling cats.

    Plays a loud, startling sound (hawk screech, etc.) when prey
    is detected to make the cat drop its catch.
    """

    def __init__(
        self,
        sound_file: str | Path | None = None,
        volume: float = 0.8,
        enabled: bool = True,
    ):
        """
        Initialize the audio deterrent.

        Args:
            sound_file: Path to the sound file (WAV recommended)
            volume: Playback volume (0.0 to 1.0)
            enabled: Whether audio playback is enabled
        """
        self.enabled = enabled
        self._volume = max(0.0, min(1.0, volume))
        self._sound_file: Path | None = None
        self._sound: "pygame.mixer.Sound | None" = None
        self._is_playing = False
        self._lock = threading.Lock()

        # Callbacks
        self._on_play_callbacks: list[Callable[[], None]] = []

        if sound_file:
            self.load_sound(sound_file)

        logger.info(f"AudioDeterrent initialized (enabled={enabled}, volume={volume})")

    @property
    def volume(self) -> float:
        """Get current volume level."""
        return self._volume

    @volume.setter
    def volume(self, value: float) -> None:
        """Set volume level (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, value))
        if self._sound and _pygame_initialized:
            with _pygame_lock:
                try:
                    self._sound.set_volume(self._volume)
                except Exception as e:
                    logger.error(f"Error setting volume: {e}")
        logger.info(f"Volume set to {self._volume}")

    @property
    def is_playing(self) -> bool:
        """Check if sound is currently playing."""
        return self._is_playing

    @property
    def sound_loaded(self) -> bool:
        """Check if a sound file is loaded."""
        return self._sound is not None

    def load_sound(self, sound_file: str | Path) -> bool:
        """
        Load a sound file for playback.

        Args:
            sound_file: Path to WAV/OGG/MP3 file

        Returns:
            True if loaded successfully
        """
        sound_path = Path(sound_file)

        if not sound_path.exists():
            logger.error(f"Sound file not found: {sound_path}")
            return False

        self._sound_file = sound_path

        if _ensure_pygame_initialized():
            with _pygame_lock:
                try:
                    self._sound = pygame.mixer.Sound(str(sound_path))
                    self._sound.set_volume(self._volume)
                    logger.info(f"Loaded sound: {sound_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to load sound {sound_path}: {e}")
                    self._sound = None
                    return False
        else:
            logger.info(f"[MOCK] Would load sound: {sound_path}")
            return True

    def play(self, loops: int = 0) -> bool:
        """
        Play the deterrent sound.

        Args:
            loops: Number of times to repeat (-1 for infinite, 0 for once)

        Returns:
            True if playback started
        """
        if not self.enabled:
            logger.debug("Audio disabled, skipping playback")
            return False

        with self._lock:
            if self._is_playing:
                logger.debug("Already playing, ignoring play request")
                return False

            if _ensure_pygame_initialized() and self._sound:
                with _pygame_lock:
                    try:
                        self._sound.play(loops=loops)
                        self._is_playing = True
                        logger.info("SCREAM! Audio deterrent triggered")
                    except Exception as e:
                        logger.error(f"Failed to play sound: {e}")
                        return False

                # Notify callbacks (outside pygame lock to avoid deadlock)
                for callback in self._on_play_callbacks:
                    try:
                        callback()
                    except Exception as e:
                        logger.error(f"Play callback error: {e}")

                return True
            else:
                # Simulation mode
                self._is_playing = True
                logger.info("[MOCK] SCREAM! Audio deterrent triggered")
                return True

    def stop(self) -> None:
        """Stop any currently playing sound."""
        with self._lock:
            if _pygame_initialized and self._sound:
                with _pygame_lock:
                    try:
                        self._sound.stop()
                    except Exception as e:
                        logger.error(f"Error stopping sound: {e}")
            self._is_playing = False
            logger.debug("Audio playback stopped")

    async def play_async(self, duration: float | None = None) -> None:
        """
        Play sound asynchronously with optional duration limit.

        Args:
            duration: Maximum playback duration (seconds), None for full sound
        """
        self.play()

        if duration:
            await asyncio.sleep(duration)
            self.stop()
        else:
            # Wait for sound to finish (if we can determine length)
            if PYGAME_AVAILABLE and self._sound:
                length = self._sound.get_length()
                await asyncio.sleep(length)
            else:
                await asyncio.sleep(2.0)  # Default mock duration

        self._is_playing = False

    def on_play(self, callback: Callable[[], None]) -> None:
        """Register callback for when sound plays."""
        self._on_play_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get audio system status."""
        return {
            "enabled": self.enabled,
            "is_playing": self._is_playing,
            "volume": self._volume,
            "sound_file": str(self._sound_file) if self._sound_file else None,
            "sound_loaded": self.sound_loaded,
            "pygame_available": PYGAME_AVAILABLE,
        }

    def cleanup(self) -> None:
        """Clean up audio resources."""
        global _pygame_initialized
        self.stop()
        if _pygame_initialized:
            with _pygame_lock:
                try:
                    pygame.mixer.quit()
                    _pygame_initialized = False
                except Exception as e:
                    logger.error(f"Error cleaning up pygame mixer: {e}")
        logger.info("Audio resources cleaned up")


# Factory function to create from config
def _create_default_audio() -> AudioDeterrent:
    """Create audio deterrent from config."""
    try:
        from mousehunter.config import audio_config, PROJECT_ROOT

        sound_path = Path(audio_config.sound_file)
        if not sound_path.is_absolute():
            sound_path = PROJECT_ROOT / sound_path

        return AudioDeterrent(
            sound_file=sound_path if sound_path.exists() else None,
            volume=audio_config.volume,
            enabled=audio_config.enabled,
        )
    except ImportError:
        logger.warning("Config not available, using defaults")
        return AudioDeterrent(enabled=True, volume=0.8)


# Global instance (lazy)
_audio_instance: AudioDeterrent | None = None


def get_audio_deterrent() -> AudioDeterrent:
    """Get or create the global audio deterrent instance."""
    global _audio_instance
    if _audio_instance is None:
        _audio_instance = _create_default_audio()
    return _audio_instance


# Convenience alias
audio_deterrent = property(lambda self: get_audio_deterrent())


def test_audio() -> None:
    """Test audio playback (CLI)."""
    import time

    logging.basicConfig(level=logging.INFO)
    print("=== Audio Deterrent Test ===")
    print(f"Pygame Available: {PYGAME_AVAILABLE}")

    audio = AudioDeterrent(volume=0.5, enabled=True)

    # Try to load a test sound
    test_sounds = [
        Path(__file__).parent.parent.parent.parent / "sounds" / "hawk_screech.wav",
        Path("/usr/share/sounds/alsa/Front_Center.wav"),  # Common Linux test sound
    ]

    for sound in test_sounds:
        if sound.exists():
            audio.load_sound(sound)
            break
    else:
        print("No test sound found, will simulate")

    print(f"Status: {audio.get_status()}")

    input("Press Enter to play sound...")
    audio.play()
    time.sleep(3)
    audio.stop()

    print("Test complete")
    audio.cleanup()


if __name__ == "__main__":
    test_audio()
