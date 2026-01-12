"""
Hardware Abstraction Layer for MouseHunter.

Provides interfaces for:
- Jammer: RFID jammer control via GPIO relay
- AudioDeterrent: Sound playback for startling cats
"""

from .jammer import Jammer, jammer
from .audio import AudioDeterrent, audio_deterrent

__all__ = ["Jammer", "jammer", "AudioDeterrent", "audio_deterrent"]
