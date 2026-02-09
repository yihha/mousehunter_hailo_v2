"""
Hardware Abstraction Layer for MouseHunter.

Provides interfaces for:
- Jammer: RFID jammer control via GPIO relay
- AudioDeterrent: Sound playback for startling cats
"""

from .jammer import Jammer, get_jammer
from .audio import AudioDeterrent, get_audio_deterrent

__all__ = ["Jammer", "get_jammer", "AudioDeterrent", "get_audio_deterrent"]
