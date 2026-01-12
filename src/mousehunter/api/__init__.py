"""
API module for MouseHunter.

Provides:
- FastAPI server for remote control
- REST endpoints for status, control, and monitoring
"""

from .server import create_app, start_server

__all__ = ["create_app", "start_server"]
