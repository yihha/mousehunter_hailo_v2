"""
FastAPI Server - REST API for remote control

Provides HTTP endpoints for:
- System status and health checks
- Manual jammer control
- Audio deterrent trigger
- Camera snapshots

Security: Designed for local network or VPN (Tailscale) access.
Do NOT expose directly to the internet without authentication.
"""

import asyncio
import logging
from datetime import datetime
from io import BytesIO
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available")

try:
    import uvicorn

    UVICORN_AVAILABLE = True
except ImportError:
    UVICORN_AVAILABLE = False


# Pydantic models for API
if FASTAPI_AVAILABLE:

    class StatusResponse(BaseModel):
        """System status response."""

        timestamp: str
        system: dict[str, Any]
        jammer: dict[str, Any] | None
        camera: dict[str, Any] | None
        inference: dict[str, Any] | None

    class ActionResponse(BaseModel):
        """Action result response."""

        success: bool
        message: str
        timestamp: str

    class ControlRequest(BaseModel):
        """Control request body."""

        duration: float | None = None


# Global component references
_jammer = None
_audio = None
_camera = None
_detector = None


def set_components(jammer=None, audio=None, camera=None, detector=None) -> None:
    """Set references to system components."""
    global _jammer, _audio, _camera, _detector
    _jammer = jammer
    _audio = audio
    _camera = camera
    _detector = detector


def create_app() -> "FastAPI":
    """Create and configure the FastAPI application."""
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI not available")

    app = FastAPI(
        title="MouseHunter API",
        description="REST API for Cat Prey Detection & Interdiction System",
        version="2.0.0",
    )

    # ==================== Status Endpoints ====================

    @app.get("/", response_model=dict)
    async def root():
        """Root endpoint - basic health check."""
        return {
            "service": "MouseHunter",
            "version": "2.0.0",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    @app.get("/status", response_model=StatusResponse)
    async def get_status():
        """Get full system status."""
        import psutil

        system_status = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "uptime_seconds": _get_uptime(),
        }

        # Try to get CPU temperature
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                system_status["cpu_temp_c"] = int(f.read()) / 1000
        except Exception:
            system_status["cpu_temp_c"] = None

        return StatusResponse(
            timestamp=datetime.now().isoformat(),
            system=system_status,
            jammer=_jammer.get_status() if _jammer else None,
            camera=_camera.get_status() if _camera else None,
            inference=_detector.get_status() if _detector else None,
        )

    # ==================== Jammer Control ====================

    @app.post("/jammer/lock", response_model=ActionResponse)
    async def lock_jammer(request: ControlRequest = None):
        """Activate the RFID jammer (lock cat flap)."""
        if not _jammer:
            raise HTTPException(status_code=503, detail="Jammer not available")

        duration = request.duration if request else None

        if duration:
            success = await _jammer.activate_with_auto_off(duration)
        else:
            success = _jammer.activate()

        if success:
            return ActionResponse(
                success=True,
                message=f"Jammer activated, {_jammer.time_remaining:.0f}s remaining",
                timestamp=datetime.now().isoformat(),
            )
        else:
            return ActionResponse(
                success=False,
                message=f"Jammer already active, {_jammer.time_remaining:.0f}s remaining",
                timestamp=datetime.now().isoformat(),
            )

    @app.post("/jammer/unlock", response_model=ActionResponse)
    async def unlock_jammer():
        """Deactivate the RFID jammer (unlock cat flap)."""
        if not _jammer:
            raise HTTPException(status_code=503, detail="Jammer not available")

        success = _jammer.deactivate("API request")

        return ActionResponse(
            success=success,
            message="Jammer deactivated" if success else "Jammer was not active",
            timestamp=datetime.now().isoformat(),
        )

    @app.get("/jammer/status")
    async def jammer_status():
        """Get jammer status."""
        if not _jammer:
            raise HTTPException(status_code=503, detail="Jammer not available")
        return _jammer.get_status()

    # ==================== Audio Control ====================

    @app.post("/audio/scream", response_model=ActionResponse)
    async def trigger_scream():
        """Trigger the audio deterrent."""
        if not _audio:
            raise HTTPException(status_code=503, detail="Audio not available")

        success = _audio.play()

        return ActionResponse(
            success=success,
            message="Scream triggered!" if success else "Audio failed or already playing",
            timestamp=datetime.now().isoformat(),
        )

    @app.post("/audio/stop", response_model=ActionResponse)
    async def stop_audio():
        """Stop audio playback."""
        if not _audio:
            raise HTTPException(status_code=503, detail="Audio not available")

        _audio.stop()

        return ActionResponse(
            success=True,
            message="Audio stopped",
            timestamp=datetime.now().isoformat(),
        )

    @app.get("/audio/status")
    async def audio_status():
        """Get audio system status."""
        if not _audio:
            raise HTTPException(status_code=503, detail="Audio not available")
        return _audio.get_status()

    # ==================== Camera Endpoints ====================

    @app.get("/camera/snapshot")
    async def camera_snapshot():
        """Capture and return a camera snapshot."""
        if not _camera:
            raise HTTPException(status_code=503, detail="Camera not available")

        image_bytes = _camera.capture_snapshot_bytes(quality=85)

        if not image_bytes:
            raise HTTPException(status_code=500, detail="Failed to capture image")

        return Response(
            content=image_bytes,
            media_type="image/jpeg",
            headers={"Content-Disposition": 'inline; filename="snapshot.jpg"'},
        )

    @app.get("/camera/status")
    async def camera_status():
        """Get camera status."""
        if not _camera:
            raise HTTPException(status_code=503, detail="Camera not available")
        return _camera.get_status()

    # ==================== Detection Endpoints ====================

    @app.get("/detection/status")
    async def detection_status():
        """Get detection system status."""
        if not _detector:
            raise HTTPException(status_code=503, detail="Detector not available")
        return _detector.get_status()

    @app.get("/detection/history")
    async def detection_history(count: int = 10):
        """Get recent detection history."""
        if not _detector:
            raise HTTPException(status_code=503, detail="Detector not available")

        frames = _detector.get_recent_detections(count)
        return {"detections": [f.to_dict() for f in frames]}

    @app.post("/detection/reset", response_model=ActionResponse)
    async def reset_detection():
        """Reset detection state to IDLE."""
        if not _detector:
            raise HTTPException(status_code=503, detail="Detector not available")

        _detector.reset()

        return ActionResponse(
            success=True,
            message="Detection state reset to IDLE",
            timestamp=datetime.now().isoformat(),
        )

    return app


def _get_uptime() -> float:
    """Get system uptime in seconds."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


async def start_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    jammer=None,
    audio=None,
    camera=None,
    detector=None,
) -> None:
    """
    Start the API server.

    Args:
        host: Bind host
        port: Bind port
        jammer: Jammer controller instance
        audio: Audio deterrent instance
        camera: Camera service instance
        detector: Prey detector instance
    """
    if not FASTAPI_AVAILABLE or not UVICORN_AVAILABLE:
        logger.error("FastAPI or uvicorn not available")
        return

    # Set component references
    set_components(jammer, audio, camera, detector)

    # Create app
    app = create_app()

    # Configure uvicorn
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True,
    )

    server = uvicorn.Server(config)

    logger.info(f"Starting API server on {host}:{port}")
    await server.serve()


def test_server() -> None:
    """Test the API server standalone."""
    logging.basicConfig(level=logging.INFO)
    print("=== API Server Test ===")
    print(f"FastAPI Available: {FASTAPI_AVAILABLE}")
    print(f"Uvicorn Available: {UVICORN_AVAILABLE}")

    if FASTAPI_AVAILABLE and UVICORN_AVAILABLE:
        app = create_app()
        uvicorn.run(app, host="127.0.0.1", port=8080)


if __name__ == "__main__":
    test_server()
