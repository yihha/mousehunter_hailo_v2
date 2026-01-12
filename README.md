# MouseHunter Hailo v2

Real-time Cat Prey Detection & Interdiction System for Raspberry Pi 5 + Hailo-8L.

## Overview

MouseHunter uses AI-powered computer vision to detect when your cat is carrying prey (mice, birds, etc.) and automatically blocks the cat flap to prevent entry with the catch. When prey is detected, the system:

1. **Blocks the cat flap** using RFID jamming (134.2kHz interference)
2. **Plays a loud sound** (hawk screech) to startle the cat into dropping prey
3. **Sends a Telegram notification** with the detection image
4. **Records video evidence** from the circular buffer

## Hardware Requirements

| Component | Specification |
|-----------|---------------|
| Compute | Raspberry Pi 5 (8GB RAM) |
| NPU | Raspberry Pi AI HAT+ (Hailo-8L, 13 TOPS) |
| Camera | PiCamera 3 NoIR + IR Illumination |
| Jammer | DONGKER 134.2KHz RFID Reader |
| Relay | 5V Single-Channel Relay Module |
| Audio | USB Audio Adapter + Active Speaker |
| Cat Flap | SureFlap Microchip Cat Flap |
| Power | 27W USB-C PD Power Supply |
| Cooling | Raspberry Pi Active Cooler |

### Wiring Diagram

```
Raspberry Pi 5
    |
    |-- GPIO 17 --> Relay IN
    |-- 5V ------> Relay VCC
    |-- GND -----> Relay GND
    |
    |-- CSI ----> PiCamera 3 NoIR
    |
    |-- PCIe ---> AI HAT+ (Hailo-8L)
    |
    |-- USB ----> Audio Adapter --> Speaker

Relay (NO/COM) --> DONGKER 5V Power
```

## Installation

### 1. System Setup (Raspberry Pi OS Bookworm 64-bit)

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable PCIe Gen 3 for Hailo
sudo nano /boot/firmware/config.txt
# Add these lines:
# dtparam=pciex1
# dtparam=pciex1_gen=3

# Reboot
sudo reboot
```

### 2. Install Hailo Runtime

```bash
# Follow Hailo's official instructions for Pi 5
# https://github.com/hailo-ai/hailo-rpi5-examples
```

### 3. Install MouseHunter

```bash
# Clone or copy the project
cd /home/pi
git clone <repository> mousehunter_hailo_v2
cd mousehunter_hailo_v2

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### 4. Configure

```bash
# Copy environment template
cp .env.example .env

# Edit with your Telegram credentials
nano .env

# Or edit config directly
nano config/config.json
```

### 5. Add Sound File

Place a WAV file at `sounds/hawk_screech.wav` for the audio deterrent.

### 6. Add YOLOv8 Model

Place your compiled Hailo model at `models/yolov8n_catprey.hef`.

To create the model:
1. Train YOLOv8n on cat-with-prey dataset
2. Export to ONNX
3. Compile with Hailo Dataflow Compiler

### 7. Test Hardware

```bash
# Test jammer relay
python -m mousehunter.hardware.jammer

# Test audio
python -m mousehunter.hardware.audio

# Test camera
python -m mousehunter.camera.camera_service
```

### 8. Run

```bash
# Direct run
python -m mousehunter.main

# Or use the CLI entry point
mousehunter
```

### 9. Install as Service

```bash
# Copy service file
sudo cp systemd/mousehunter.service /etc/systemd/system/

# Edit paths if needed
sudo nano /etc/systemd/system/mousehunter.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mousehunter
sudo systemctl start mousehunter

# Check status
sudo systemctl status mousehunter

# View logs
journalctl -u mousehunter -f
```

## Configuration

Edit `config/config.json` or use environment variables:

```json
{
    "telegram": {
        "bot_token": "YOUR_BOT_TOKEN",
        "chat_id": "YOUR_CHAT_ID",
        "enabled": true
    },
    "jammer": {
        "gpio_pin": 17,
        "lockdown_duration_seconds": 300
    },
    "inference": {
        "confidence_threshold": 0.60,
        "consecutive_frames_required": 3
    }
}
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | System health check |
| `/photo` | Capture camera snapshot |
| `/lock` | Manually lock cat flap |
| `/unlock` | Manually unlock cat flap |
| `/scream` | Trigger audio deterrent |
| `/help` | Show commands |

## REST API

When enabled, the API server provides HTTP endpoints:

```
GET  /status          - Full system status
GET  /health          - Health check
POST /jammer/lock     - Lock cat flap
POST /jammer/unlock   - Unlock cat flap
POST /audio/scream    - Trigger audio
GET  /camera/snapshot - Get camera image
```

Default: `http://localhost:8080`

## State Machine

```
IDLE -----> VERIFYING -----> LOCKDOWN -----> COOLDOWN -----> IDLE
  ^              |                |                             |
  |              |                |                             |
  +--------------+----------------+-----------------------------+
       (no prey)      (auto-timeout)        (cooldown complete)
```

- **IDLE**: Normal monitoring, processing frames
- **VERIFYING**: Potential prey detected, requiring N consecutive frames
- **LOCKDOWN**: Prey confirmed, jammer active, notifications sent
- **COOLDOWN**: Post-lockdown period before returning to normal

## Architecture

```
main.py (Async Controller)
    |
    +-- camera/ (Dual-stream PiCamera2)
    |       |-- camera_service.py
    |       +-- circular_buffer.py
    |
    +-- inference/ (Hailo-8L)
    |       |-- hailo_engine.py
    |       |-- prey_detector.py
    |       +-- detection.py
    |
    +-- hardware/
    |       |-- jammer.py (GPIO Relay)
    |       +-- audio.py (USB Audio)
    |
    +-- notifications/
    |       +-- telegram_bot.py
    |
    +-- api/
            +-- server.py (FastAPI)
```

## Troubleshooting

### Camera not working
```bash
# Check camera is detected
libcamera-hello -t 0

# Check permissions
sudo usermod -aG video $USER
```

### Hailo not detected
```bash
# Check PCIe device
lspci | grep Hailo

# Check kernel module
dmesg | grep hailo
```

### GPIO permission denied
```bash
# Add user to gpio group
sudo usermod -aG gpio $USER
```

### Jammer not blocking
- Check relay wiring (NO vs NC)
- Verify DONGKER antenna placement near cat flap
- Test with `python -m mousehunter.hardware.jammer`

## License

MIT License

## Acknowledgments

- [Hailo](https://hailo.ai/) for the AI accelerator
- [python-telegram-bot](https://python-telegram-bot.org/) for Telegram integration
- Original catflap-prey-detector project for inspiration
