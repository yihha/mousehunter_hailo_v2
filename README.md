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

# Enable PCIe for Hailo (Gen 2 recommended for stability)
sudo nano /boot/firmware/config.txt
# Add these lines:
# dtparam=pciex1
# dtparam=pciex1_gen=2

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
        "model_path": "models/yolov8n_catprey.hef",
        "classes": {"0": "cat", "1": "rodent"},
        "reg_max": 8,
        "thresholds": {
            "cat": 0.60,
            "rodent": 0.35
        },
        "spatial_validation": {
            "enabled": true,
            "box_expansion": 0.25
        },
        "prey_confirmation": {
            "mode": "score_accumulation",
            "window_seconds": 3.0,
            "score_threshold": 0.9,
            "min_detection_score": 0.20,
            "reset_on_cat_lost_seconds": 1.5
        }
    },
    "cloud_storage": {
        "enabled": false,
        "rclone_remote": "",
        "remote_path": "MouseHunter"
    },
    "training_data": {
        "enabled": false,
        "periodic_interval_minutes": 30,
        "capture_cat_only": true,
        "capture_near_miss": true,
        "max_images_per_day": 100
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

### Cat-as-Anchor Strategy
The system uses a "cat-as-anchor" approach - prey is only confirmed when detected near a cat:

- **IDLE**: No cat detected, waiting for cat (prey detector internally monitors for cats)
- **VERIFYING**: Prey detected near cat, accumulating confidence score
- **LOCKDOWN**: Prey confirmed (score ≥ 0.9), jammer active, notifications sent
- **COOLDOWN**: Post-lockdown period before returning to normal

### Score Accumulation
Instead of requiring consecutive frames, the system uses time-based score accumulation:
1. Each prey detection within a 3-second window adds its confidence to the accumulated score
2. When accumulated score reaches threshold (0.9), prey is confirmed
3. If cat is lost for 1.5 seconds, score resets to zero

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
    |       |-- prey_detector.py (State machine + score accumulation)
    |       +-- detection.py
    |
    +-- hardware/
    |       |-- jammer.py (GPIO Relay)
    |       +-- audio.py (USB Audio)
    |
    +-- storage/
    |       |-- cloud_storage.py (rclone integration)
    |       +-- training_data.py (Capture for model improvement)
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

### Hailo crashes or system instability
PCIe Gen 3 can cause instability on Raspberry Pi 5. Switch to Gen 2:
```bash
# Edit /boot/firmware/config.txt
# Change dtparam=pciex1_gen=3 to:
dtparam=pciex1_gen=2
```
Gen 2 provides sufficient bandwidth for YOLOv8n inference at 30fps. The performance difference is negligible for this workload.

### GPIO permission denied
```bash
# Add user to gpio group
sudo usermod -aG gpio $USER
```

### Jammer not blocking
- Check relay wiring (NO vs NC)
- Verify DONGKER antenna placement near cat flap
- Test with `python -m mousehunter.hardware.jammer`

## Useful Commands

### Service Management
```bash
# Start/stop/restart service
sudo systemctl start mousehunter
sudo systemctl stop mousehunter
sudo systemctl restart mousehunter

# Check service status
sudo systemctl status mousehunter

# View live logs
journalctl -u mousehunter -f

# View recent logs (last 100 lines)
journalctl -u mousehunter -n 100

# Enable/disable auto-start on boot
sudo systemctl enable mousehunter
sudo systemctl disable mousehunter
```

### API Endpoints
```bash
# Get system status
curl -s http://localhost:8080/status | python -m json.tool

# Health check
curl http://localhost:8080/health

# Lock cat flap manually
curl -X POST http://localhost:8080/jammer/lock

# Unlock cat flap
curl -X POST http://localhost:8080/jammer/unlock

# Trigger audio deterrent
curl -X POST http://localhost:8080/audio/scream

# Get camera snapshot
curl http://localhost:8080/camera/snapshot --output snapshot.jpg
```

### Manual Testing
```bash
# Activate virtual environment
cd ~/mousehunter_hailo_v2
source venv/bin/activate

# Test Hailo inference only
python -c "
from mousehunter.inference.hailo_engine import HailoEngine
import numpy as np
engine = HailoEngine(model_path='models/yolov8n.hef', confidence_threshold=0.5)
frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
result = engine.infer(frame)
print(f'Inference: {result.inference_time_ms:.1f}ms')
engine.cleanup()
"

# Test camera + Hailo
python -c "
from mousehunter.inference.hailo_engine import HailoEngine
from picamera2 import Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={'size': (640, 640), 'format': 'RGB888'})
picam2.configure(config)
picam2.start()
engine = HailoEngine(model_path='models/yolov8n.hef', confidence_threshold=0.3)
for i in range(5):
    frame = picam2.capture_array()
    result = engine.infer(frame)
    print(f'Frame {i+1}: {result.inference_time_ms:.1f}ms, {len(result.detections)} detections')
picam2.stop()
engine.cleanup()
"
```

### Hardware Diagnostics
```bash
# Check Hailo device
hailortcli fw-control identify

# Check PCIe
lspci | grep Hailo

# Check camera
libcamera-hello -t 5000

# Test Hailo with official example
rpicam-hello -t 5000 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json

# Check GPIO
pinctrl get 17
```

## Custom Model Training & Deployment

### Overview
The system uses a custom-trained YOLOv8n model optimized for cat and prey detection. The model is trained with `reg_max=8` for better INT8 quantization on Hailo-8L.

### Current Model (v3)
| Class ID | Name   |
|----------|--------|
| 0        | cat    |
| 1        | rodent |

**Training Results:**
- mAP50: 0.907
- Cat mAP50: 0.978
- Rodent mAP50: 0.796

### Step 1: Train Model (Google Colab)

1. Prepare dataset with labeled images (cat, rodent)
2. Train YOLOv8n with reg_max=8 in Google Colab:
```python
from ultralytics import YOLO

# Load pretrained YOLOv8n
model = YOLO('yolov8n.pt')

# Patch Detect head for reg_max=8
model.model.model[-1].reg_max = 8

# Train on custom dataset
model.train(data='cat_prey_dataset/data.yaml', epochs=100, imgsz=640)
```

3. Export to ONNX with 6 separate output tensors (for Hailo):
   - Use graph surgery to expose cv2 (bbox) and cv3 (class) outputs
   - See `hailo_build/YOLOv8_CatPrey_Training_for_Hailo_v4.ipynb` for details

4. Download files from Google Drive:
   - `yolov8n_catrodent_reg8_6outputs.onnx` (ONNX with 6 outputs)

### Step 2: Compile for Hailo-8L

The ONNX model must be compiled to HEF format using Hailo Dataflow Compiler (DFC).

**Option A: Use Hailo Model Zoo (Recommended)**
```bash
# On a machine with Hailo DFC installed
hailo optimize yolov8n_catprey.onnx --hw-arch hailo8l --calib-set /path/to/calibration/images
hailo compile yolov8n_catprey_optimized.har --hw-arch hailo8l
```

**Option B: Use Hailo's Docker Container**
```bash
# Pull Hailo DFC container
docker pull hailo/hailo_dfc:latest

# Run compilation
docker run -v /path/to/models:/models hailo/hailo_dfc \
    hailo optimize /models/yolov8n_catprey.onnx --hw-arch hailo8l
```

**Important**: Use `--hw-arch hailo8l` (not `hailo8`) for the 13 TOPS AI HAT+.

### Step 3: Deploy to Raspberry Pi

1. Copy the compiled HEF to the Pi:
```bash
scp yolov8n_catprey.hef pi@<PI_IP>:~/mousehunter_hailo_v2/models/
```

2. Verify `config/config.json` matches model:
```json
{
    "inference": {
        "model_path": "models/yolov8n_catprey.hef",
        "classes": {"0": "cat", "1": "rodent"},
        "reg_max": 8
    }
}
```

3. Restart the service:
```bash
sudo systemctl restart mousehunter
```

### Step 4: Verify Custom Model

```bash
# Run hardware test
python test_hardware.py

# Check logs for detections
journalctl -u mousehunter -n 50 | grep -i "detection"

# Test inference
curl -s http://localhost:8080/status | python -m json.tool
```

### Model Files Location

| File | Location | Description |
|------|----------|-------------|
| PyTorch weights | Google Drive: `cat_prey/yolo_v8n_models_v4/` | Training checkpoints |
| ONNX model | Google Drive: `cat_prey/yolo_v8n_models_v4/` | 6-output format for Hailo |
| HEF model | Pi: `~/mousehunter_hailo_v2/models/` | Compiled for Hailo-8L |

### Current Deployed Model (v3)
- **File**: `yolov8n_catprey.hef`
- **Classes**: cat (0), rodent (1)
- **reg_max**: 8 (optimized for INT8)
- **Input size**: 640x640 RGB
- **Outputs**: 6 raw tensors (DFL bbox + class scores)

## Cloud Storage (rclone)

The system can automatically upload evidence and training data to cloud storage using rclone.

### Setup

1. Install rclone:
```bash
sudo apt install rclone
```

2. Configure a remote (e.g., Google Drive):
```bash
rclone config
# Follow prompts to create a remote named "gdrive"
```

3. Update `config/config.json`:
```json
{
    "cloud_storage": {
        "enabled": true,
        "rclone_remote": "gdrive",
        "remote_path": "MouseHunter",
        "upload_after_detection": true,
        "delete_local_after_upload": false
    }
}
```

Directories are created automatically on the remote if they don't exist.

## Training Data Capture

To improve detection accuracy, the system can capture images for model retraining:

### Capture Modes

| Mode | Trigger | Purpose |
|------|---------|---------|
| Periodic | Every 30 minutes | Environment baseline images |
| Cat-Only | Cat present for 2+ seconds, no prey | Normal cat behavior samples |
| Near-Miss | Verifying state resets without confirmation | Potential false negatives |

### Enable Training Data Capture

```json
{
    "training_data": {
        "enabled": true,
        "periodic_interval_minutes": 30,
        "capture_cat_only": true,
        "cat_only_delay_seconds": 2.0,
        "capture_near_miss": true,
        "include_detections_json": true,
        "max_images_per_day": 100,
        "use_inference_resolution": true,
        "local_dir": "runtime/training_data",
        "remote_path": "MouseHunter/training"
    }
}
```

Images are saved locally and optionally uploaded to cloud storage. Each capture includes a JSON file with detection metadata for labeling assistance.

## License

MIT License

## Acknowledgments

- [Hailo](https://hailo.ai/) for the AI accelerator
- [python-telegram-bot](https://python-telegram-bot.org/) for Telegram integration
- Original catflap-prey-detector project for inspiration
