# 🚦 Edge-Calibrated Vehicle Speed & Flow Estimation Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8 / YOLO11](https://img.shields.io/badge/YOLO-v8%20%7C%2011-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Tests Passing](https://img.shields.io/badge/tests-5%2F5%20passing-brightgreen.svg)](tests/)

A robust, computer-vision pipeline for multi-class vehicle detection, persistent multi-object tracking (ByteTrack/BoT-SORT), frame-synchronized velocity estimation, and lane density monitoring from static camera feeds.

---

## 📸 Demo Preview

![Demo](ScreenRecording2025-08-05at12.11.41-ezgif.com-video-to-gif-converter.gif)

---

## 📐 Mathematical & Architectural Design

### 1. Frame-Synchronized Velocity Calculation
A frequent pitfall in CV velocity tracking is relying on system wall-clock time (`time.time()`). When GPU inference fluctuates (e.g., from 15ms to 60ms under heavy batches), measuring wall-clock time measures **inference latency**, distorting real-world speed.

This pipeline synchronizes measurement with the video stream's intrinsic frame metadata:

$$\Delta t = \frac{\Delta \text{frames}}{\text{FPS}} = \frac{|f_{\text{exit}} - f_{\text{entry}}|}{\text{FPS}}$$

$$v = \left(\frac{d}{\Delta t}\right) \times 3.6 \quad [\text{km/h}]$$

where $d$ is the physically calibrated Euclidean distance (meters) between the virtual optical tripwires.

### 2. Perspective Distortion & Homography Calibration
In surveillance cameras with acute tilt angles, planar 2D pixel coordinates undergo non-linear perspective compression. 
- In pure 2D space, a 10-pixel movement in the far field represents significantly more physical distance than a 10-pixel movement in the near field.
- For maximum precision, this engine supports planar homography mapping:

$$\begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Using known landmark points (e.g., standard lane dash lengths: 3.0m in the US/EU, 4.5m in UAE), a $3 \times 3$ projective transformation matrix $H$ maps pixel centroids $(u, v)$ directly to ground plane metric coordinates $(X_w, Y_w)$.

### 3. State-Machine Gate Tracking
To eliminate double-counting and trajectory fragmentation:
- Bounding box centroids $(\bar{x}, \bar{y})$ are tracked across frame boundaries.
- Directional vectors differentiate oncoming traffic from receding traffic via independent entry/exit gates.
- Track state transitions are atomic and idempotent: entries cannot trigger duplicate velocity outputs without a full gate traversal.

---

## 🏗️ Repository Architecture

```
YOLO-Vehicle-Speed-Tracker/
├── speed_tracker.py       # Core mathematical estimator & traffic density engine
├── traffic.py             # CLI application & video stream processor
├── tests/
│   └── test_tracker.py    # Unit tests validating speed physics & outlier rejection
├── forcalc.jpg            # Calibration reference frame
├── requirements.txt       # Production dependencies
└── LICENSE                # MIT License
```

---

## ⚡ Quickstart & Usage

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/shameel0505/YOLO-Vehicle-Speed-Tracker.git
cd YOLO-Vehicle-Speed-Tracker

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Inference
Weights are automatically fetched on first execution via Ultralytics:

```bash
# Run on a local video file
python traffic.py --source sample.mp4 --model yolo11n.pt --distance 18.0

# Run headless and export annotated video
python traffic.py --source sample.mp4 --no-show --output output_annotated.mp4

# Run with custom detection classes (2=car, 3=motorcycle, 5=bus, 7=truck)
python traffic.py --source sample.mp4 --classes 2 7 --conf 0.60
```

### 3. Running Unit Tests
Validate physical calculation accuracy across varying frame rates:
```bash
pytest tests/ -v
```

---

## 📊 Limitations & Operational Boundaries

| Factor | Threshold / Constraint | Mitigation Strategy |
| :--- | :--- | :--- |
| **Severe Occlusion** | > 60% bounding box overlap | Kalman filter state estimation preserves ID across short occlusions. |
| **Frame Rate** | < 15 FPS | Lower FPS increases temporal quantization error; 30+ FPS recommended. |
| **Camera Jitter** | Unstabilized wind vibrations | Digital video stabilization recommended prior to passing frames to tracker. |

---

## 📄 License
Distributed under the [MIT License](LICENSE).
