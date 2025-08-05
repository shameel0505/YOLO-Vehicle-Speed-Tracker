# 🚦 Vehicle Speed Estimation & Traffic Density Analysis using YOLOv8

A real-time traffic analysis system using **YOLOv8** that detects and tracks vehicles from video input, estimates their speeds, classifies lane crossings, and monitors live traffic density.

---

## 📸 Demo Preview

![Demo](ScreenRecording2025-08-05at12.11.41-ezgif.com-video-to-gif-converter.gif)

---

## 🔧 Features

- 🔍 **Real-time Vehicle Detection** using Ultralytics YOLOv8
- 📍 **Speed Estimation** based on positional timestamps
- 🧭 **Lane-wise Vehicle Classification** (Left and Right)
- 📊 **Traffic Congestion Levels**: Low / Moderate / High
- 🧠 Annotated Frame Display with:
  - Vehicle ID
  - Type (car, bus, truck)
  - Calculated speed (km/h)
  - Lane-based count display

---

## 🗂️ Project Structure

```bash
YOLO-Vehicle-Speed-Tracker/
├── main.py             # Main script with detection, speed logic & overlay
├── requirements.txt    # Python dependencies
├── yolo11n.pt          # YOLO model weights (excluded from GitHub)
├── sample.mp4          # Sample input video (optional, <100MB)
├── README.md           # Project overview and usage guide
├── .gitignore          # To ignore temporary/system files
