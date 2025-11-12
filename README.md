# 🕵️‍♂️ Real-Time Object Detection with YOLOv8

![Python](https://img.shields.io/badge/python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-orange)
![Ultralytics YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-red)

Real-time object detection system using **YOLOv8**. Detects all objects in view, including guns, knives, keys, mice, and more. Frames with detected objects are saved automatically, and alerts are printed in the console.

---

## Features

- Real-time detection via laptop webcam.
- Detects **all YOLOv8 classes**.
- Draws bounding boxes with **object names and confidence scores**.
- Saves frames of detected objects in `alerts/`.
- Console alerts for every detection.
- Optional PowerShell script to view alerts automatically.

---

## Demo

![Demo](docs/demo.gif)  
*Example of detection with bounding boxes and alerts.*

---

## Installation

```bash
git clone https://github.com/your-username/weapon-detection.git
cd weapon-detection
python -m venv venv
.\venv\Scripts\activate
pip install ultralytics opencv-python
