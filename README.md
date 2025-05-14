# 🧍‍♂️ Fall Detection System 🚨

> A real-time fall detection system using YOLOv8, OpenCV, and SMTP-based email alerts.  
> Detects human falls in video or webcam feeds and notifies via email with a captured frame.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-ultralytics-brightgreen?logo=ai)
![OpenCV](https://img.shields.io/badge/OpenCV-computer--vision-red?logo=opencv)
![SMTP](https://img.shields.io/badge/SMTP-Email%20Alert-yellow?logo=gmail)

---

## ✨ Features

- 🎯 **Accurate Detection** using YOLOv8 (`yolov8s` or `yolov8m`)
- 🧍 **Person Tracking** with unique IDs & position history
- ⚠️ **Fall Detection Logic** based on:
  1. **Aspect Ratio** (detect lying down)
  2. **Sudden Vertical Motion**
  3. **Screen Position** (proximity to floor)
- 📸 **Email Alert** with captured image via **Brevo SMTP**
- 📹 **Video File & Live Webcam Modes**
- ⚙️ **Configurable Thresholds** and detection sensitivity

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install ultralytics opencv-python cvzone numpy
