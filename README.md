Fall Detection System 🚨
A real-time fall detection system using YOLOv8, OpenCV, and email alerts via SMTP. It detects human falls from video input or webcam and sends email notifications with an image when a fall is detected.

Features
🎯 Accurate Detection using YOLOv8 (yolov8s or yolov8m)

📦 Person Tracking with unique IDs and history

📉 Fall Detection Logic based on:

1 Aspect ratio (person lying down)
2 Sudden vertical motion
3 Position on screen

📸 Email Alert with frame image via Brevo SMTP

🔁 Video & Live Webcam Modes

🔧 Configurable Thresholds and Parameters

Requirements:

Install dependencies:

pip install ultralytics opencv-python cvzone numpy


Usage:

Run with default video: 
python fall_detection.py

Run with webcam:
python fall_detection.py live


Controls
Key	Action
q	Quit the program
s	Switch mode (video/live)
r	Reset detection

Email Configuration
Uses Brevo SMTP.
Modify the following variables in the script:

TO_EMAIL = 'reciever-email'                # Receiver email

Notes
Works with any YOLOv8 model (yolov8s.pt, yolov8m.pt).

Resumes video playback from the start once finished.

Detection thresholds can be adjusted for sensitivity tuning.