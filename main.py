import streamlit as st
import cv2
import numpy as np
import time
import threading
import smtplib
import os
import tempfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
import math
from collections import defaultdict
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Fall Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'person_tracker' not in st.session_state:
    st.session_state.person_tracker = None
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'email_config' not in st.session_state:
    st.session_state.email_config = {
        'smtp_server': 'smtp-relay.brevo.com',
        'smtp_port': 587,
        'sender_email': 'fallalert688@gmail.com',
        'receiver_email': 'receiver@gmail.com',
        'smtp_user': '',
        'smtp_pass': '',
        'email_cooldown': 60
    }
if 'detection_params' not in st.session_state:
    st.session_state.detection_params = {
        'aspect_ratio_threshold': 1.6,
        'confidence_threshold': 60,
        'fall_confirmation_threshold': 1,
        'time_window': 1.5
    }

# Person tracking class (same as original)
class PersonTracker:
    def __init__(self, max_history=30):
        self.persons = {}
        self.max_history = max_history
        self.next_id = 0
        self.fallen_persons = set()
        self.last_email_time = 0
    
    def update(self, detections):
        """Update tracking with new detections"""
        if not self.persons:
            for box, conf in detections:
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                self.persons[self.next_id] = {
                    'positions': [(center_x, center_y, time.time())],
                    'boxes': [(box, conf)],
                    'fall_confirmation': 0,
                    'last_seen': time.time()
                }
                self.next_id += 1
            return
        
        # Match new detections with existing persons
        matched_ids = set()
        unmatched_detections = []
        
        for box, conf in detections:
            x1, y1, x2, y2 = box
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            best_id = None
            min_dist = float('inf')
            
            for person_id, data in self.persons.items():
                if person_id in matched_ids:
                    continue
                
                if time.time() - data['last_seen'] > 2.0:
                    continue
                    
                last_x, last_y, _ = data['positions'][-1]
                dist = math.sqrt((center_x - last_x)**2 + (center_y - last_y)**2)
                
                if dist < min_dist and dist < 150:
                    min_dist = dist
                    best_id = person_id
            
            if best_id is not None:
                person = self.persons[best_id]
                person['positions'].append((center_x, center_y, time.time()))
                person['boxes'].append((box, conf))
                person['last_seen'] = time.time()
                
                if len(person['positions']) > self.max_history:
                    person['positions'].pop(0)
                if len(person['boxes']) > self.max_history:
                    person['boxes'].pop(0)
                    
                matched_ids.add(best_id)
            else:
                unmatched_detections.append((box, conf, center_x, center_y))
        
        for box, conf, cx, cy in unmatched_detections:
            self.persons[self.next_id] = {
                'positions': [(cx, cy, time.time())],
                'boxes': [(box, conf)],
                'fall_confirmation': 0,
                'last_seen': time.time()
            }
            self.next_id += 1
        
        person_ids = list(self.persons.keys())
        for person_id in person_ids:
            if time.time() - self.persons[person_id]['last_seen'] > 5.0:
                if person_id in self.fallen_persons:
                    self.fallen_persons.remove(person_id)
                del self.persons[person_id]

    def detect_falls(self, params):
        """Detect falls for all tracked persons"""
        newly_fallen = []
        
        for person_id, data in self.persons.items():
            if len(data['positions']) < 3 or person_id in self.fallen_persons:
                continue
                
            latest_box, latest_conf = data['boxes'][-1]
            x1, y1, x2, y2 = latest_box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height
            
            positions = data['positions']
            times = [p[2] for p in positions[-5:] if time.time() - p[2] < params['time_window']]
            y_positions = [p[1] for p in positions[-5:] if time.time() - p[2] < params['time_window']]
            
            aspect_ratio_check = aspect_ratio > params['aspect_ratio_threshold']
            
            movement_check = False
            position_check = False
            
            if len(y_positions) >= 3:
                y_velocity = (y_positions[-1] - y_positions[0]) / max(0.1, times[-1] - times[0])
                movement_check = y_velocity > 50
                
                frame_height = 740
                position_check = y_positions[-1] > frame_height * 0.7
            
            is_fallen = (aspect_ratio_check and (position_check or movement_check))
            
            if is_fallen:
                data['fall_confirmation'] += 1
                if data['fall_confirmation'] >= params['fall_confirmation_threshold'] and person_id not in self.fallen_persons:
                    self.fallen_persons.add(person_id)
                    newly_fallen.append(person_id)
            else:
                data['fall_confirmation'] = max(0, data['fall_confirmation'] - 1)
                
        return newly_fallen, list(self.fallen_persons)

def send_email_notification(frame, email_config):
    """Send fall detection email with frame attachment"""
    current_time = time.time()
    if current_time - st.session_state.person_tracker.last_email_time < email_config['email_cooldown']:
        return False
    
    msg = MIMEMultipart()
    body = f"""
    ⚠️ FALL DETECTION ALERT ⚠️
    
    The fall detection system has detected a potential fall.
    
    Time of detection: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    This is an automated message. Please check on the monitored person immediately.
    """
    
    msg.attach(MIMEText(body, 'plain'))
    
    if frame is not None:
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, frame)
            
            with open(temp_file.name, 'rb') as img_file:
                img_attachment = MIMEImage(img_file.read())
                img_attachment.add_header('Content-Disposition', 'attachment', filename='fall_detection.jpg')
                msg.attach(img_attachment)
            
            os.unlink(temp_file.name)
    
    msg['Subject'] = '⚠️ URGENT: Fall Detection Alert'
    msg['From'] = email_config['sender_email']
    msg['To'] = email_config['receiver_email']
    
    try:
        with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as smtp:
            smtp.starttls()
            smtp.login(email_config['smtp_user'], email_config['smtp_pass'])
            smtp.send_message(msg)
        
        st.session_state.person_tracker.last_email_time = current_time
        return True
    except Exception as e:
        st.error(f"Failed to send email: {e}")
        return False

def draw_detections_on_frame(frame, person_tracker, fallen_ids, params):
    """Draw detection boxes and information on frame"""
    annotated_frame = frame.copy()
    
    for person_id, data in person_tracker.persons.items():
        if time.time() - data['last_seen'] > 1.0:
            continue
            
        (x1, y1, x2, y2), conf = data['boxes'][-1]
        width = x2 - x1
        height = y2 - y1
        
        if person_id in fallen_ids:
            color = (0, 0, 255)  # Red for fallen
            thickness = 3
            status = "FALLEN"
        elif data['fall_confirmation'] > 0:
            color = (0, 165, 255)  # Orange for potential fall
            thickness = 2
            status = f"POTENTIAL FALL ({data['fall_confirmation']}/{params['fall_confirmation_threshold']})"
        else:
            color = (0, 255, 0)  # Green for normal
            thickness = 2
            status = "NORMAL"
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw person info
        conf_percentage = int(conf * 100)
        label = f'Person #{person_id} ({conf_percentage}%) - {status}'
        
        # Calculate text size and draw background
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

def main():
    st.title("🚨 Fall Detection System")
    st.markdown("Real-time fall detection using AI and computer vision")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Detection parameters
    st.sidebar.subheader("Detection Parameters")
    st.session_state.detection_params['aspect_ratio_threshold'] = st.sidebar.slider(
        "Aspect Ratio Threshold", 1.0, 3.0, 1.6, 0.1
    )
    st.session_state.detection_params['confidence_threshold'] = st.sidebar.slider(
        "Confidence Threshold (%)", 30, 90, 60, 5
    )
    st.session_state.detection_params['fall_confirmation_threshold'] = st.sidebar.slider(
        "Fall Confirmation Threshold", 1, 5, 1, 1
    )
    st.session_state.detection_params['time_window'] = st.sidebar.slider(
        "Time Window (seconds)", 0.5, 3.0, 1.5, 0.1
    )
    
    # Email configuration
    st.sidebar.subheader("Email Configuration")
    st.session_state.email_config['sender_email'] = st.sidebar.text_input(
        "Sender Email", st.session_state.email_config['sender_email']
    )
    st.session_state.email_config['receiver_email'] = st.sidebar.text_input(
        "Receiver Email", st.session_state.email_config['receiver_email']
    )
    st.session_state.email_config['smtp_user'] = st.sidebar.text_input(
        "SMTP User", st.session_state.email_config['smtp_user']
    )
    st.session_state.email_config['smtp_pass'] = st.sidebar.text_input(
        "SMTP Password", st.session_state.email_config['smtp_pass'], type="password"
    )
    st.session_state.email_config['email_cooldown'] = st.sidebar.slider(
        "Email Cooldown (seconds)", 30, 300, 60, 10
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Video Feed")
        video_source = st.selectbox(
            "Select Video Source",
            ["Upload Video File", "Use Webcam", "Use Sample Video"]
        )
        
        # Video processing area
        video_placeholder = st.empty()
        
        if video_source == "Upload Video File":
            uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov', 'mkv'])
            if uploaded_file is not None:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                    temp_file.write(uploaded_file.read())
                    temp_video_path = temp_file.name
                
                if st.button("Start Detection"):
                    process_video(temp_video_path, video_placeholder)
        
        elif video_source == "Use Webcam":
            st.info("Webcam functionality requires local deployment")
            if st.button("Start Webcam Detection"):
                process_video(0, video_placeholder)
        
        elif video_source == "Use Sample Video":
            st.info("Make sure 'fall.mp4' is in the same directory")
            if st.button("Start Sample Video Detection"):
                if os.path.exists('fall.mp4'):
                    process_video('fall.mp4', video_placeholder)
                else:
                    st.error("Sample video 'fall.mp4' not found")
    
    with col2:
        st.subheader("📊 Detection Statistics")
        
        # Real-time statistics
        if st.session_state.person_tracker:
            tracker = st.session_state.person_tracker
            
            # Current statistics
            st.metric("Active Persons", len(tracker.persons))
            st.metric("Fallen Persons", len(tracker.fallen_persons))
            st.metric("Total Detections", len(st.session_state.detection_log))
            
            # Person details
            if tracker.persons:
                st.subheader("👥 Person Details")
                for person_id, data in tracker.persons.items():
                    status = "🔴 FALLEN" if person_id in tracker.fallen_persons else "🟢 NORMAL"
                    confidence = int(data['boxes'][-1][1] * 100) if data['boxes'] else 0
                    
                    with st.expander(f"Person #{person_id} - {status}"):
                        st.write(f"**Confidence:** {confidence}%")
                        st.write(f"**Fall Confirmation:** {data['fall_confirmation']}")
                        st.write(f"**Last Seen:** {time.strftime('%H:%M:%S', time.localtime(data['last_seen']))}")
        
        # Detection log
        st.subheader("📋 Detection Log")
        if st.session_state.detection_log:
            log_df = pd.DataFrame(st.session_state.detection_log)
            st.dataframe(log_df.tail(10), use_container_width=True)
        else:
            st.info("No detections yet")
        
        # Clear log button
        if st.button("Clear Log"):
            st.session_state.detection_log = []
            st.rerun()

def process_video(video_source, video_placeholder):
    """Process video and detect falls"""
    try:
        # Load YOLO model
        @st.cache_resource
        def load_model():
            if os.path.exists('yolov8m.pt'):
                return YOLO('yolov8m.pt')
            else:
                return YOLO('yolov8s.pt')
        
        model = load_model()
        
        # Initialize person tracker
        if st.session_state.person_tracker is None:
            st.session_state.person_tracker = PersonTracker()
        
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            st.error("Could not open video source")
            return
        
        # Process frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                if isinstance(video_source, str) and video_source != '0':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                    continue
                else:
                    break
            
            frame_count += 1
            if frame_count % 3 != 0:  # Process every 3rd frame for performance
                continue
            
            # Resize frame
            frame = cv2.resize(frame, (980, 740))
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Extract person detections
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        class_id = int(box.cls[0])
                        
                        if class_id == 0 and confidence * 100 > st.session_state.detection_params['confidence_threshold']:
                            detections.append(((x1, y1, x2, y2), confidence))
            
            # Update tracking
            st.session_state.person_tracker.update(detections)
            
            # Detect falls
            newly_fallen, fallen_ids = st.session_state.person_tracker.detect_falls(
                st.session_state.detection_params
            )
            
            # Log new falls
            for person_id in newly_fallen:
                log_entry = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'person_id': person_id,
                    'event': 'Fall Detected',
                    'confidence': int(st.session_state.person_tracker.persons[person_id]['boxes'][-1][1] * 100)
                }
                st.session_state.detection_log.append(log_entry)
                
                # Send email notification
                if st.session_state.email_config['smtp_user'] and st.session_state.email_config['smtp_pass']:
                    email_sent = send_email_notification(frame, st.session_state.email_config)
                    if email_sent:
                        st.success(f"Email alert sent for Person #{person_id}")
            
            # Draw detections
            annotated_frame = draw_detections_on_frame(
                frame, st.session_state.person_tracker, fallen_ids, st.session_state.detection_params
            )
            
            # Convert BGR to RGB for Streamlit
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
            
            # Add a small delay and check for stop condition
            time.sleep(0.1)
            
            # Break condition (you might want to add a stop button)
            if len(st.session_state.detection_log) > 100:  # Limit for demo
                break
        
        cap.release()
        
    except Exception as e:
        st.error(f"Error processing video: {e}")

if __name__ == "__main__":
    main()