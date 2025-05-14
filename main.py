import cv2
import cvzone
import math
import smtplib
import time
import threading
import numpy as np
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from ultralytics import YOLO
from collections import defaultdict

EMAIL_ADDRESS = 'fallalert688@gmail.com'  
TO_EMAIL = 'reciever@gmail.com'  #alert reciever mail id

# Fall detection parameters
ASPECT_RATIO_THRESHOLD = 1.6  
CONFIDENCE_THRESHOLD = 60 
FALL_CONFIRMATION_THRESHOLD = 1 
TIME_WINDOW = 1.5 

EMAIL_COOLDOWN = 60  
last_email_time = 0

# Person tracking data structure
class PersonTracker:
    def __init__(self, max_history=30):
        self.persons = {}  
        self.max_history = max_history
        self.next_id = 0
        self.fallen_persons = set()  
    
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
            
            # Find closest match
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
                # Update existing person
                person = self.persons[best_id]
                person['positions'].append((center_x, center_y, time.time()))
                person['boxes'].append((box, conf))
                person['last_seen'] = time.time()
                
                # Maintain history length
                if len(person['positions']) > self.max_history:
                    person['positions'].pop(0)
                if len(person['boxes']) > self.max_history:
                    person['boxes'].pop(0)
                    
                matched_ids.add(best_id)
            else:
                unmatched_detections.append((box, conf, center_x, center_y))
        
        # Create new tracks for unmatched detections
        for box, conf, cx, cy in unmatched_detections:
            self.persons[self.next_id] = {
                'positions': [(cx, cy, time.time())],
                'boxes': [(box, conf)],
                'fall_confirmation': 0,
                'last_seen': time.time()
            }
            self.next_id += 1
        
        # Remove old tracks
        person_ids = list(self.persons.keys())
        for person_id in person_ids:
            if time.time() - self.persons[person_id]['last_seen'] > 5.0:  
                if person_id in self.fallen_persons:
                    self.fallen_persons.remove(person_id)
                del self.persons[person_id]

    def detect_falls(self):
        """Detect falls for all tracked persons"""
        newly_fallen = []
        
        for person_id, data in self.persons.items():
            if len(data['positions']) < 3 or person_id in self.fallen_persons:
                continue
                
            # Get current properties
            latest_box, latest_conf = data['boxes'][-1]
            x1, y1, x2, y2 = latest_box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height
            
            # Get motion properties
            positions = data['positions']
            times = [p[2] for p in positions[-5:] if time.time() - p[2] < TIME_WINDOW]
            y_positions = [p[1] for p in positions[-5:] if time.time() - p[2] < TIME_WINDOW]
            
            # Fall indicators
            aspect_ratio_check = aspect_ratio > ASPECT_RATIO_THRESHOLD
            
            
            movement_check = False
            position_check = False
            
            if len(y_positions) >= 3:
                y_velocity = (y_positions[-1] - y_positions[0]) / max(0.1, times[-1] - times[0])
                movement_check = y_velocity > 50  
                
                frame_height = 740  # Estimated from resize in original code
                position_check = y_positions[-1] > frame_height * 0.7
            
            # Detect fall based on combined factors
            is_fallen = (aspect_ratio_check and (position_check or movement_check))
            
            if is_fallen:
                data['fall_confirmation'] += 1
                if data['fall_confirmation'] >= FALL_CONFIRMATION_THRESHOLD and person_id not in self.fallen_persons:
                    self.fallen_persons.add(person_id)
                    newly_fallen.append(person_id)
            else:
                data['fall_confirmation'] = max(0, data['fall_confirmation'] - 1)
                
        return newly_fallen, list(self.fallen_persons)

    def get_person_data(self, person_id):
        """Get data for specific person"""
        return self.persons.get(person_id)


def send_email_notification(frame=None):
    """Send fall detection email with optional frame attachment"""
    global last_email_time 
    # Prevent sending multiple emails within cooldown period
    current_time = time.time()
    if current_time - last_email_time < EMAIL_COOLDOWN:
        print("Email cooldown active, skipping notification")
        return False
    
    msg = MIMEMultipart()
    body = """
    ⚠️ FALL DETECTION ALERT ⚠️
    
    The fall detection system has detected a potential fall.
    
    Time of detection: {}
    
    This is an automated message. Please check on the monitored person immediately.
    """.format(time.strftime("%Y-%m-%d %H:%M:%S"))
    
    msg.attach(MIMEText(body, 'plain'))
    

    if frame is not None:
        temp_img_path = "fall_detection.jpg"
        cv2.imwrite(temp_img_path, frame)
        
        with open(temp_img_path, 'rb') as img_file:
            img_attachment = MIMEImage(img_file.read())
            img_attachment.add_header('Content-Disposition', 'attachment', filename='fall_detection.jpg')
            msg.attach(img_attachment)
        
        try:
            os.remove(temp_img_path)
        except:
            pass
    
    msg['Subject'] = '⚠️ URGENT: Fall Detection Alert'
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = TO_EMAIL
    
    try:
        email_thread = threading.Thread(target=_send_email, args=(msg,))
        email_thread.daemon = True
        email_thread.start()
        last_email_time = current_time
        print('✅ Fall notification triggered.')
        return True
    except Exception as e:
        print(f'❌ Failed to start email thread: {e}')
        return False

def _send_email(msg):
    """Send email using Brevo SMTP"""
    try:
        SMTP_SERVER = 'smtp-relay.brevo.com'
        SMTP_PORT = 587
        SMTP_USER = 'copy user @smtp-brevo.com' #create your own smtp user id
        SMTP_PASS = 'copy pass'

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(SMTP_USER, SMTP_PASS)
            smtp.send_message(msg)
            print('✅ Fall notification email sent successfully via Brevo.')
    except smtplib.SMTPAuthenticationError:
        print('❌ Brevo SMTP authentication failed. Check your credentials.')
    except Exception as e:
        print(f'❌ Email sending failed: {e}')


def draw_fall_detection_ui(frame, person_tracker, newly_fallen, fallen_ids):
    """Draw UI elements for fall detection visualization"""
    
    # Draw person tracking information
    for person_id, data in person_tracker.persons.items():
        if time.time() - data['last_seen'] > 1.0:  
            continue
            
        # Get latest box
        (x1, y1, x2, y2), conf = data['boxes'][-1]
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        
        # Calculate color based on fall status
        if person_id in fallen_ids:
            color = (0, 0, 255)  # Red for fallen
            thickness = 3
        elif data['fall_confirmation'] > 0:
            # Yellow for potential fall
            color = (0, 165, 255)
            thickness = 2
        else:
            color = (0, 255, 0)  # Green for normal
            thickness = 2
        
        # Draw bounding box with ID
        cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=5, colorR=color, colorC=color)
        
        # Person ID and confidence
        conf_percentage = math.ceil(conf * 100)
        display_text = f'Person #{person_id} {conf_percentage}%'
        cvzone.putTextRect(frame, display_text, [x1, y1 - 10], 
                          thickness=2, scale=1.5, colorR=color)
        
        # Draw fall information if detected
        if person_id in fallen_ids:
            cvzone.putTextRect(frame, f"FALL DETECTED", 
                              [x1, y2 + 30], scale=1.5, thickness=2, 
                              colorR=(0, 0, 255))
        elif data['fall_confirmation'] > 0:
            cvzone.putTextRect(frame, f"Potential Fall ({data['fall_confirmation']}/{FALL_CONFIRMATION_THRESHOLD})", 
                              [x1, y2 + 30], scale=1.5, thickness=2, 
                              colorR=(0, 165, 255))
    
    # Display global fall alert if anyone has fallen
    if fallen_ids:
        alert_text = f"FALL DETECTED: {len(fallen_ids)} person(s)!"
        text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        cv2.putText(frame, alert_text, (text_x, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    
    # Display system info
    cvzone.putTextRect(frame, f"People tracked: {len(person_tracker.persons)}", [20, 40], 
                      scale=1.2, thickness=2)
    
    mode_text = "VIDEO MODE" if video_mode else "LIVE MODE"
    cvzone.putTextRect(frame, mode_text, [frame.shape[1] - 200, 40], 
                      scale=1.2, thickness=2)
    
    # Draw controls help
    controls = [
        "q: Quit",
        "s: Switch mode",
        "r: Reset detection"
    ]
    y_pos = 80
    for control in controls:
        cvzone.putTextRect(frame, control, [20, y_pos], 
                          scale=0.8, thickness=1)
        y_pos += 30
        
    return frame


def detect_falls(video_source='fall.mp4'):
    """Main function to detect falls from video input"""
    global video_mode
    
    person_tracker = PersonTracker()
    
    # Initialize video capture
    if video_source == 0 or video_source == '0':
        print("Starting in LIVE mode with webcam")
        cap = cv2.VideoCapture(0)
        video_mode = False
    else:
        print(f"Starting in VIDEO mode with file: {video_source}")
        cap = cv2.VideoCapture(video_source)
        video_mode = True
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Load the YOLO model with a more accurate variant
    try:
        if os.path.exists('yolov8m.pt'):
            model = YOLO('yolov8m.pt')  
        else:
            model = YOLO('yolov8s.pt') 
        print(f"YOLO model loaded successfully")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        return
    
    # Variables for fall detection state
    email_sent = False
    newly_fallen_ids = []
    fallen_person_ids = []
    
    # Main detection loop
    while True:
        ret, frame = cap.read()
        if not ret:
            if video_mode:
                print("End of video or error reading frame. Restarting video...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
                continue
            else:
                print("Error reading from camera")
                break
        
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (980, 740))
        
        # Run detection
        results = model(frame, verbose=False)
        
        # Extract person detections
        detections = []
        for info in results:
            boxes = info.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                class_detect = int(box.cls[0])

                if class_detect == 0 and confidence * 100 > CONFIDENCE_THRESHOLD:  
                    detections.append(((x1, y1, x2, y2), confidence))
        
        # Update person tracking with new detections
        person_tracker.update(detections)
        
        # Detect falls
        newly_fallen_ids, fallen_person_ids = person_tracker.detect_falls()
        
        # Handle email notifications
        if newly_fallen_ids and not email_sent:
            email_sent = send_email_notification(frame)
        
        # If no more fallen people, reset email flag
        if not fallen_person_ids:
            email_sent = False
        
        # Draw UI elements
        frame = draw_fall_detection_ui(frame, person_tracker, newly_fallen_ids, fallen_person_ids)
        
        # Show the frame
        cv2.imshow('Enhanced Fall Detection System', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Exit
            break
        elif key == ord('s'):  # Switch mode
            cap.release()
            if video_mode:
                print("Switching to LIVE mode")
                cap = cv2.VideoCapture(0)
                video_mode = False
            else:
                print("Switching to VIDEO mode")
                cap = cv2.VideoCapture('fall.mp4')
                video_mode = True
        elif key == ord('r'):  # Reset detection
            print("Resetting fall detection")
            person_tracker = PersonTracker()
            email_sent = False
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_mode = True  # Start with video mode by default
    
    # Check for command line arguments
    import sys
    if len(sys.argv) > 1 and (sys.argv[1] == 'live' or sys.argv[1] == '0'):
        detect_falls(0)  # Use webcam
    else:
        detect_falls('fall.mp4') 
