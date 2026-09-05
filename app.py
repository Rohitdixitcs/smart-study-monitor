import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame

# Initialize Pygame Mixer
pygame.mixer.init()
alarm_sleep = pygame.mixer.Sound("alarm.mp3")       # Sleep alarm
alarm_facehide = pygame.mixer.Sound("faudio.mp3")   # Face hidden/covered alarm
alarm_phone = pygame.mixer.Sound("paudio.mp3")      # Phone alarm

# Track active audio state ('sleep', 'facehide', 'phone', or None)
current_playing = None

# Initialize Camera, Face Mesh, and YOLO Model
cap = cv2.VideoCapture(0)
face_detector = FaceMeshDetector(maxFaces=1)
phone_detector = YOLO("yolov8n.pt")  # COCO pre-trained lightweight model

# COCO Dataset Class Names
classNames = phone_detector.names  # Class '67' is 'cell phone' in standard COCO

# Landmark Indices for Sleep Detection
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
FACE_LEFT = 130
FACE_RIGHT = 243

# Frame Counters & Thresholds
closed_frames = 0
SLEEP_THRESHOLD_FRAMES = 15

covered_frames = 0
COVER_THRESHOLD_FRAMES = 20  # ~0.6 seconds of missing face triggers warning

while True:
    success, img = cap.read()
    if not success:
        break

    # Check if ANY sound is currently playing to completion
    is_audio_busy = pygame.mixer.get_busy()

    # If audio finished playing, reset tracking state to None
    if not is_audio_busy:
        current_playing = None

    # ------------------- 1. SLEEP & FACE COVER DETECTION -------------------
    img, faces = face_detector.findFaceMesh(img, draw=False)
    is_sleepy = False
    is_face_covered = False

    if faces:
        # Face visible -> Reset covered frames counter
        covered_frames = 0
        face = faces[0]

        eye_dist, _ = face_detector.findDistance(face[LEFT_EYE_TOP], face[LEFT_EYE_BOTTOM])
        face_dist, _ = face_detector.findDistance(face[FACE_LEFT], face[FACE_RIGHT])

        # Eye Aspect Ratio calculation
        ratio = (eye_dist / face_dist) * 100

        if ratio < 11.0:
            closed_frames += 1
        else:
            closed_frames = 0

        if closed_frames >= SLEEP_THRESHOLD_FRAMES:
            is_sleepy = True

        cvzone.putTextRect(img, f"Eye Ratio: {int(ratio)}", (30, 40), scale=1, thickness=1)

    else:
        # Face missing/covered -> Increment covered frames counter
        closed_frames = 0
        covered_frames += 1
        
        if covered_frames >= COVER_THRESHOLD_FRAMES:
            is_face_covered = True

    # ------------------- 2. PHONE DETECTION -------------------
    results = phone_detector.predict(img, stream=True, verbose=False)
    phone_detected = False

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if classNames[cls_id] == "cell phone" and conf > 0.5:
                phone_detected = True
                
                # Bounding box visual
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cvzone.putTextRect(img, f"Phone detected! {int(conf*100)}%", (x1, max(y1 - 10, 30)), scale=1, thickness=1, colorR=(255, 0, 255))

    # ------------------- 3. ALARM LOGIC & DISPLAY -------------------
    
    # Priority 1: Face covered warning
    if is_face_covered:
        cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        if not is_audio_busy:
            alarm_facehide.play(0)  # Play full track once
            current_playing = 'facehide'

    # Priority 2: Sleep warning
    elif is_sleepy:
        cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        if not is_audio_busy:
            alarm_sleep.play(0)  # Play full track once
            current_playing = 'sleep'

    # Priority 3: Phone warning
    elif phone_detected:
        cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))
        if not is_audio_busy:
            alarm_phone.play(0)  # Play full track once
            current_playing = 'phone'

    # Display active warning overlay if audio is still finishing up
    elif is_audio_busy:
        if current_playing == 'facehide':
            cvzone.putTextRect(img, "DONT COVER YOUR FACE!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        elif current_playing == 'sleep':
            cvzone.putTextRect(img, "WAKE UP & STUDY!", (50, 100), scale=2, thickness=3, colorR=(0, 0, 255))
        elif current_playing == 'phone':
            cvzone.putTextRect(img, "PUT THE PHONE AWAY!", (50, 100), scale=2, thickness=3, colorR=(0, 165, 255))

    cv2.imshow("Smart Study Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
#by Dibyendu pradhan
# https://dibyendup.netlify.app