import streamlit as st
import cv2
import cvzone
import math
import pygame
import time
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO

# --- PREMIUM CSS ---
st.set_page_config(page_title="Smart Study Monitor", layout="wide", page_icon="📚")
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #ffffff; }
    .main-header { background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(0, 255, 255, 0.3); box-shadow: 0 0 20px rgba(0, 255, 255, 0.2); }
    .main-header h1 { font-size: 2.8rem; background: -webkit-linear-gradient(#00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255, 255, 255, 0.03); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 10px; }
    .status-card { padding: 15px; border-radius: 12px; margin-bottom: 10px; text-align: center; font-weight: 800; font-size: 1rem; transition: all 0.3s ease-in-out; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
    .active-red { background: linear-gradient(45deg, #ff4b4b, #cc0000); color: white; animation: pulse-red 2s infinite; }
    .active-orange { background: linear-gradient(45deg, #ffa500, #cc8400); color: white; animation: pulse-red 2s infinite; }
    .ok-green { background: linear-gradient(45deg, #28a745, #1e7e34); color: white; }
    .warn-yellow { background: linear-gradient(45deg, #ffc107, #d39e00); color: #212529; }
    .footer { margin-top: 40px; text-align: center; color: #d1d5db; padding: 15px; border-top: 1px solid rgba(255, 255, 255, 0.1); }
    .footer a { color: #00ffff; text-decoration: none; font-weight: bold; margin: 0 5px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📚 Smart Study Monitor</h1>
    <p>Local AI Neural Focus System</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛠️ Settings")
    sleep_threshold = st.slider("Eye Closure Sensitivity", 0.05, 0.5, 0.2, 0.01)
    phone_threshold = st.slider("Phone Detection Confidence", 0.0, 1.0, 0.4, 0.05)
    st.info("💡 Click Start to run locally. Audio plays directly on your PC.")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    face_detector = FaceMeshDetector(maxFaces=1)
    hand_detector = HandDetector(detectionCon=0.8, maxHands=2)
    yolo_model = YOLO("yolov8n.pt")
    return face_detector, hand_detector, yolo_model

# --- LOCAL AUDIO SETUP ---
pygame.mixer.init()
def play_sound_loop(file_path):
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play(-1)

def stop_sound():
    if pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()

# --- MAIN LOOP ---
if st.sidebar.button("▶️ START MONITORING"):
    face_detector, hand_detector, model = load_models()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("### 📹 Live Neural Feed")
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("### 🟢 Status Dashboard")
        status_sleep = st.empty()
        status_face = st.empty()
        status_phone = st.empty()
        status_general = st.empty()

    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    current_playing = None
    frame_count = 0
    phone_lock_time = 0
    face_cover_lock_time = 0

    while True:
        success, img = cap.read()
        if not success:
            st.error("Cannot access camera!")
            break

        frame_count += 1

        # *** FIX: Initialize variables before frame skipping ***
        faces = []
        hands = []

        h, w, _ = img.shape
        img_small = cv2.resize(img, (320, 240))
        sx, sy = w / 320, h / 240

        # 1. Face & Hand Detection (Runs every 2nd frame)
        if frame_count % 2 == 0:
            img_small, faces = face_detector.findFaceMesh(img_small, draw=False)
            hands, _ = hand_detector.findHands(img_small, draw=False)
        
        is_sleepy = False
        is_face_covered = False
        is_phone = False
        face_visible = False

        if faces:
            face_visible = True
            face = faces[0]
            def dist(p1, p2):
                return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            r_horiz = dist(face[33], face[133])
            r_vert = dist(face[160], face[144])
            right_ear = r_vert / r_horiz
            l_horiz = dist(face[362], face[263])
            l_vert = dist(face[385], face[380])
            left_ear = l_vert / l_horiz
            ear = (right_ear + left_ear) / 2
            if ear < sleep_threshold: is_sleepy = True

        # FACE COVER LOGIC
        if not face_visible and hands:
            is_face_covered = True
            face_cover_lock_time = time.time() + 1.5

        if time.time() < face_cover_lock_time:
            is_face_covered = True

        # 2. Phone Detection (Runs every 3rd frame)
        if frame_count % 3 == 0:
            results = model(img_small, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 67 and conf > phone_threshold:
                        is_phone = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cvzone.putTextRect(img, "PHONE DETECTED", (x1, y1 - 10), scale=2, colorR=(0, 255, 255))
                        phone_lock_time = time.time() + 1.5
        
        if time.time() < phone_lock_time:
            is_phone = True

        # 3. AUDIO LOGIC
        if is_sleepy:
            if current_playing != 'sleep':
                stop_sound()
                play_sound_loop("alarm.mp3")
                current_playing = 'sleep'
        elif is_face_covered:
            if current_playing != 'face':
                stop_sound()
                play_sound_loop("faudio.mp3")
                current_playing = 'face'
        elif is_phone:
            if current_playing != 'phone':
                stop_sound()
                play_sound_loop("paudio.mp3")
                current_playing = 'phone'
        else:
            if current_playing is not None:
                stop_sound()
                current_playing = None

        # 4. UPDATE UI
        frame_placeholder.image(img, channels="BGR", width='stretch')

        if is_sleepy:
            status_sleep.markdown('<div class="status-card active-red">😴 SLEEPING</div>', unsafe_allow_html=True)
        else:
            status_sleep.markdown('<div class="status-card ok-green">😊 Eyes Open</div>', unsafe_allow_html=True)

        if is_face_covered:
            status_face.markdown('<div class="status-card active-red">🙈 FACE COVERED</div>', unsafe_allow_html=True)
        elif not face_visible:
            status_face.markdown('<div class="status-card warn-yellow">👀 Searching...</div>', unsafe_allow_html=True)
        else:
            status_face.markdown('<div class="status-card ok-green">🙂 Face Visible</div>', unsafe_allow_html=True)

        if is_phone:
            status_phone.markdown('<div class="status-card active-orange">📱 PHONE DETECTED!</div>', unsafe_allow_html=True)
        else:
            status_phone.markdown('<div class="status-card ok-green">📵 No Phone</div>', unsafe_allow_html=True)

        status_general.markdown("### ⚡ System Running Locally...")

    cap.release()
    cv2.destroyAllWindows()

else:
    st.info("👈 Press **START MONITORING** in the sidebar to engage the local AI.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Made with ❤️ by <strong>Rohit Dixit</strong> | 
    <a href="https://www.linkedin.com/in/rohitdixitcs/" target="_blank">LinkedIn</a> | 
    <a href="https://rohitdixitdev.vercel.app/" target="_blank">Portfolio</a>
</div>
""", unsafe_allow_html=True)