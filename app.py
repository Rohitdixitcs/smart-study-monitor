import streamlit as st
import cv2
import cvzone
import math
import base64
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.HandTrackingModule import HandDetector
from ultralytics import YOLO

# --- FULLY RESPONSIVE CSS (Works on Phone, Tablet, Laptop, TV) ---
st.set_page_config(page_title="Smart Study Monitor", layout="wide", page_icon="📚")

st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #ffffff; }
    
    /* Responsive Header */
    .main-header { background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(0, 255, 255, 0.3); box-shadow: 0 0 30px rgba(0, 255, 255, 0.2); }
    .main-header h1 { background: -webkit-linear-gradient(#00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 3rem; }
    
    /* Status Cards */
    .status-card { padding: 18px; border-radius: 15px; margin-bottom: 15px; text-align: center; font-weight: 800; font-size: 1.1rem; transition: all 0.3s ease-in-out; }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
    @keyframes pulse-orange { 0% { box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(255, 165, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 165, 0, 0); } }
    .active-red { background: linear-gradient(45deg, #ff4b4b, #cc0000); color: white; animation: pulse-red 2s infinite; }
    .active-orange { background: linear-gradient(45deg, #ffa500, #cc8400); color: white; animation: pulse-orange 2s infinite; }
    .ok-green { background: linear-gradient(45deg, #28a745, #1e7e34); color: white; }
    .warn-yellow { background: linear-gradient(45deg, #ffc107, #d39e00); color: #212529; }

    /* Video Frame */
    div[data-testid="stImage"] { border-radius: 15px; overflow: hidden; box-shadow: 0 0 30px rgba(0, 255, 255, 0.2); border: 2px solid rgba(0, 255, 255, 0.3); }
    
    /* Footer */
    .footer { position: fixed; left: 0; bottom: 0; width: 100%; background: rgba(0, 0, 0, 0.7); color: white; text-align: center; padding: 10px; font-size: 1rem; z-index: 1000; }
    .footer a { color: #00ffff; text-decoration: none; font-weight: bold; margin: 0 5px; }

    /* ---------- MOBILE PHONE SETTINGS ---------- */
    @media (max-width: 768px) {
        .main-header h1 { font-size: 2rem; }
        .main-header p { font-size: 1rem; }
        .stButton > button { padding: 10px; font-size: 14px; }
        .status-card { font-size: 0.9rem; padding: 12px; }
    }

    /* ---------- TV / LARGE SCREEN SETTINGS ---------- */
    @media (min-width: 1600px) {
        .main-header h1 { font-size: 4rem; }
        .status-card { font-size: 1.4rem; padding: 20px; }
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📚 Smart Study Monitor</h1>
    <p>Neural Focus System</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("## 🛠️ Control Center")
    sleep_threshold = st.slider("Eye Closure Sensitivity", 0.05, 0.5, 0.2, 0.01)
    phone_threshold = st.slider("Phone Detection Confidence", 0.0, 1.0, 0.4, 0.05)
    st.info("💡 **Tip:** Tap/Click the Start button once to allow looping alarms.")

# --- LOAD MODELS (Cached) ---
@st.cache_resource
def load_models():
    face_detector = FaceMeshDetector(maxFaces=1)
    hand_detector = HandDetector(detectionCon=0.8, maxHands=2)
    yolo_model = YOLO("yolov8s.pt") 
    return face_detector, hand_detector, yolo_model

# --- LOGIC (In the Cloud, using browser video frames) ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector, self.hand_detector, self.model = load_models()
        self.status = {'sleep': False, 'cover': False, 'phone': False, 'face': False}
        self.audio_state = None 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Face & Sleep Detection
        img, faces = self.face_detector.findFaceMesh(img, draw=False)
        hands, img = self.hand_detector.findHands(img, draw=False)

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

        if not face_visible and hands:
            is_face_covered = True

        # Phone Detection
        results = self.model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if cls == 67 and conf > phone_threshold:
                    is_phone = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cvzone.putTextRect(img, "PHONE DETECTED", (x1, y1 - 10), scale=2, colorR=(0, 255, 255))

        # Update Status & Audio
        self.status = {'sleep': is_sleepy, 'cover': is_face_covered, 'phone': is_phone, 'face': face_visible}
        
        if is_sleepy: self.audio_state = 'sleep'
        elif is_face_covered: self.audio_state = 'cover'
        elif is_phone: self.audio_state = 'phone'
        else: self.audio_state = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- LAYOUT (Automatically stacks on Mobile) ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Live Neural Feed")
    ctx = webrtc_streamer(key="smart-study-monitor", video_processor_factory=VideoProcessor, rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

with col2:
    st.markdown("### 🟢 Status")
    status_sleep = st.empty()
    status_face = st.empty()
    status_phone = st.empty()
    status_general = st.empty()
    audio_placeholder = st.empty()

# --- UI & AUDIO UPDATE LOOP ---
if ctx.video_processor:
    def play_audio_loop(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f'<audio autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

    while True:
        proc = ctx.video_processor
        if proc:
            # Status Cards
            if proc.status['sleep']:
                status_sleep.markdown('<div class="status-card active-red">😴 SLEEPING</div>', unsafe_allow_html=True)
            else:
                status_sleep.markdown('<div class="status-card ok-green">😊 Eyes Open</div>', unsafe_allow_html=True)

            if proc.status['cover']:
                status_face.markdown('<div class="status-card active-red">🙈 FACE COVERED</div>', unsafe_allow_html=True)
            elif not proc.status['face']:
                status_face.markdown('<div class="status-card warn-yellow">👀 Searching...</div>', unsafe_allow_html=True)
            else:
                status_face.markdown('<div class="status-card ok-green">🙂 Face Visible</div>', unsafe_allow_html=True)

            if proc.status['phone']:
                status_phone.markdown('<div class="status-card active-orange">📱 PHONE DETECTED</div>', unsafe_allow_html=True)
            else:
                status_phone.markdown('<div class="status-card ok-green">📵 No Phone</div>', unsafe_allow_html=True)

            # Audio Logic
            if proc.audio_state == 'sleep':
                play_audio_loop("alarm.mp3")
            elif proc.audio_state == 'cover':
                play_audio_loop("faudio.mp3")
            elif proc.audio_state == 'phone':
                play_audio_loop("paudio.mp3")
            else:
                audio_placeholder.empty()

            status_general.markdown("### ⚡ System Running...")
        time.sleep(0.1) 
else:
    st.info("👈 Press **Start** on the video player to engage the AI.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Made with ❤️ by <strong>Rohit Dixit</strong> | 
    <a href="https://www.linkedin.com/in/rohitdixitcs/" target="_blank">LinkedIn</a> | 
    <a href="https://rohitdixitdev.vercel.app/" target="_blank">Portfolio</a>
</div>
""", unsafe_allow_html=True)