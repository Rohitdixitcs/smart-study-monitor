import streamlit as st
import cv2
import cvzone
import math
import base64
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO

# --- CSS for responsive UI (same as before) ---
st.set_page_config(page_title="Smart Study Monitor", layout="wide", page_icon="📚")
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: #ffffff; }
    .main-header { background: rgba(255, 255, 255, 0.05); padding: 20px; border-radius: 15px; text-align: center; margin-bottom: 20px; border: 1px solid rgba(0, 255, 255, 0.3); box-shadow: 0 0 20px rgba(0, 255, 255, 0.2); }
    .main-header h1 { font-size: 2.8rem; background: -webkit-linear-gradient(#00ffff, #ff00ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    
    div[data-testid="stVerticalBlockBorderWrapper"] { background: rgba(255, 255, 255, 0.03); border-radius: 15px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 10px; }

    .status-card { padding: 15px; border-radius: 12px; margin-bottom: 10px; text-align: center; font-weight: 800; font-size: 1rem; transition: all 0.3s ease-in-out; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }
    @keyframes pulse-red { 0% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(255, 0, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 0, 0, 0); } }
    @keyframes pulse-orange { 0% { box-shadow: 0 0 0 0 rgba(255, 165, 0, 0.7); } 70% { box-shadow: 0 0 0 15px rgba(255, 165, 0, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 165, 0, 0); } }
    .active-red { background: linear-gradient(45deg, #ff4b4b, #cc0000); color: white; animation: pulse-red 2s infinite; }
    .active-orange { background: linear-gradient(45deg, #ffa500, #cc8400); color: white; animation: pulse-orange 2s infinite; }
    .ok-green { background: linear-gradient(45deg, #28a745, #1e7e34); color: white; }
    .warn-yellow { background: linear-gradient(45deg, #ffc107, #d39e00); color: #212529; }
    .footer { margin-top: 40px; text-align: center; color: #d1d5db; padding: 15px; border-top: 1px solid rgba(255, 255, 255, 0.1); }
    .footer a { color: #00ffff; text-decoration: none; font-weight: bold; margin: 0 5px; }
    @media (max-width: 768px) { .main-header h1 { font-size: 2rem; } .status-card { font-size: 0.9rem; padding: 10px; } }
    @media (min-width: 1600px) { .main-header h1 { font-size: 4rem; } .status-card { font-size: 1.3rem; padding: 20px; } }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>📚 Smart Study Monitor</h1>
    <p>AI-Powered Neural Focus System</p>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("## 🛠️ Settings")
    sleep_threshold = st.slider("Eye Closure Sensitivity", 0.05, 0.5, 0.2, 0.01)
    phone_threshold = st.slider("Phone Detection Confidence", 0.0, 1.0, 0.4, 0.05)
    st.info("💡 Tap/Click Start to allow camera & audio.")

# --- LIGHTWEIGHT MODELS (No Hand Tracking for Cloud Speed) ---
@st.cache_resource
def load_models():
    face_detector = FaceMeshDetector(maxFaces=1)
    yolo_model = YOLO("yolov8n.pt") # Fast model for cloud
    return face_detector, yolo_model

# --- VIDEO PROCESSOR (Optimized for Cloud) ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector, self.model = load_models()
        self.status = {'sleep': False, 'cover': False, 'phone': False, 'face': False}
        self.audio_state = None
        self.frame_count = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # 1. Face & Sleep Detection (Runs every frame - it's fast)
        img, faces = self.face_detector.findFaceMesh(img, draw=False)
        
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
        else:
            # Face Cover heuristic: If no face is detected, it's likely covered
            # (Temporarily disabled to prevent false positives)
            is_face_covered = False

        # 2. Phone Detection (Runs every 2nd frame to save CPU & prevent lag)
        if self.frame_count % 2 == 0:
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

        self.status = {'sleep': is_sleepy, 'cover': is_face_covered, 'phone': is_phone, 'face': face_visible}
        
        if is_sleepy: self.audio_state = 'sleep'
        elif is_face_covered: self.audio_state = 'cover'
        elif is_phone: self.audio_state = 'phone'
        else: self.audio_state = None

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Live Neural Feed")
    with st.container(border=True):
        # IMPORTANT: Removed custom iceServers to let Streamlit handle TURN/STUN automatically
        ctx = webrtc_streamer(key="smart-study-monitor", video_processor_factory=VideoProcessor)

with col2:
    st.markdown("### 🟢 Status Dashboard")
    with st.container(border=True):
        status_sleep = st.empty()
        status_face = st.empty()
        status_phone = st.empty()
        status_general = st.empty()
        audio_placeholder = st.empty()

# --- STABLE UI UPDATE LOOP (NO AUTOREFRESH, NO CLOSING) ---
if ctx.video_processor:
    proc = ctx.video_processor
    
    # Loop to update statuses without reloading the page
    while True:
        if proc:
            # Update Status Cards
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

            # Update Audio (Uses session state to avoid reloads)
            if proc.audio_state == 'sleep':
                if st.session_state.get("audio_playing") != 'sleep':
                    with open("alarm.mp3", "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    audio_placeholder.markdown(f'<audio autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
                    st.session_state["audio_playing"] = 'sleep'
            elif proc.audio_state == 'cover':
                if st.session_state.get("audio_playing") != 'cover':
                    with open("faudio.mp3", "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    audio_placeholder.markdown(f'<audio autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
                    st.session_state["audio_playing"] = 'cover'
            elif proc.audio_state == 'phone':
                if st.session_state.get("audio_playing") != 'phone':
                    with open("paudio.mp3", "rb") as f:
                        b64 = base64.b64encode(f.read()).decode()
                    audio_placeholder.markdown(f'<audio autoplay loop><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)
                    st.session_state["audio_playing"] = 'phone'
            else:
                if st.session_state.get("audio_playing") is not None:
                    audio_placeholder.empty()
                    st.session_state["audio_playing"] = None

            status_general.markdown("### ⚡ System Running...")
        
        time.sleep(1) # Updates UI every 1 second, but never closes the page
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