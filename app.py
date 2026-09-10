import streamlit as st
import cv2
import cvzone
import math
import av
import time
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
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
    
    st.markdown("---")
    # Audio Unlock Button (Crucial for Browsers)
    if st.button("🔊 Enable Alarm Sounds", use_container_width=True):
        st.session_state["audio_enabled"] = True
        st.rerun()
    
    if st.session_state.get("audio_enabled"):
        st.success("✅ Sound Enabled")
    else:
        st.warning("⚠️ Click button above to enable sound")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    face_detector = FaceMeshDetector(maxFaces=1)
    hand_detector = HandDetector(detectionCon=0.8, maxHands=2)
    yolo_model = YOLO("yolov8n.pt")
    return face_detector, hand_detector, yolo_model

# --- VIDEO PROCESSOR (WebRTC - Smooth 30fps) ---
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_detector, self.hand_detector, self.model = load_models()
        self.status = {'sleep': False, 'cover': False, 'phone': False, 'face': False}
        self.frame_count = 0
        self.phone_lock = 0
        self.cover_lock = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        # Resize for faster AI processing
        h, w, _ = img.shape
        img_small = cv2.resize(img, (320, 240))
        sx, sy = w / 320, h / 240

        faces = []
        hands = []

        # Run AI every 2nd frame for smoothness
        if self.frame_count % 2 == 0:
            img_small, faces = self.face_detector.findFaceMesh(img_small, draw=False)
            hands, _ = self.hand_detector.findHands(img_small, draw=False)

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

        # Face Cover Logic
        if not face_visible and hands:
            is_face_covered = True
            self.cover_lock = time.time() + 1.5
        
        if time.time() < self.cover_lock:
            is_face_covered = True

        # Phone Detection (Runs every 4th frame on a slightly larger image)
        if self.frame_count % 4 == 0:
            img_phone = cv2.resize(img, (480, 320))
            results = self.model(img_phone, stream=True)
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls == 67 and conf > phone_threshold:
                        is_phone = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Scale coordinates back to original size
                        x1, y1, x2, y2 = int(x1 * (w/480)), int(y1 * (h/320)), int(x2 * (w/480)), int(y2 * (h/320))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        cvzone.putTextRect(img, "PHONE DETECTED", (x1, y1 - 10), scale=2, colorR=(0, 255, 255))
                        self.phone_lock = time.time() + 2.0

        if time.time() < self.phone_lock:
            is_phone = True

        self.status = {'sleep': is_sleepy, 'cover': is_face_covered, 'phone': is_phone, 'face': face_visible}
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- LAYOUT ---
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📹 Live Neural Feed")
    with st.container(border=True):
        ctx = webrtc_streamer(key="smart-study-monitor", video_processor_factory=VideoProcessor)

with col2:
    st.markdown("### 🟢 Status Dashboard")
    with st.container(border=True):
        status_sleep = st.empty()
        status_face = st.empty()
        status_phone = st.empty()
        status_general = st.empty()
        audio_placeholder = st.empty()

# --- NON-BLOCKING UI UPDATE (Updates every 1 second without freezing video) ---
if ctx.video_processor:
    @st.fragment(run_every=1)
    def update_dashboard():
        proc = ctx.video_processor
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
                status_phone.markdown('<div class="status-card active-orange">📱 PHONE DETECTED!</div>', unsafe_allow_html=True)
            else:
                status_phone.markdown('<div class="status-card ok-green">📵 No Phone</div>', unsafe_allow_html=True)

            # AUDIO LOGIC (Only plays if button was clicked)
            if st.session_state.get("audio_enabled"):
                if proc.status['sleep']:
                    if st.session_state.get("audio_playing") != 'sleep':
                        audio_placeholder.audio("alarm.mp3", format="audio/mp3", autoplay=True, loop=True)
                        st.session_state["audio_playing"] = 'sleep'
                elif proc.status['cover']:
                    if st.session_state.get("audio_playing") != 'face':
                        audio_placeholder.audio("faudio.mp3", format="audio/mp3", autoplay=True, loop=True)
                        st.session_state["audio_playing"] = 'face'
                elif proc.status['phone']:
                    if st.session_state.get("audio_playing") != 'phone':
                        audio_placeholder.audio("paudio.mp3", format="audio/mp3", autoplay=True, loop=True)
                        st.session_state["audio_playing"] = 'phone'
                else:
                    if st.session_state.get("audio_playing") is not None:
                        audio_placeholder.empty()
                        st.session_state["audio_playing"] = None

            status_general.markdown("### ⚡ System Running...")
    
    update_dashboard()
else:
    st.info("👈 Press **Start** on the video player, then click **Enable Alarm Sounds** in the sidebar.")

# --- FOOTER ---
st.markdown("""
<div class="footer">
    Made with ❤️ by <strong>Rohit Dixit</strong> | 
    <a href="https://www.linkedin.com/in/rohitdixitcs/" target="_blank">LinkedIn</a> | 
    <a href="https://rohitdixitdev.vercel.app/" target="_blank">Portfolio</a>
</div>
""", unsafe_allow_html=True)